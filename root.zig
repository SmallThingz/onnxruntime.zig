const std = @import("std");
const builtin = @import("builtin");

/// General Utils N stuff
pub const Utils = struct {
  pub const PathChar = if (builtin.os.tag == .windows) u16 else u8;
  comptime { std.debug.assert(PathChar == Api.c.ORTCHAR_T); }
  pub const Path = [*:0]const PathChar;

  const VEnum = enum {usize, cstr};
  fn OptionsKVLRetvalType(I: type, vEnum: VEnum) type {
    return struct {
      const T = if (@typeInfo(I) == .pointer) @typeInfo(I).pointer.child else I;
      const V = switch (vEnum) {
        .usize => usize,
        .cstr => [*:0]const u8,
      };
      const max_len = @typeInfo(T).@"struct".fields.len;

      _keys: [max_len][*:0]const u8 = undefined,
      _vals: [max_len]V = undefined,
      len: usize = 0,

      fn add(self: *@This(), options: I, comptime k: [:0]const u8) void {
        self._keys[self.len] = k;
        const v = @field(options, k);
        if (vEnum == .usize) {
          self._vals[self.len] = switch (@typeInfo(@TypeOf(v))) {
            .optional => |oi| switch (@typeInfo(oi.child)) {
              .int => v.?,
              else => @compileError("unreachable"),
            },
            .@"enum" => @intFromEnum(v),
            .int => v,
            else => @compileError("unreachable"),
          };
        } else {
          self._vals[self.len] = switch (@typeInfo(@TypeOf(v))) {
            .optional => |oi| switch (@typeInfo(oi.child)) {
              .pointer => |pi| switch (pi.size) {
                .one, .c, .many => @intFromPtr(v.?),
                .slice => @intFromPtr(v.?.ptr),
              },
              else => @compileError("unreachable"),
            },
            .pointer => |pi| switch (pi.size) {
              .one, .c, .many => @intFromPtr(v),
              .slice => @intFromPtr(v.ptr),
            },
            else => @compileError("unreachable"),
          };
        }
        self.len += 1;
      }

      fn fromInstance(instance: I) @This() {
        var retval: @This() = .{};
        inline for (@typeInfo(T).@"struct".fields) |f| {
          if (@field(instance, f.name) != @as(*const @FieldType(T, f.name), @alignCast(@ptrCast(f.default_value_ptr.?))).*)
            retval.add(instance, f.name);
        }
        return retval;
      }

      pub fn keys(self: *const @This()) [*]const [*:0] const u8 { return @ptrCast(&self._keys); }
      pub fn vals(self: *const @This()) [*]const V { return @ptrCast(&self._vals); }
    };
  }

  /// Helper function to convert a struct to keys, values array and length, for V2 functions that take string arrays
  pub fn createOptionsKVL(instance: anytype, comptime V: VEnum) OptionsKVLRetvalType(@TypeOf(instance), V) {
    return .fromInstance(instance);
  }

  pub fn CopyPointerAttrs(From: type, size: std.builtin.Type.Pointer.Size, To: type) type {
    const info = @typeInfo(From).pointer;
    return @Type(.{
      .pointer = .{
        .size = size,
        .is_const = info.is_const,
        .is_volatile = info.is_volatile,
        .is_allowzero = info.is_allowzero,
        .alignment = info.alignment,
        .address_space = info.address_space,
        .child = To,
        .sentinel_ptr = null,
      },
    });
  }
};

/// This is usually used by vendors
pub const Ep = struct {
  pub const api = opaque {
    pub var underlying: *const Api.c.OrtEpApi = undefined;
  };

  /// The Wrapper for OrtEp Struct
  pub const Interface = struct {
    underlying: Api.c.OrtEp,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

    // Call the underlying vtable
    pub fn getName(self: *const @This()) [*:0]const u8 {
      return self.underlying.GetName.?(@ptrCast(self));
    }

    // Call the underlying vtable
    pub fn setDynamicOptions(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (self.underlying.SetDynamicOptions) |f| {
        try Error.check(f(@ptrCast(self), @ptrCast(keys.ptr), @ptrCast(values.ptr), keys.len));
      }
    }

    // Call the underlying vtable
    pub fn getCompiledModelCompatibilityInfo(self: *@This(), graph: *const Graph) [*:0]const u8 {
      return self.underlying.GetCompiledModelCompatibilityInfo.?(@ptrCast(self), @ptrCast(graph));
    }

    pub const DataLayout = enum(c_uint) {
      NCHW = @bitCast(Api.c.OrtEpDataLayout_NCHW),
      NHWC = @bitCast(Api.c.OrtEpDataLayout_NHWC),
    };

    /// Initialize the Ep structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer (T):
    /// - Must have a field `ep: Ep`.
    /// - fn getName(self: *const T) [*:0]const u8
    /// - fn getCapability(self: *T, graph: *const Graph, support_info: *GraphSupportInfo) !void
    /// - fn compile(self: *T, graphs: []const *const Graph, fused_nodes: []const *const Node, node_compute_infos: []*NodeCompute.Info, ep_context_nodes: []*Node) !void
    /// - fn releaseNodeComputeInfos(self: *T, node_compute_infos: []*NodeCompute.Info) void
    /// - fn getCompiledModelCompatibilityInfo(self: *T, graph: *const Graph) [*:0]const u8
    ///
    /// Optional methods for Implementer (T):
    /// - fn getPreferredDataLayout(self: *T) !DataLayout
    /// - fn shouldConvertDataLayoutForOp(self: *T, domain: [*:0]const u8, op_type: [*:0]const u8, layout: DataLayout) i32 (1: conv, 0: no, -1: let ORT decide)
    /// - fn setDynamicOptions(self: *T, keys: []const [*:0]const u8, values: []const [*:0]const u8) !void
    /// - fn onRunStart(self: *T, options: *const RunOptions) !void
    /// - fn onRunEnd(self: *T, options: *const RunOptions, sync_stream: bool) !void
    /// - fn createAllocator(self: *T, info: *const Allocator.MemoryInfo) !?*Allocator
    /// - fn createSyncStreamForDevice(self: *T, device: *const Allocator.MemoryDevice) !?*SyncStreamImpl
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(ptr: ?*const Api.c.OrtEp) *T {
          return @fieldParentPtr("ep", @as(*Ep, @constCast(@ptrCast(ptr.?))));
        }

        fn getName(ptr: ?*const Api.c.OrtEp) callconv(.c) [*:0]const u8 {
          return getSelf(ptr).getName();
        }

        fn getCapability(ptr: ?*Api.c.OrtEp, graph: ?*const Api.c.OrtGraph, info: ?*Api.c.OrtEpGraphSupportInfo) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).getCapability(@ptrCast(graph.?), @ptrCast(info.?)) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          return null;
        }

        fn compile(
          ptr: ?*Api.c.OrtEp,
          graphs: [*c]?*const Api.c.OrtGraph,
          fused: [*c]?*const Api.c.OrtNode,
          count: usize,
          infos_out: [*c]?*Api.c.OrtNodeComputeInfo,
          context_nodes_out: [*c]?*Api.c.OrtNode,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const g_slice = @as([*]const *const Graph, @ptrCast(graphs))[0..count];
          const f_slice = @as([*]const *const Node, @ptrCast(fused))[0..count];
          const i_slice = @as([*]*NodeCompute.Info, @ptrCast(infos_out))[0..count];
          const c_slice = @as([*]*Node, @ptrCast(context_nodes_out))[0..count];

          getSelf(ptr).compile(g_slice, f_slice, i_slice, c_slice) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          return null;
        }

        fn releaseNodeComputeInfos(ptr: ?*Api.c.OrtEp, infos: [*c]?*Api.c.OrtNodeComputeInfo, count: usize) callconv(.c) void {
          getSelf(ptr).releaseNodeComputeInfos(@as([*]*NodeCompute.Info, @ptrCast(infos))[0..count]);
        }

        fn getPreferredDataLayout(ptr: ?*Api.c.OrtEp, layout_out: ?*Api.c.OrtEpDataLayout) callconv(.c) ?*Api.c.OrtStatus {
          const layout = getSelf(ptr).getPreferredDataLayout() catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          layout_out.?.* = @intFromEnum(layout);
          return null;
        }

        fn shouldConvertDataLayoutForOp(ptr: ?*Api.c.OrtEp, domain: [*:0]const u8, op_type: [*:0]const u8, layout: Api.c.OrtEpDataLayout, out: ?*c_int) callconv(.c) ?*Api.c.OrtStatus {
          out.?.* = getSelf(ptr).shouldConvertDataLayoutForOp(domain, op_type, @enumFromInt(layout));
          return null;
        }

        fn setDynamicOptions(ptr: ?*Api.c.OrtEp, keys: [*c]const [*:0]const u8, vals: [*c]const [*:0]const u8, count: usize) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).setDynamicOptions(keys[0..count], vals[0..count]) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          return null;
        }

        fn onRunStart(ptr: ?*Api.c.OrtEp, options: ?*const Api.c.OrtRunOptions) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).onRunStart(@ptrCast(options.?)) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          return null;
        }

        fn onRunEnd(ptr: ?*Api.c.OrtEp, options: ?*const Api.c.OrtRunOptions, sync: bool) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).onRunEnd(@ptrCast(options.?), sync) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          return null;
        }

        fn createAllocator(ptr: ?*Api.c.OrtEp, info: ?*const Api.c.OrtMemoryInfo, out: ?*?*Api.c.OrtAllocator) callconv(.c) ?*Api.c.OrtStatus {
          const alloc = getSelf(ptr).createAllocator(@ptrCast(info.?)) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          out.?.* = @ptrCast(alloc);
          return null;
        }

        fn createSyncStreamForDevice(ptr: ?*Api.c.OrtEp, dev: ?*const Api.c.OrtMemoryDevice, out: ?*?*Api.c.OrtSyncStreamImpl) callconv(.c) ?*Api.c.OrtStatus {
          const stream = getSelf(ptr).createSyncStreamForDevice(@ptrCast(dev.?)) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          out.?.* = @ptrCast(stream);
          return null;
        }

        fn getCompiledModelCompatibilityInfo(ptr: ?*Api.c.OrtEp, graph: ?*const Api.c.OrtGraph) callconv(.c) [*:0]const u8 {
          return getSelf(ptr).getCompiledModelCompatibilityInfo(@ptrCast(graph.?));
        }
      };

      return .{
        .underlying = .{
          .ort_version_supported = Api.c.ORT_API_VERSION,
          .GetName = VTable.getName,
          .GetCapability = VTable.getCapability,
          .Compile = VTable.compile,
          .ReleaseNodeComputeInfos = VTable.releaseNodeComputeInfos,
          .GetPreferredDataLayout = if (@hasDecl(T, "getPreferredDataLayout")) VTable.getPreferredDataLayout else null,
          .ShouldConvertDataLayoutForOp = if (@hasDecl(T, "shouldConvertDataLayoutForOp")) VTable.shouldConvertDataLayoutForOp else null,
          .SetDynamicOptions = if (@hasDecl(T, "setDynamicOptions")) VTable.setDynamicOptions else null,
          .OnRunStart = if (@hasDecl(T, "onRunStart")) VTable.onRunStart else null,
          .OnRunEnd = if (@hasDecl(T, "onRunEnd")) VTable.onRunEnd else null,
          .CreateAllocator = if (@hasDecl(T, "createAllocator")) VTable.createAllocator else null,
          .CreateSyncStreamForDevice = if (@hasDecl(T, "createSyncStreamForDevice")) VTable.createSyncStreamForDevice else null,
          .GetCompiledModelCompatibilityInfo = VTable.getCompiledModelCompatibilityInfo,
        },
      };
    }
  };

  /// Struct that an EP implements for IDataTransfer to copy between devices it uses and CPU
  pub const DataTransfer = struct {
    underlying: Api.c.OrtDataTransferImpl,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

    /// Check if the implementation can copy between the source and destination memory devices.
    /// returns true if the implementation can copy between the devices.
    /// since Version 1.23.
    pub fn canCopy(self: *const @This(), src: *const Allocator.MemoryDevice, dst: *const Allocator.MemoryDevice) bool {
      return self.underlying.CanCopy.?(@ptrCast(self), @ptrCast(src), @ptrCast(dst));
    }

    /// Copy tensors from src_tensors to dst_tensors using the provided streams.
    ///
    /// The implementation can use the provided streams to perform asynchronous copies if supported.
    /// If a stream is not available, the copy is performed synchronously.
    ///
    /// streams: Array of OrtSyncStream pointers for the copy operations, if the execution provider is stream aware. nullptr if it is not.
    ///
    /// since Version 1.23.
    pub fn copy(self: *@This(), src: []*const Value, dst: []*Value, streams: ?[]*SyncStream) !void {
      std.debug.assert(src.len == dst.len);
      if (streams) |s| std.debug.assert(src.len == s.len);
      try Error.check(self.underlying.CopyTensors.?(
          @ptrCast(self),
          @ptrCast(src.ptr),
          @ptrCast(dst.ptr),
          @ptrCast(if (streams) |s| s.ptr else null),
          if (streams) |s| s.len else 0
      ));
    }

    /// Release the OrtDataTransferImpl instance.
    ///
    /// This is called by ORT when the OrtDataTransferImpl instance is no longer needed.
    /// The implementation should release any resources held by the instance.
    ///
    /// since Version 1.23.
    pub fn deinit(self: *@This()) void {
      return self.underlying.Release.?(@ptrCast(self));
    }

    /// Initialize the DataTransfer structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer:
    /// - Must have a field `data_transfer: DataTransfer` as the a member, we use @fieldParentPtr on that member to get actual pointer
    /// - fn canCopy(self: *const Implementer, src: *const Allocator.MemoryDevice, dst: *const Allocator.MemoryDevice) bool
    /// - fn copyTensors(self: *Implementer, src: []const *const Value, dst: []*Value, streams: ?[]*SyncStream) ?*Error.Status
    /// - (Optional) fn deinit(self: *Implementer) void
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(self: ?*Api.c.OrtDataTransferImpl) *T {
          return @fieldParentPtr("data_transfer", @as(@FieldType(T, "data_transfer"), @alignCast(@ptrCast(self))));
        }

        fn release(self: ?*Api.c.OrtDataTransferImpl) callconv(.c) void {
          if (@hasDecl(T, "deinit")) getSelf(self).deinit();
        }

        fn canCopy(
          self: ?*const Api.c.OrtDataTransferImpl,
          src: ?*const Api.c.OrtMemoryDevice,
          dst: ?*const Api.c.OrtMemoryDevice,
        ) callconv(.c) bool {
          return getSelf(self).canCopy(@as(*const Allocator.MemoryDevice, @ptrCast(src.?)), @as(*const Allocator.MemoryDevice, @ptrCast(dst.?)));
        }

        fn copyTensors(
          self: ?*Api.c.OrtDataTransferImpl,
          src: [*c]?*const Api.c.OrtValue,
          dst: [*c]?*Api.c.OrtValue,
          streams: [*c]?*Api.c.OrtSyncStream,
          num: usize,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const src_zig = @as([*]const *const Value, @ptrCast(src))[0..num];
          const dst_zig = @as([*]*Value, @ptrCast(dst))[0..num];
          const streams_zig: ?[]*SyncStream = if (streams != null) @as([*]*SyncStream, @ptrCast(streams))[0..num] else null;

          return @ptrCast(getSelf(self).copyTensors(src_zig, dst_zig, streams_zig));
        }
      };

      return .{
        .underlying = .{
          .ort_version_supported = Api.c.ORT_API_VERSION,
          .Release = VTable.release,
          .CanCopy = VTable.canCopy,
          .CopyTensors = VTable.copyTensors,
        },
      };
    }
  };

  /// Struct that an EP implements for Stream Notifications.
  pub const SyncNotificationImpl = struct {
    underlying: Api.c.OrtSyncNotificationImpl,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

    /// Called by ORT to activate the notification.
    ///
    /// since Version 1.23.
    pub fn activate(self: *@This()) !void {
      try Error.check(self.underlying.Activate.?(@ptrCast(self)));
    }

    /// Wait for a device to device operation to complete.
    ///
    /// this_ptr: Pointer to the OrtSyncNotificationImpl instance.
    /// stream: The OrtSyncStream instance that will wait on this notification to be activated.
    ///
    /// since Version 1.23.
    pub fn waitOnDevice(self: *@This(), stream: *SyncStream) !void {
      try Error.check(self.underlying.WaitOnDevice.?(@ptrCast(self), @ptrCast(stream)));
    }

    /// Wait for a device to host operation to complete.
    ///
    /// since Version 1.23.
    pub fn waitOnHost(self: *@This()) !void {
      try Error.check(self.underlying.WaitOnHost.?(@ptrCast(self)));
    }

    /// Release the OrtSyncNotificationImpl instance.
    ///
    /// This is called by ORT when the OrtSyncNotificationImpl instance is no longer needed.
    /// The implementation should release any resources held by the instance.
    ///
    /// since Version 1.23.
    pub fn deinit(self: *@This()) void {
      self.underlying.Release.?(@ptrCast(self));
    }

    /// Initialize the SyncNotification structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer:
    /// - Must have a field `sync_notification: SyncNotificationImpl` as a member.
    /// - fn activate(self: *Implementer) ?*Error.Status
    /// - fn waitOnDevice(self: *Implementer, consumer_stream: *SyncStream) ?*Error.Status
    /// - fn waitOnHost(self: *Implementer) ?*Error.Status
    /// - (Optional) fn deinit(self: *Implementer) void
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(self: ?*Api.c.OrtSyncNotificationImpl) *T {
          return @fieldParentPtr("sync_notification", @as(*SyncNotificationImpl, @ptrCast(self.?)));
        }

        fn release(self: ?*Api.c.OrtSyncNotificationImpl) callconv(.c) void {
          if (@hasDecl(T, "deinit")) getSelf(self).deinit();
        }

        fn activate(self: ?*Api.c.OrtSyncNotificationImpl) callconv(.c) ?*Api.c.OrtStatus {
          return @ptrCast(getSelf(self).activate());
        }

        fn waitOnDevice(self: ?*Api.c.OrtSyncNotificationImpl, stream: ?*Api.c.OrtSyncStream) callconv(.c) ?*Api.c.OrtStatus {
          return @ptrCast(getSelf(self).waitOnDevice(@as(*SyncStream, @ptrCast(stream.?))));
        }

        fn waitOnHost(self: ?*Api.c.OrtSyncNotificationImpl) callconv(.c) ?*Api.c.OrtStatus {
          return @ptrCast(getSelf(self).waitOnHost());
        }
      };

      return .{
        .underlying = .{
          .ort_version_supported = Api.c.ORT_API_VERSION,
          .Release = VTable.release,
          .Activate = VTable.activate,
          .WaitOnDevice = VTable.waitOnDevice,
          .WaitOnHost = VTable.waitOnHost,
        },
      };
    }
  };

  /// Struct that an EP implements if it wishes to implement Stream support.
  /// This struct provides the overrides for onnxruntime::Stream's virtual methods.
  ///
  /// since Version 1.23.
  pub const SyncStreamImpl = struct {
    underlying: Api.c.OrtSyncStreamImpl,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

    ///Get the handle of the stream.
    ///
    /// This returns the native handle for the stream. e.g. cudaStream_t for CUDA streams.
    ///
    /// since Version 1.23.
    pub fn handle(self: *@This()) *anyopaque {
      return self.underlying.GetHandle.?(@ptrCast(self)).?;
    }

    /// Create an OrtSyncNotificationImpl for the OrtSyncStreamImpl instance.
    ///
    /// since Version 1.23.
    pub fn createNotification(self: *@This()) !*SyncNotificationImpl {
      var out: ?*SyncNotificationImpl = null;
      try Error.check(self.underlying.CreateNotification.?(@ptrCast(self), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Notify the stream that a session run has ended.
    ///
    /// This is called by ORT to notify the stream that a session run has ended, allowing the stream to perform any
    /// necessary cleanup or finalization.
    ///
    /// since Version 1.23.
    pub fn endSessionRun(self: *@This()) !void {
      try Error.check(self.underlying.OnSessionRunEnd.?(@ptrCast(self)));
    }

    /// Flush the stream.
    ///
    /// This is called by ORT to flush the stream, ensuring that all operations submitted to the stream are completed.
    ///
    /// since Version 1.23.
    pub fn flush(self: *@This()) !void {
      try Error.check(self.underlying.Flush.?(@ptrCast(self)));
    }

    /// Release the OrtSyncStreamImpl instance.
    ///
    /// This is called by ORT when the OrtSyncStreamImpl instance is no longer needed.
    /// The implementation should release any resources held by the instance.
    ///
    /// since Version 1.23.
    pub fn deinit(self: *@This()) void {
      self.underlying.Release.?(@ptrCast(self));
    }

    /// Requirements for Implementer:
    /// - Must have a field `sync_stream: SyncStreamImpl`.
    /// - fn getHandle(self: *Implementer) ?*anyopaque
    /// - fn createNotification(self: *Implementer, out: **SyncNotificationImpl) ?*Error.Status
    /// - fn flush(self: *Implementer) ?*Error.Status
    /// - fn onSessionRunEnd(self: *Implementer) ?*Error.Status
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(self: ?*Api.c.OrtSyncStreamImpl) *T {
          return @fieldParentPtr("sync_stream", @as(*SyncStreamImpl, @ptrCast(self.?)));
        }

        fn release(self: ?*Api.c.OrtSyncStreamImpl) callconv(.c) void {
          if (@hasDecl(T, "deinit")) getSelf(self).deinit();
        }

        fn getHandle(self: ?*Api.c.OrtSyncStreamImpl) callconv(.c) ?*anyopaque {
          return getSelf(self).getHandle();
        }

        fn createNotification(self: ?*Api.c.OrtSyncStreamImpl, out: ?*?*Api.c.OrtSyncNotificationImpl) callconv(.c) ?*Api.c.OrtStatus {
          return @ptrCast(getSelf(self).createNotification(@as(*?*SyncNotificationImpl, @ptrCast(out.?))));
        }

        fn flush(self: ?*Api.c.OrtSyncStreamImpl) callconv(.c) ?*Api.c.OrtStatus {
          return @ptrCast(getSelf(self).flush());
        }

        fn onSessionRunEnd(self: ?*Api.c.OrtSyncStreamImpl) callconv(.c) ?*Api.c.OrtStatus {
          return @ptrCast(getSelf(self).onSessionRunEnd());
        }
      };

      return .{
        .underlying = .{
          .ort_version_supported = Api.c.ORT_API_VERSION,
          .Release = VTable.release,
          .GetHandle = VTable.getHandle,
          .CreateNotification = VTable.createNotification,
          .Flush = VTable.flush,
          .OnSessionRunEnd = VTable.onSessionRunEnd,
        },
      };
    }
  };

  pub const NodeFusionOptions = Api.c.OrtNodeFusionOptions;

  pub const NodeCompute = struct {
    /// Opaque to add type to createState
    pub const State = opaque {};

    pub const Context = opaque {
      /// Query a OrtNodeComputeContext for the name of the node that encapsulates the compiled/fused node.
      ///
      /// Used in OrtNodeComputeInfo::CreateComputeState().
      ///
      /// context The OrtNodeComputeContext instance to query.
      /// returns The node's name.
      ///
      /// Note: Returned string is owned by ORT and valid only while OrtNodeComputeInfo::CreateComputeState() is called.
      ///
      /// since Version 1.23.
      pub fn name(self: *const @This()) [*:0]const u8 {
        return api.underlying.NodeComputeContext_NodeName.?(@ptrCast(self));
      }
    };

    /// The OrtNodeComputeInfo struct provides functions that an OrtEp implements to specify the compute function for a compiled OrtGraph instance.
    /// since Version 1.23.
    pub const Info = struct {
      underlying: Api.c.OrtNodeComputeInfo,
      comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

      /// Creates an opaque compute state object that is then passed to the Compute() function during inference.
      /// compute_context OrtNodeComputeContext instance that contains compiled/fused node's name and host
      ///                 memory allocation functions. Can optionally be used to build the compute state.
      /// returns: compute_state: the opaque computation state. ONNX Runtime calls ReleaseState() (after calling Compute())
      ///          to allow the implementer to release the compute state.
      ///
      /// since Version 1.23.
      pub fn createState(self: *@This(), context: *Context) !*State {
        var out: ?*State = null;
        try Error.check(self.underlying.CreateState.?(@ptrCast(self), @ptrCast(context), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Computation function called to execute the fused node compiled by an OrtEp instance.
      ///
      /// compute_state: The opaque computation state returned by CreateState().
      /// kernel_context: The OrtKernelContext instance used to access inputs/outputs.
      ///
      /// since Version 1.23.
      pub fn compute(self: *@This(), state: *State, kernel_context: *KernelContext) !void {
        try Error.check(self.underlying.Compute.?(@ptrCast(self), @ptrCast(state), @ptrCast(kernel_context)));
      }

      /// Releases the compute state returned by CreateState().
      ///
      /// compute_state: The opaque compute state returned by CreateState().
      ///
      /// since Version 1.23.
      pub fn releaseState(self: *@This(), state: *State) void {
        self.underlying.ReleaseState.?(@ptrCast(self), @ptrCast(state));
      }

      /// Initialize the NodeCompute.Info structure with vtables pointing to the provided Implementer type.
      ///
      /// Requirements for Implementer:
      /// - Must have a field `compute_info: Ep.NodeCompute.Info`.
      /// - fn createState(self: *Implementer, context: *Context) !*State
      /// - fn compute(self: *Implementer, state: *State, kernel_context: *KernelContext) !void
      /// - fn releaseState(self: *Implementer, state: *State) void
      pub fn init(comptime T: type) @This() {
        const VTable = struct {
          fn getSelf(self: ?*Api.c.OrtNodeComputeInfo) *T {
            return @fieldParentPtr("compute_info", @as(*Info, @ptrCast(self.?)));
          }

          fn createState(
            self: ?*Api.c.OrtNodeComputeInfo,
            ctx: ?*Api.c.OrtNodeComputeContext,
            state_out: ?*?*anyopaque,
          ) callconv(.c) ?*Api.c.OrtStatus {
            const result = getSelf(self).createState(@ptrCast(ctx.?)) catch |err| {
                return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
            };
            state_out.?.* = @ptrCast(result);
            return null;
          }

          fn compute(
            self: ?*Api.c.OrtNodeComputeInfo,
            state: ?*anyopaque,
            kernel_ctx: ?*Api.c.OrtKernelContext,
          ) callconv(.c) ?*Api.c.OrtStatus {
            getSelf(self).compute(@ptrCast(state.?), @ptrCast(kernel_ctx.?)) catch |err| {
                return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
            };
            return null;
          }

          fn releaseState(self: ?*Api.c.OrtNodeComputeInfo, state: ?*anyopaque) callconv(.c) void {
            getSelf(self).releaseState(@ptrCast(state.?));
          }
        };

        return .{
          .underlying = .{
            .ort_version_supported = Api.c.ORT_API_VERSION,
            .CreateState = VTable.createState,
            .Compute = VTable.compute,
            .ReleaseState = VTable.releaseState,
          },
        };
      }
    };
  };

  /// Holds information about the nodes supported by an Execution Provider.
  /// This is passed to the EP's GetCapability function.
  /// Wraps OrtEpGraphSupportInfo.
  pub const GraphSupportInfo = opaque {
    /// Specify nodes that are supported by an OrtEp and should be fused into one node.
    ///
    /// Because the nodes will be fused into one "fused node", there must not exist an unsupported node in
    /// a path between two of the provided nodes. Otherwise, the graph will become invalid.
    ///
    /// This function can be called multiple times. A subsequent call to this function will force the next set of
    /// nodes to be fused into a different node.
    ///
    /// graph_support_info OrtEpGraphSupportInfo instance to which to add the supported nodes.
    /// nodes Array of nodes supported by the EP that should be fused/compiled.
    /// num_nodes The number of supported nodes.
    /// node_fusion_options Optional node fusion options. Ignored if set to NULL.
    ///
    /// since Version 1.23.
    pub fn addNodesToFuse(
      self: *@This(),
      nodes: []const *const Node,
      options: ?*const NodeFusionOptions,
    ) !void {
      try Error.check(api.underlying.EpGraphSupportInfo_AddNodesToFuse.?(
          @ptrCast(self),
          @ptrCast(nodes.ptr),
          nodes.len,
          @ptrCast(options),
      ));
    }

    /// Specify a node that is supported by an OrtEp and should be run with a registered EP kernel.
    ///
    /// graph_support_info OrtEpGraphSupportInfo instance to which to add the supported node.
    /// node The supported OrtNode instance.
    ///
    /// since Version 1.23.
    pub fn addSingleNode(self: *@This(), node: *const Node) !void {
      try Error.check(api.underlying.EpGraphSupportInfo_AddSingleNode.?(
          @ptrCast(self),
          @ptrCast(node),
      ));
    }
  };


  // this is const everywhere
  /// Represents an instance of an Execution Provider mapped to a specific hardware device.
  /// Wraps OrtEpDevice.
  pub const Device = opaque {
    /// Create an OrtEpDevice for the EP and an OrtHardwareDevice.
    /// ep_factory Execution provider factory that is creating the instance.
    /// hardware_device Hardware device that the EP can utilize.
    /// ep_metadata (Optional) OrtKeyValuePairs instance for execution provider metadata that may be used
    ///             during execution provider selection and passed to CreateEp.
    ///             ep_device will copy this instance and the user should call ReleaseKeyValuePairs.
    /// ep_options (Optional) OrtKeyValuePairs instance for execution provider options that will be added
    ///            to the Session configuration options if the execution provider is selected.
    ///            ep_device will copy this instance and the user should call ReleaseKeyValuePairs.
    /// returns: ep_device OrtExecutionDevice that is created.
    ///
    /// since Version 1.22.
    pub fn init(
      factory: *Factory,
      hw_device: *const HardwareDevice,
      ep_metadata: ?*const KeyValuePairs,
      ep_options: ?*const KeyValuePairs
    ) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.CreateEpDevice.?(
          @ptrCast(factory),
          @ptrCast(hw_device),
          @ptrCast(ep_metadata),
          @ptrCast(ep_options),
          @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Returns the name of the execution provider (e.g., "CUDA", "CPU").
    /// Wraps OrtApi::EpDevice_EpName
    pub fn getEpName(self: *const @This()) [*:0]const u8 {
      return @ptrCast(Api.ort.EpDevice_EpName.?(@ptrCast(self)));
    }

    /// Returns the name of the execution provider's vendor.
    /// Wraps OrtApi::EpDevice_EpVendor
    pub fn getEpVendor(self: *const @This()) [*:0]const u8 {
      return @ptrCast(Api.ort.EpDevice_EpVendor.?(@ptrCast(self)));
    }

    /// Returns an OrtKeyValuePairs instance containing the metadata for the EP device.
    /// Note: ORT owns this instance; do NOT call deinit on it.
    /// Wraps OrtApi::EpDevice_EpMetadata
    pub fn getEpMetadata(self: *const @This()) *const KeyValuePairs {
      return @ptrCast(Api.ort.EpDevice_EpMetadata.?(@ptrCast(self)));
    }

    /// Returns an OrtKeyValuePairs instance containing the options for the EP device.
    /// Note: ORT owns this instance; do NOT call deinit on it.
    /// Wraps OrtApi::EpDevice_EpOptions
    pub fn getEpOptions(self: *const @This()) *const KeyValuePairs {
      return @ptrCast(Api.ort.EpDevice_EpOptions.?(@ptrCast(self)));
    }

    /// Returns the underlying HardwareDevice (physical CPU/GPU/NPU) instance for this EP device.
    /// Note: ORT owns this instance.
    /// Wraps OrtApi::EpDevice_Device
    pub fn getHardwareDevice(self: *const @This()) *const HardwareDevice {
      return @ptrCast(Api.ort.EpDevice_Device.?(@ptrCast(self)));
    }

    /// Get the OrtMemoryInfo for the device.
    /// If memory_type is DEFAULT and null is returned, the EP uses CPU memory.
    /// Wraps OrtApi::EpDevice_MemoryInfo
    pub fn getMemoryInfo(self: *const @This(), mem_type: Allocator.DeviceMemoryType) ?*const Allocator.MemoryInfo {
      const ptr = Api.ort.EpDevice_MemoryInfo.?(@ptrCast(self), @intFromEnum(mem_type));
      return @ptrCast(ptr);
    }

    /// Register an allocator with the OrtEpDevice.
    ///
    /// This allows an EP to provide OrtMemoryInfo for DEFAULT and HOST_ACCESSIBLE memory type as needed.
    /// The registered values will be used in calls to OrtEpFactory::CreateAllocator to ensure the required allocator/s
    /// are available for EP usage.
    ///
    /// Multiple calls for the same entry type will replace a previous entry.
    ///
    /// Available entries:
    ///   - OrtDeviceAllocator with type of OrtDeviceMemoryType_DEFAULT
    ///   - OrtDeviceAllocator with type of OrtDeviceMemoryType_HOST_ACCESSIBLE
    ///   - OrtReadOnlyAllocator with type of OrtDeviceMemoryType_DEFAULT
    ///     - if provided this allocator will only be used to copy initializers to the device the EP uses.
    ///       ORT will use the OrtDeviceAllocator if not provided.
    ///
    /// ep_device The OrtEpDevice instance to register the OrtMemoryInfo with.
    /// allocator_memory_info The OrtMemoryInfo information for the allocator.
    ///
    /// since Version 1.23.
    pub fn addAllocatorInfo(self: *@This(), info: *const Allocator.MemoryInfo) !void {
      try Error.check(api.underlying.EpDevice_AddAllocatorInfo.?(@ptrCast(self), @ptrCast(info)));
    }

    /// Releases the EpDevice instance. 
    /// Use only if you manually created the device via CreateEpDevice.
    /// Devices from Api.env.getEpDevices() are managed by the environment.
    pub fn deinit(self: *@This()) void {
      api.underlying.ReleaseEpDevice.?(@ptrCast(self));
    }
  };

  pub const Factory = struct {
    underlying: Api.c.OrtEpFactory,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

    pub fn getName(self: *const @This()) [*:0]const u8 {
      return self.underlying.GetName.?(@ptrCast(self));
    }

    pub fn getVersion(self: *const @This()) [*:0]const u8 {
      return self.underlying.GetVersion.?(@ptrCast(self));
    }

    pub fn validateCompatibility(self: *@This(), devices: []const *const HardwareDevice, info: [*:0]const u8) !Api.c.OrtCompiledModelCompatibility {
      var out: Api.c.OrtCompiledModelCompatibility = undefined;
      try Error.check(self.underlying.ValidateCompiledModelCompatibilityInfo.?(@ptrCast(self), @ptrCast(devices.ptr), devices.len, info, &out));
      return out;
    }

    /// Initialize the Factory structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer (T):
    /// - Must have a field `factory: Ep.Factory`.
    /// - fn getName(self: *const T) [*:0]const u8
    /// - fn getVendor(self: *const T) [*:0]const u8
    /// - fn getSupportedDevices(self: *T, devices: []const *const HardwareDevice, ep_devices_out: []*Ep.Device) !usize (return num added)
    /// - fn createEp(self: *T, devices: []const *const HardwareDevice, metadata: []const *const KeyValuePairs, options: *const Session.Options.C, logger: *const KernelInfo.Logger) !*Ep
    /// - fn releaseEp(self: *T, ep: *Ep) void
    /// - fn getVendorId(self: *const T) u32
    /// - fn getVersion(self: *const T) [*:0]const u8
    /// - fn validateCompiledModelCompatibilityInfo(self: *T, devices: []const *const HardwareDevice, info: [*:0]const u8) Api.c.OrtCompiledModelCompatibility
    ///
    /// Optional methods for Implementer (T):
    /// - fn createAllocator(self: *T, info: ?*const Allocator.MemoryInfo, options: ?*const KeyValuePairs) !?*Allocator
    /// - fn releaseAllocator(self: *T, allocator: *Allocator) void
    /// - fn createDataTransfer(self: *T) !?*Ep.DataTransfer
    /// - fn isStreamAware(self: *const T) bool
    /// - fn createSyncStreamForDevice(self: *T, device: *const Allocator.MemoryDevice, options: ?*const KeyValuePairs) !?*SyncStreamImpl
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(ptr: ?*const Api.c.OrtEpFactory) *T {
          return @fieldParentPtr("factory", @as(*Factory, @constCast(@ptrCast(ptr.?))));
        }

        fn getName(ptr: ?*const Api.c.OrtEpFactory) callconv(.c) [*:0]const u8 {
          return getSelf(ptr).getName();
        }

        fn getVendor(ptr: ?*const Api.c.OrtEpFactory) callconv(.c) [*:0]const u8 {
          return getSelf(ptr).getVendor();
        }

        fn getSupportedDevices(
          ptr: ?*Api.c.OrtEpFactory,
          devices: [*c]const ?*const Api.c.OrtHardwareDevice,
          num_devices: usize,
          ep_devices_out: [*c]?*Api.c.OrtEpDevice,
          max_ep_devices: usize,
          num_ep_devices_out: ?*usize,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const dev_slice = @as([*]const *const HardwareDevice, @ptrCast(devices))[0..num_devices];
          const ep_out_slice = @as([*]*Ep.Device, @ptrCast(ep_devices_out))[0..max_ep_devices];

          const count = getSelf(ptr).getSupportedDevices(dev_slice, ep_out_slice) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          num_ep_devices_out.?.* = count;
          return null;
        }

        fn createEp(
          ptr: ?*Api.c.OrtEpFactory,
          devices: [*c]const ?*const Api.c.OrtHardwareDevice,
          metadata: [*c]const ?*const Api.c.OrtKeyValuePairs,
          num_devices: usize,
          options: ?*const Api.c.OrtSessionOptions,
          logger: ?*const Api.c.OrtLogger,
          ep_out: ?*?*Api.c.OrtEp,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const dev_slice = @as([*]const *const HardwareDevice, @ptrCast(devices))[0..num_devices];
          const meta_slice = @as([*]const *const KeyValuePairs, @ptrCast(metadata))[0..num_devices];

          const ep = getSelf(ptr).createEp(dev_slice, meta_slice, @ptrCast(options.?), @ptrCast(logger.?)) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          ep_out.?.* = @ptrCast(ep);
          return null;
        }

        fn releaseEp(ptr: ?*Api.c.OrtEpFactory, ep: ?*Api.c.OrtEp) callconv(.c) void {
          getSelf(ptr).releaseEp(@ptrCast(ep.?));
        }

        fn getVendorId(ptr: ?*const Api.c.OrtEpFactory) callconv(.c) u32 {
          return getSelf(ptr).getVendorId();
        }

        fn getVersion(ptr: ?*const Api.c.OrtEpFactory) callconv(.c) [*:0]const u8 {
          return getSelf(ptr).getVersion();
        }

        fn validateCompatibility(
          ptr: ?*Api.c.OrtEpFactory,
          devices: [*c]const ?*const Api.c.OrtHardwareDevice,
          num_devices: usize,
          info: [*:0]const u8,
          out: ?*Api.c.OrtCompiledModelCompatibility,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const dev_slice = @as([*]const *const HardwareDevice, @ptrCast(devices))[0..num_devices];
          out.?.* = getSelf(ptr).validateCompiledModelCompatibilityInfo(dev_slice, info);
          return null;
        }

        fn createAllocator(
          ptr: ?*Api.c.OrtEpFactory,
          info: ?*const Api.c.OrtMemoryInfo,
          opts: ?*const Api.c.OrtKeyValuePairs,
          out: ?*?*Api.c.OrtAllocator,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const alloc = getSelf(ptr).createAllocator(@ptrCast(info), @ptrCast(opts)) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          out.?.* = @ptrCast(alloc);
          return null;
        }

        fn releaseAllocator(ptr: ?*Api.c.OrtEpFactory, alloc: ?*Api.c.OrtAllocator) callconv(.c) void {
          getSelf(ptr).releaseAllocator(@ptrCast(alloc.?));
        }

        fn createDataTransfer(ptr: ?*Api.c.OrtEpFactory, out: ?*?*Api.c.OrtDataTransferImpl) callconv(.c) ?*Api.c.OrtStatus {
          const dt = getSelf(ptr).createDataTransfer() catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          out.?.* = @ptrCast(dt);
          return null;
        }

        fn isStreamAware(ptr: ?*const Api.c.OrtEpFactory) callconv(.c) bool {
          return getSelf(ptr).isStreamAware();
        }

        fn createSyncStream(
          ptr: ?*Api.c.OrtEpFactory,
          dev: ?*const Api.c.OrtMemoryDevice,
          opts: ?*const Api.c.OrtKeyValuePairs,
          out: ?*?*Api.c.OrtSyncStreamImpl,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const stream = getSelf(ptr).createSyncStreamForDevice(@ptrCast(dev.?), @ptrCast(opts)) catch |err| {
            return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err)) catch null);
          };
          out.?.* = @ptrCast(stream);
          return null;
        }
      };

      return .{
        .underlying = .{
          .ort_version_supported = Api.c.ORT_API_VERSION,
          .GetName = VTable.getName,
          .GetVendor = VTable.getVendor,
          .GetSupportedDevices = VTable.getSupportedDevices,
          .CreateEp = VTable.createEp,
          .ReleaseEp = VTable.releaseEp,
          .GetVendorId = VTable.getVendorId,
          .GetVersion = VTable.getVersion,
          .ValidateCompiledModelCompatibilityInfo = VTable.validateCompatibility,
          .CreateAllocator = if (@hasDecl(T, "createAllocator")) VTable.createAllocator else null,
          .ReleaseAllocator = if (@hasDecl(T, "releaseAllocator")) VTable.releaseAllocator else null,
          .CreateDataTransfer = if (@hasDecl(T, "createDataTransfer")) VTable.createDataTransfer else null,
          .IsStreamAware = if (@hasDecl(T, "isStreamAware")) VTable.isStreamAware else null,
          .CreateSyncStreamForDevice = if (@hasDecl(T, "createSyncStreamForDevice")) VTable.createSyncStream_ else null,
        },
      };
    }
  };
};

const Training = struct {
  pub const api = struct {
    pub var underlying: *const Api.c.OrtTrainingApi = undefined;
  };

  pub const CheckpointState = opaque {
    /// Load a checkpoint state from a file on disk.
    pub fn load(path: [*:0]const u8) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.LoadCheckpoint.?(path, @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Load a checkpoint state from a buffer.
    pub fn loadFromBuffer(buffer: []const u8) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.LoadCheckpointFromBuffer.?(
        @ptrCast(buffer.ptr), 
        buffer.len, 
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Save the given state to a checkpoint file on disk.
    pub fn save(self: *const @This(), path: [*:0]const u8, include_optimizer_state: bool) !void {
      try Error.check(api.underlying.SaveCheckpoint.?(
        @ptrCast(self), 
        path, 
        include_optimizer_state
      ));
    }

    /// Adds or updates the given property in the checkpoint state.
    pub fn addProperty(self: *@This(), name: [*:0]const u8, prop_type: PropertyType, value: *anyopaque) !void {
      try Error.check(api.underlying.AddProperty.?(
        @ptrCast(self),
        name,
        @intFromEnum(prop_type),
        value
      ));
    }

    /// Gets the property value associated with the given name from the checkpoint state.
    /// allocator: Used to allocate the output buffer if the property is a string.
    pub fn getProperty(self: *const @This(), name: [*:0]const u8, allocator: *Allocator) !struct { type: PropertyType, value: *anyopaque } {
      var out_type: PropertyType = undefined;
      var out_val: ?*anyopaque = null;
      try Error.check(api.underlying.GetProperty.?(
        @ptrCast(self),
        name,
        @ptrCast(allocator),
        @ptrCast(&out_type),
        &out_val
      ));
      return .{ .type = @enumFromInt(out_type), .value = out_val orelse return error.OutOfMemory };
    }

    pub const PropertyType = enum(c_uint) {
      Int = @bitCast(Api.c.OrtIntProperty),
      Float = @bitCast(Api.c.OrtFloatProperty),
      String = @bitCast(Api.c.OrtStringProperty),
    };

    pub fn deinit(self: *@This()) void {
      api.underlying.ReleaseCheckpointState.?(@ptrCast(self));
    }
  };

  pub const TrainingSession = opaque {
    /// Create a training session that can be used to begin or resume training.
    /// 
    /// env: The environment.
    /// options: Session options.
    /// checkpoint: The checkpoint state.
    /// train_model_path: Path to the training model.
    /// eval_model_path: Optional path to the eval model.
    /// optimizer_model_path: Optional path to the optimizer model.
    pub fn init(
      options: *const Session.Options,
      checkpoint: *CheckpointState,
      train_model_path: [*:0]const u8,
      eval_model_path: ?[*:0]const u8,
      optimizer_model_path: ?[*:0]const u8,
    ) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.CreateTrainingSession.?(
        Api.env.underlying,
        @ptrCast(options),
        @ptrCast(checkpoint),
        train_model_path,
        eval_model_path,
        optimizer_model_path,
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Computes the outputs of the training model and the gradients of the trainable parameters
    /// for the given inputs.
    pub fn trainStep(
      self: *@This(),
      run_options: ?*const RunOptions,
      inputs: []const *const Value,
      outputs: []*Value,
    ) !void {
      try Error.check(api.underlying.TrainStep.?(
        @ptrCast(self),
        @ptrCast(run_options),
        inputs.len,
        @ptrCast(inputs.ptr),
        outputs.len,
        @ptrCast(outputs.ptr),
      ));
    }

    /// Computes the outputs for the eval model for the given inputs.
    pub fn evalStep(
      self: *@This(),
      run_options: ?*const RunOptions,
      inputs: []const *const Value,
      outputs: []*Value,
    ) !void {
      try Error.check(api.underlying.EvalStep.?(
        @ptrCast(self),
        @ptrCast(run_options),
        inputs.len,
        @ptrCast(inputs.ptr),
        outputs.len,
        @ptrCast(outputs.ptr),
      ));
    }

    /// Performs the weight updates for the trainable parameters using the optimizer model.
    pub fn optimizerStep(self: *@This(), run_options: ?*const RunOptions) !void {
      try Error.check(api.underlying.OptimizerStep.?(
        @ptrCast(self),
        @ptrCast(run_options)
      ));
    }

    /// Sets the learning rate for this training session.
    pub fn setLearningRate(self: *@This(), learning_rate: f32) !void {
      try Error.check(api.underlying.SetLearningRate.?(@ptrCast(self), learning_rate));
    }

    /// Gets the current learning rate for this training session.
    pub fn getLearningRate(self: *const @This()) f32 {
      return api.underlying.GetLearningRate.?(@ptrCast(self));
    }

    /// Registers a linear learning rate scheduler for the training session.
    pub fn registerLinearLRScheduler(self: *@This(), warmup_step_count: i64, total_step_count: i64, initial_lr: f32) !void {
      try Error.check(api.underlying.RegisterLinearLRScheduler.?(
        @ptrCast(self),
        warmup_step_count,
        total_step_count,
        initial_lr
      ));
    }

    /// Update the learning rate based on the registered learning rate scheduler.
    pub fn schedulerStep(self: *@This()) !void {
      try Error.check(api.underlying.SchedulerStep.?(@ptrCast(self)));
    }

    /// Reset the gradients of all trainable parameters to zero lazily.
    pub fn resetGrad(self: *@This()) !void {
      try Error.check(api.underlying.ResetGrad.?(@ptrCast(self)));
    }

    /// Export a model that can be used for inferencing.
    pub fn exportModelForInferencing(self: *@This(), path: [*:0]const u8, graph_output_names: []const [*:0]const u8) !void {
      try Error.check(api.underlying.ExportModelForInferencing.?(
        @ptrCast(self),
        path,
        graph_output_names.len,
        @ptrCast(graph_output_names.ptr)
      ));
    }

    pub fn deinit(self: *@This()) void {
      api.underlying.ReleaseTrainingSession.?(@ptrCast(self));
    }
  };
};

pub const Api = struct {
  /// API docs: https://onnxruntime.ai/docs/api/c/struct_Api.ort.html
  pub const c = @cImport({ @cInclude("onnxruntime/onnxruntime_training_c_api.h"); });

  pub var base: *const Api.c.OrtApiBase = undefined;
  /// a pointer to a static api struct
  pub var ort: *const Api.c.OrtApi = undefined;
  /// a pointer to a static version string
  pub var version_string: [:0]const u8 = undefined;

  /// the pointer to the Api.env used for logging n stuff.
  /// This is here because the Api.env instance is global so no point making it non_static
  /// The Env holds the logging state used by all other objects.
  pub const env = opaque {
    var underlying: *c.OrtEnv = undefined;

    /// Wraps OrtApi::CreateEnv, OrtApi::CreateEnvWithGlobalThreadPools, OrtApi::CreateEnvWithCustomLogger and OrtApi::CreateEnvWithCustomLoggerAndGlobalThreadPools
    /// The correct function is called depending on the provided options
    pub fn init(
      logging_level: Logging.Level,
      logid: [*:0]const u8,
      logging_interface: ?Logging.Interface,
      threading_options: ?ThreadingOptions,
    ) !void {
      var self: ?*@This() = null;
      if (logging_interface == null and threading_options == null) {
        try Error.check(Api.ort.CreateEnv.?(@intFromEnum(logging_level), logid, @ptrCast(&self)));
      } else if (logging_interface == null) {
        const to = try threading_options.?.c();
        defer to.deinit();
        try Error.check(Api.ort.CreateEnvWithGlobalThreadPools.?(
            @intFromEnum(logging_level),
            @ptrCast(logid),
            @ptrCast(to),
            @ptrCast(&self),
        ));
      } else if (threading_options == null) {
        try Error.check(Api.ort.CreateEnvWithCustomLogger.?(
            logging_interface.?.log_fn,
            logging_interface.?.ptr,
            @intFromEnum(logging_level),
            @ptrCast(logid),
            @ptrCast(&self),
        ));
      } else {
        const to = try threading_options.?.c();
        defer to.deinit();
        try Error.check(Api.ort.CreateEnvWithCustomLoggerAndGlobalThreadPools.?(
            logging_interface.?.log_fn,
            logging_interface.?.ptr,
            @intFromEnum(logging_level),
            @ptrCast(logid),
            @ptrCast(to),
            @ptrCast(&self),
        ));
      }
      underlying = @ptrCast(self orelse return error.OutOfMemory);
    }

    /// Wraps OrtApi::EnableTelemetry and OrtApi::DisableTelemetry
    pub fn setTelemetryEventsState(state: Options.TelemetryEventsState) !void {
      switch (state) {
        .enable => try Error.check(Api.ort.EnableTelemetryEvents.?(underlying)),
        .disable => try Error.check(Api.ort.DisableTelemetryEvents.?(underlying)),
      }
    }

    /// Wraps OrtApi::UpdateEnvWithCustomLogLevel
    pub fn updateCustomLogLevel(level: Logging.Level) !void {
      try Error.check(Api.ort.UpdateEnvWithCustomLogLevel.?(underlying, @intFromEnum(level)));
    }

    /// Identifies the programming language calling the API for telemetry tracking.
    /// see: OrtApi::setLanguageProjection
    pub const LanguageProjection = enum(c_uint) {
      C = @bitCast(Api.c.ORT_PROJECTION_C),
      CPLUSPLUS = @bitCast(Api.c.ORT_PROJECTION_CPLUSPLUS),
      CSHARP = @bitCast(Api.c.ORT_PROJECTION_CSHARP),
      PYTHON = @bitCast(Api.c.ORT_PROJECTION_PYTHON),
      JAVA = @bitCast(Api.c.ORT_PROJECTION_JAVA),
      WINML = @bitCast(Api.c.ORT_PROJECTION_WINML),
      NODEJS = @bitCast(Api.c.ORT_PROJECTION_NODEJS),
    };

    /// Wraps OrtApi::SetLanguageProjection
    pub fn setLanguageProjection(projection: LanguageProjection) !void {
      try Error.check(ort.SetLanguageProjection.?(underlying, @intFromEnum(projection)));
    }

    /// Wraps OrtApi::RegisterExecutionProviderLibrary
    pub fn registerExecutionProviderLibrary(registration_name: [*:0]const u8, path: [*:0] const c.ORTCHAR_T) !void {
      try Error.check(ort.RegisterExecutionProviderLibrary.?(underlying, @ptrCast(registration_name), path));
    }

    /// Wraps OrtApi::UnregisterExecutionProviderLibrary
    pub fn unregisterExecutionProviderLibrary(registration_name: [*:0]const u8) !void {
      try Error.check(ort.UnregisterExecutionProviderLibrary.?(underlying, @ptrCast(registration_name)));
    }

    /// Wraps OrtApi::GetEpDevices
    pub fn getEpDevices() ![]const *const Ep.Device {
      var retval_ptr: ?[*]const *const Ep.Device = null;
      var retval_len: usize = 0;
      try Error.check(ort.GetEpDevices.?(underlying, @ptrCast(&retval_ptr), &retval_len));
      return (retval_ptr orelse return error.OutOfMemory)[0 .. retval_len];
    }

    /// Wraps OrtApi::CreateSharedAllocator
    /// Create/replace a shared allocator for the OrtEpDevice in the OrtEnv.
    /// If a shared allocator already exists for the OrtEpDevice and OrtDeviceMemoryType, it is replaced.
    ///
    /// mem_type: Memory type (Default vs Host Accessible).
    /// allocator_type: The type of allocator to create.
    /// options: Optional key-value pairs to configure the allocator.
    /// returns: The created shared allocator, or null.
    pub fn createSharedAllocator(
      ep_device: *const Ep.Device,
      mem_type: Allocator.DeviceMemoryType,
      allocator_type: Allocator.Type,
      options: ?*const KeyValuePairs,
    ) !?*Allocator {
      var retval: ?*Allocator = null;
      try Error.check(Api.ort.CreateSharedAllocator.?(
          underlying,
          @ptrCast(ep_device),
          @intFromEnum(mem_type),
          @intFromEnum(allocator_type),
          @ptrCast(options),
          @ptrCast(&retval),
      ));
      return retval orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::ReleaseSharedAllocator
    /// Release a shared allocator from the OrtEnv for the OrtEpDevice and memory type.
    /// If no shared allocator exists, this is a no-op.
    pub fn releaseSharedAllocator(ep_device: *const Ep.Device, mem_type: Allocator.DeviceMemoryType) !void {
      try Error.check(Api.ort.ReleaseSharedAllocator.?(underlying, @ptrCast(ep_device), @intFromEnum(mem_type)));
    }

    /// Wraps OrtApi::CopyTensors
    /// Copy OrtValue instances containing Tensors between devices.
    /// src and dst must be the same length.
    /// stream: Optional SyncStream for async copy.
    pub fn copyTensors(src: []const *const Value, dst: []const *Value, stream: ?*SyncStream) !void {
      std.debug.assert(src.len == dst.len);
      try Error.check(Api.ort.CopyTensors.?(underlying, @ptrCast(src.ptr), @ptrCast(dst.ptr), @ptrCast(stream), src.len));
    }

    /// Release the Env object.
    pub fn deinit() void {
      Api.ort.ReleaseEnv.?(underlying);
    }
  };

  /// The model editor api
  pub const editor = opaque {
    var underlying: *const c.OrtModelEditorApi = undefined;

    /// Create an OrtValueInfo for use as an OrtGraph input or output.
    pub fn createValueInfo(name: [*:0]const u8, type_info: *const TypeInfo) !*Value.Info {
      var out: ?*Value.Info = null;
      try Error.check(underlying.CreateValueInfo.?(name, @ptrCast(type_info), @ptrCast(&out)));
      return out orelse return error.OutOfMemory;
    }

    /// Create an OrtSession using the OrtModel.
    /// The OrtModel must have a Graph with inputs/outputs set.
    pub fn createSessionFromModel(model: *const Model, options: *const Session.Options.C) !*Session {
      var out: ?*Session = null;
      try Error.check(underlying.CreateSessionFromModel.?(env.underlying, @ptrCast(model), @ptrCast(options), @ptrCast(&out)));
      return out orelse return error.OutOfMemory;
    }

    /// Create an OrtSession to augment an existing model (from path).
    pub fn createModelEditorSession(model_path: [*:0]const c.ORTCHAR_T, options: *const Session.Options.C) !*Session {
      var out: ?*Session = null;
      try Error.check(underlying.CreateModelEditorSession.?(env.underlying, model_path, @ptrCast(options), @ptrCast(&out)));
      return out orelse return error.OutOfMemory;
    }

    /// Create an OrtSession to augment an existing model (from memory).
    pub fn createModelEditorSessionFromArray(data: []const u8, options: *const Session.Options.C) !*Session {
      var out: ?*Session = null;
      try Error.check(underlying.CreateModelEditorSessionFromArray.?(
        env.underlying, 
        data.ptr, 
        data.len, 
        @ptrCast(options), 
        @ptrCast(&out)
      ));
      return out orelse return error.OutOfMemory;
    }

    /// Query the session for the opset version of a domain.
    pub fn sessionGetOpsetForDomain(session: *const Session, domain: [*:0]const u8) !c_int {
      var opset: c_int = 0;
      try Error.check(underlying.SessionGetOpsetForDomain.?(@ptrCast(session), domain, &opset));
      return opset;
    }

    /// Apply changes to augment the ONNX model in a session.
    pub fn applyModelToModelEditorSession(session: *Session, model: *Model) !void {
      try Error.check(underlying.ApplyModelToModelEditorSession.?(@ptrCast(session), @ptrCast(model)));
    }

    /// Finalize the Model Editor session.
    pub fn finalizeModelEditorSession(session: *Session, options: *const Session.Options.C, prepacked: ?*PrepackedWeightsContainer) !void {
      try Error.check(underlying.FinalizeModelEditorSession.?(@ptrCast(session), @ptrCast(options), @ptrCast(prepacked)));
    }
  };

  /// The compiler api
  pub const compiler = opaque {
    var underlying: *const c.OrtCompileApi = undefined;

    /// Translated from OrtApi::OrtCompileApiFlags
    pub const Flags = packed struct {
      ErrorIfNoNodesCompiled: bool = false,
      ErrorIfOutputFileExists: bool = false,
      _padding: u30 = 0,
    };

    comptime {
      std.debug.assert(@bitSizeOf(Flags) == 32);
    }
  };

  const Options = struct {
    /// The logging level for messages generated.
    log_level: Logging.Level,
    /// This must outlive `this` and any instances created by `this` OnnxInstanceCreator.
    log_id: [*:0]const u8,
    /// The custom logger with context. If this is null, the default logger is used.
    logging_interface: ?Logging.Interface = null,
    /// The threadpool used for initializing env, when null, default is used
    threading_options: ?ThreadingOptions = null,
    /// Telemetry, just useless stuff including things like System Info, Errors and Performance Metrics etc.
    /// Null means default (at the time of writing, they are enabled by default)
    telemetry_events: ?TelemetryEventsState = .disable,
    /// Weather or not to initialize the editor api
    editor: bool = false,
    /// Weather or not to initialize the compiler api
    compiler: bool = false,
    /// Weather or not to initialize the ep api
    ep: bool = false,
    /// Weather or not to initialize the training api
    training: bool = false,

    pub const TelemetryEventsState = enum {enable, disable};
  };

  pub const ComptimeOptions = struct {
    editor_behavior: UninitializedBehavior = .panicking,
    compile_behavior: UninitializedBehavior = .panicking,
    ep_behavior: UninitializedBehavior = .panicking,
    training_behavior: UninitializedBehavior = .panicking,

    pub const UninitializedBehavior = enum {
      /// All function pointers in the uninitialized api are null, causes ub in release mode
      uninitialized,
      /// If any of the functions in the impl are called, it causes a panic (no ub in release mode) [recommend]
      /// The binary will include panic implementation thus will be slightly larger.
      panicking,
    };

    fn impl(behavior: UninitializedBehavior, T: type) *const T {
      if (behavior == .uninitialized) return undefined;

      comptime var retval: T = undefined;
      // When arguments are passed to functions in c api, they are either put on stack or passed through registers.
      // Since our panic function is notreturn, we don't really care about possibly corrupting the stack when returning, thus we can just pointercast the panic function.
      if (@import("builtin").mode == .ReleaseSmall) {
        const panicFn = struct {pub fn panicFn() callconv(.c) noreturn {
          @panic("cannot call any function inside " ++ @typeName(T) ++ " as it was not initialized");
        }}.panicFn;
        inline for (@typeInfo(T).@"struct".fields) |f| @field(retval, f.name) = @ptrCast(&panicFn);
      } else {
        inline for (@typeInfo(T).@"struct".fields) |f| @field(retval, f.name) = @ptrCast(&struct {pub fn panicFn() callconv(.c) noreturn {
          @panic("cannot call " ++ f.name ++ " as " ++ @typeName(T) ++ " was not initialized");
        }}.panicFn);
      }

      const v = retval;
      return &v;
    }
  };

  /// You MUST call this function before creating anything.
  /// You need to call `deinitApi` to free resources created by this function
  pub fn init(options: Options, comptime comptime_options: ComptimeOptions) !void {
    base = Api.c.OrtGetApiBase().*;
    ort = base.GetApi.?(c.ORT_API_VERSION);
    version_string = std.mem.sliceTo(@as([*:0] const u8, @ptrCast(base.GetVersionString.?())), 0);

    editor.underlying = if (options.editor) ort.GetModelEditorApi.?() else comptime_options.impl(comptime_options.editor_behavior, c.OrtModelEditorApi);
    compiler.underlying = if (options.compiler) ort.GetCompileApi.?() else comptime_options.impl(comptime_options.compile_behavior, c.OrtCompileApi);
    Ep.api.underlying = if (options.ep) ort.GetEpApi.?() else comptime_options.impl(comptime_options.ep_behavior, c.OrtEpApi);
    const training_fallback = comptime_options.impl(comptime_options.training_behavior, c.OrtTrainingApi);
    Training.api.underlying = if (options.training) (ort.GetTrainingApi.?(c.ORT_API_VERSION) orelse training_fallback) else training_fallback;

    Api.env.underlying = .init(options.log_level, options.log_id, options.logging_interface, options.threading_options);
    errdefer Api.env.deinit();

    if (options.telemetry_events) |state| env.setTelemetryEventsState(state);
  }

  /// Frees the resources created by the `Api.init` function
  pub fn deinit() void {
    // `Api.ort` and `version_string` are static
    Api.env.deinit();
  }

  /// Returns a null terminated string of the build info including git info and cxx flags.
  /// Returned string is static and this should NOT be deallocated.
  /// Wraps OrtApi::GetBuildInfoString
  pub fn getBuildInfoString() [*:0]const u8 {
    return Api.ort.GetBuildInfoString.?();
  }

  pub const Providers = struct {
    providers: [][*:0]u8,

    pub fn has(self: *const @This(), provider: []const u8) bool {
      return self.index(provider) != null;
    }

    pub fn index(self: *const @This(), provider: []const u8) ?std.meta.Int(.unsigned, @bitSizeOf(c_uint) - 1) {
      for (self.providers, 0..) |p, i| {
        if (std.mem.eql(u8, provider, std.mem.sliceTo(p, 0))) return @intCast(i);
      }
      return null;
    }

    /// So freeing may fail too!!?!?!
    pub fn deinit(self: *const @This()) !void {
      try Error.check(ort.ReleaseAvailableProviders.?(@ptrCast(self.providers.ptr), @intCast(self.providers.len)));
    }
  };

  /// Returns a list of available execution providers. The caller must free the returned value using deinit
  /// Wraps OrtApi::GetAvailableProviders
  pub fn getAvailableProviders() !Providers {
    var ptrs: ?[*][*:0]u8 = null;
    var len: c_int = 0;
    try Error.check(Api.ort.GetAvailableProviders.?(&ptrs, &len));
    if (len < 0) return error.InvalidLength;
    return .{ .providers = (ptrs orelse return error.OutOfMemory)[0 .. @intCast(len)] };
    
  }


  /// Set current GPU device ID.
  /// Useful when multiple-GPUs are installed and it is required to restrict execution to a single GPU.
  pub fn setCurrentGpuDeviceId(device_id: c_int) !void {
    try Error.check(Api.ort.SetCurrentGpuDeviceId.?(device_id));
  }

  /// Get current GPU device ID.
  pub fn getCurrentGpuDeviceId() !c_int {
    var device_id: c_int = 0;
    try Error.check(Api.ort.GetCurrentGpuDeviceId.?(&device_id));
    return device_id;
  }

  /// Get a pointer to the requested version of the Execution Provider specific API.
  /// provider_name: e.g. "DML"
  /// version: e.g. Api.c.ORT_API_VERSION
  pub fn getExecutionProviderApi(provider_name: [*:0]const u8, version: u32) !*const anyopaque {
    var ptr: ?*const anyopaque = null;
    try Error.check(Api.ort.GetExecutionProviderApi.?(provider_name, version, &ptr));
    return ptr orelse error.NotFound;
  }

  /// Validate a compiled model's compatibility information for one or more EP devices.
  pub fn getModelCompatibilityForEpDevices(
    ep_devices: []const *const Ep.Device,
    compatibility_info: [*:0]const u8
  ) !Api.c.OrtCompiledModelCompatibility {
    var out: Api.c.OrtCompiledModelCompatibility = undefined;
    try Error.check(Api.ort.GetModelCompatibilityForEpDevices.?(
        @ptrCast(ep_devices.ptr),
        ep_devices.len,
        compatibility_info,
        &out
    ));
    return out;
  }
};

pub const Logging = struct {
  /// Levels of logging verbosity, from least severe (verbose) to most severe (fatal).
  pub const Level = enum(c_uint) {
    verbose = @bitCast(Api.c.ORT_LOGGING_LEVEL_VERBOSE),
    info = @bitCast(Api.c.ORT_LOGGING_LEVEL_INFO),
    warning = @bitCast(Api.c.ORT_LOGGING_LEVEL_WARNING),
    @"error" = @bitCast(Api.c.ORT_LOGGING_LEVEL_ERROR),
    fatal = @bitCast(Api.c.ORT_LOGGING_LEVEL_FATAL),
  };

  pub const Interface = struct {
    /// this is passed to the log_fn
    ptr: ?*anyopaque,
    /// the logging_function itself
    log_fn: @typeInfo(Api.c.OrtLoggingFunction).optional.child,

    /// context must be an instance of type with a `log` function that takes in the following arguments.
    ///   severity: Logging.Level, category: ?[*c]const u8, logid: ?[*c]const u8, code_location: ?[*c]const u8, messages: ?[*:0]const u8
    ///
    /// if (@bitSizeOf(@TypeOf(context)) != 0) then `context` must outlive the instance creator and any created instances
    pub fn fromContext(context: anytype) @This() {
      const T = @TypeOf(context);
      const Sub = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
      return .{
        .ptr = if (@bitSizeOf(Sub) == 0) null else @ptrCast(context),
        .log_fn = &struct {
          fn log_fn(
            ctx: ?*anyopaque,
            severity: Api.c.OrtLoggingLevel,
            category: ?[*:0]const u8,
            logid: ?[*:0]const u8,
            code_location: ?[*:0]const u8,
            messages: ?[*:0]const u8
          ) callconv(.c) void {
            const original: *Sub = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @ptrCast(ctx.?);
            original.log(@as(Logging.Level, @enumFromInt(severity)), category, logid, code_location, messages);
          }
        }.log_fn,
      };
    }
  };
};

pub const Error = struct {
  pub const Set = error {
    OrtErrorFail,
    OrtErrorInvalidArgument,
    OrtErrorNoSuchfile,
    OrtErrorNoModel,
    OrtErrorEngineError,
    OrtErrorRuntimeException,
    OrtErrorInvalidProtobuf,
    OrtErrorModelLoaded,
    OrtErrorNotImplemented,
    OrtErrorInvalidGraph,
    OrtErrorEpFail,
    OrtErrorModelLoadCanceled,
    OrtErrorModelRequiresCompilation,
    OrtErrorNotFound,
  } || error {
    OrtErrorUnknown
  };

  /// Return codes for API functions; Ok indicates success, others indicate specific failure types.
  pub const Code = enum(c_uint) {
    Ok = @bitCast(Api.c.ORT_OK),
    Fail = @bitCast(Api.c.ORT_FAIL),
    InvalidArgument = @bitCast(Api.c.ORT_INVALID_ARGUMENT),
    NoSuchfile = @bitCast(Api.c.ORT_NO_SUCHFILE),
    NoModel = @bitCast(Api.c.ORT_NO_MODEL),
    EngineError = @bitCast(Api.c.ORT_ENGINE_ERROR),
    RuntimeException = @bitCast(Api.c.ORT_RUNTIME_EXCEPTION),
    InvalidProtobuf = @bitCast(Api.c.ORT_INVALID_PROTOBUF),
    ModelLoaded = @bitCast(Api.c.ORT_MODEL_LOADED),
    NotImplemented = @bitCast(Api.c.ORT_NOT_IMPLEMENTED),
    InvalidGraph = @bitCast(Api.c.ORT_INVALID_GRAPH),
    EpFail = @bitCast(Api.c.ORT_EP_FAIL),
    ModelLoadCanceled = @bitCast(Api.c.ORT_MODEL_LOAD_CANCELED),
    ModelRequiresCompilation = @bitCast(Api.c.ORT_MODEL_REQUIRES_COMPILATION),
    NotFound = @bitCast(Api.c.ORT_NOT_FOUND),
  };

  /// This is a wrapper around c version of OrtStatus
  pub const Status = opaque {
    /// create a new status, this function uses c's allocator
    pub fn init(code: c_uint, msg: [*:0]const u8) !*@This() {
      return @ptrCast(Api.ort.CreateStatus.?(code, msg) orelse return error.OutOfMemory);
    }

    /// get the error messages from the status
    pub fn getErrorMessage(self: *const @This()) [*:0]const u8 {
      return @ptrCast(Api.ort.GetErrorMessage.?(@ptrCast(self)));
    }

    /// Get the error code from the status
    pub fn getErrorCode(self: *const @This()) c_uint {
      return Api.ort.GetErrorCode.?(@ptrCast(self));
    }

    /// release the status
    pub fn deinit(self: *@This()) void {
      return Api.ort.ReleaseStatus.?(@ptrCast(self));
    }
  };

  /// A simple error checking function.
  /// This function is a no-op if onnx_status is null
  /// If `onnx_status` is NOT null, this function
  ///   calls the on_error_fn and returns it's error or returns `OnnxError` if it is null
  pub fn check(ort_status: ?*Api.c.OrtStatus) Set!void {
    if (@as(?*Status, @ptrCast(ort_status))) |status| {
      const min_error_code, const max_error_code = comptime blk: {
        const fields = @typeInfo(Error.Code).@"enum".fields;
        var min = fields[0].value;
        var max = min;
        for (fields[1 .. ]) |f| {
          min = @min(min, f.value);
          max = @max(max, f.value);
        }
        break :blk .{min, max};
      };
      defer status.deinit();
      return switch (status.getErrorCode()) {
        inline min_error_code ... max_error_code => |ec| {
          // Weird .. weird .. weird
          if (ec == @intFromEnum(Error.Code.Ok)) return;
          const name = "OrtError" ++ @tagName(@as(Error.Code, @enumFromInt(ec)));
          // this is cursed i know. If you know of a cleaner way to create an error from string, please open a PR asap
          return @field(@Type(.{.error_set = &[_]std.builtin.Type.Error{.{.name = name}}}), name);
        },
        else => Set.OrtErrorUnknown,
      };
    }
  }
};

pub const KeyValuePairs = opaque {
  /// Wraps OrtApi::CreateKeyValuePairs
  pub fn init() !*@This() {
    var retval: ?*@This() = undefined;
    Api.ort.CreateKeyValuePairs.?(@ptrCast(&retval));
    return retval orelse error.OutOfMemory;
  }

  /// Wraps OrtApi::AddKeyValuePair.
  pub fn add(self: *@This(), key: [*:0]const u8, value: [*:0]const u8) void {
    // Since AddKeyValuePair returns nothing, we don't know if it errored; hence this function returns no error
    // We technically could use `get` but no need for another virtual call if library authors thought this was not an issue
    Api.ort.AddKeyValuePair.?(@ptrCast(self), key, value);
  }

  /// Wraps OrtApi::GetKeyValue
  pub fn get(self: *const @This(), key: [*:0]const u8) ?[*:0]const u8 {
    return @ptrCast(Api.ort.GetKeyValue.?(@ptrCast(self), key));
  }

  /// Wraps OrtApi::RemoveKeyValuePair
  pub fn remove(self: *@This(), key: [*:0]const u8) void {
    Api.ort.RemoveKeyValuePair.?(@ptrCast(self), key);
  }

  /// Wraps OrtApi::GetKeyValuePairs
  /// Returns slices of keys and values. These pointers are valid as long as the KeyValuePairs object is valid.
  pub fn getKeyValues(self: *const @This()) std.meta.Tuple(&.{ []const [*:0]const u8, []const [*:0]const u8 }) {
    var keys_ptr: [*c]const [*:0]const u8 = &[_][*:0]const u8{};
    var values_ptr: [*c]const [*:0]const u8 = &[_][*:0]const u8{};
    var count: usize = 0;
    Api.ort.GetKeyValuePairs.?(@ptrCast(self), @ptrCast(&keys_ptr), @ptrCast(&values_ptr), &count);
    return .{ keys_ptr[0..count], values_ptr[0..count] };
  }

  /// Wraps OrtApi::ReleaseKeyValuePairs
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseKeyValuePairs.?(@ptrCast(self));
  }
};

pub const SyncStream = opaque {
  /// Wraps OrtApi::CreateSyncStreamForEpDevice
  /// stream_options: Optional KeyValuePairs for stream configuration.
  pub fn init(device: *const Ep.Device, stream_options: ?*const KeyValuePairs) !*@This() {
    var retval: ?*@This() = undefined;
    try Error.check(Api.ort.CreateSyncStreamForEpDevice.?(@ptrCast(device), @ptrCast(stream_options), @ptrCast(&retval)));
    return retval orelse error.OutOfMemory;
  }

  /// Wraps OrtApi::SyncStream_GetHandle
  /// Returns the native handle
  pub fn getHandle(self: *@This()) ?*anyopaque {
    return Api.ort.SyncStream_GetHandle.?(@ptrCast(self));
  }

  /// Wraps OrtApi::ReleaseSyncStream
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseSyncStream.?(@ptrCast(self));
  }

  /// Get the OrtSyncStreamImpl associated with an OrtSyncStream instance.
  ///
  /// This allows an the plugin library to connect its OrtSyncStreamImpl instance with an OrtSyncStream if needed.
  ///
  /// stream: The OrtSyncStream instance to find an OrtSyncStreamImpl for.
  /// returns The associated OrtSyncStreamImpl if found. nullptr otherwise.
  ///
  /// since Version 1.23.
  ///
  /// Remarks: There should always be an OrtSyncStreamImpl associated with an OrtSyncStream instance that the EP gets.
  pub fn getImpl(self: *const @This()) *const Ep.SyncStreamImpl {
    return @ptrCast(Ep.api.underlying.SyncStream_GetImpl.?(@ptrCast(self)));
  }

  /// Get the current sync ID for a stream.
  ///
  /// stream: The OrtSyncStream to get the sync ID for.
  /// returns Current sync ID.
  ///
  /// since Version 1.23.
  pub fn getSyncId(self: *const @This()) u64 {
    return Ep.api.underlying.SyncStream_GetSyncId.?(@ptrCast(self));
  }

  /// Get the sync ID for the last time the consumer_stream waited on the producer_stream.
  ///
  /// When two streams are synchronized, the sync id represents the event used in that synchronization.
  ///
  /// producer_stream: The OrtSyncStream that produced the data.
  /// consumer_stream: The OrtSyncStream that waited on the producer_stream.
  /// returns ID for last sync. 0 if no sync has occurred between the two streams.
  ///
  /// since Version 1.23.
  pub fn getSyncIdForLastWaitOnSyncStream(producer: *const @This(), consumer: *const @This()) u64 {
    return Ep.api.underlying.GetSyncIdForLastWaitOnSyncStream.?(@ptrCast(producer), @ptrCast(consumer));
  }
};

/// Structure of function pointers that defines a memory allocator.
/// Used for both internal and custom user-defined memory management.
pub const Allocator = struct {
  /// The underlying C allocator pointer.
  underlying: Api.c.OrtAllocator,
  comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

  /// Types of memory allocators (Device-local, Arena-wrapped, or Read-only).
  pub const Type = enum(c_int) {
    invalid = @bitCast(Api.c.OrtInvalidAllocator),
    device = @bitCast(Api.c.OrtDeviceAllocator),
    arena = @bitCast(Api.c.OrtArenaAllocator),
    readonly = @bitCast(Api.c.OrtReadOnlyAllocator),
  };

  /// This matches OrtDevice::MemoryType values
  /// Specific memory access traits for devices (Default vs Host Accessible).
  pub const DeviceMemoryType = enum(c_uint) {
    DEFAULT = @bitCast(Api.c.OrtDeviceMemoryType_DEFAULT),
    HOST_ACCESSIBLE = @bitCast(Api.c.OrtDeviceMemoryType_HOST_ACCESSIBLE),
  };

  /// Configuration for an arena-based allocator.
  /// Defines maximum memory limits and growth strategies.
  pub const ArenaCfg = opaque {
    pub const Options = struct {
      /// Maximum memory limit (0 for default).
      max_mem: usize = 0,
      /// extend_strategy: 0 = NextPowerOfTwo, 1 = SameAsRequested.
      extend_strategy: ExtendStrategy = .Default,
      /// Size of the first allocation in the arena.
      initial_chunk: c_int = -1,
      /// Threshold for unused memory in a chunk before it is split.
      max_dead: c_int = -1,

      pub const ExtendStrategy = enum(c_int) {
        Default = -1,
        NextPowerOfTwo = 0,
        SameAsRequested = 1,
      };
    };

    /// Create the configuration of an arena that can be used to define an arena-based allocator's behavior.
    pub fn init_DeprecatedV1(options: Options) !*@This() {
      var self: *@This() = undefined;
      try Error.check(Api.ort.CreateArenaCfg.?(
        options.max_mem,
        @intFromEnum(options.extend_strategy),
        options.initial_chunk,
        options.max_dead,
        @ptrCast(&self)
      ));
      return self;
    }

    /// Supported keys are (See https://onnxruntime.ai/docs/get-started/with-c.html for details on what the
    /// the keys are specified in the same order as in onnxruntime/core/framework/allocator.cc:OrtArenaCfg::FromKeyValuePairs for performance
    pub const OptionsV2 = struct {
      const DEFAULT = std.math.maxInt(usize);
      /// extend_strategy: 0 = NextPowerOfTwo, 1 = SameAsRequested.
      extend_strategy: ExtendStrategy = .Default,
      /// (Possible) Size of the first allocation in the arena.
      /// Only relevant if arena strategy is `kNextPowerOfTwo`. Use maxInt to allow ORT to choose the default.
      /// Ultimately, the first allocation size is determined by the allocation memory request.
      initial_chunk_size_bytes: usize = DEFAULT,
      /// Threshold of unused memory in an allocated chunk of arena memory after
      /// crossing which the current chunk is chunked into 2.
      max_dead_bytes_per_chunk: ?usize = null,
      /// (Possible) Size of the second allocation in the arena.
      /// Only relevant if arena strategy is `kNextPowerOfTwo`. Use maxInt to allow ORT to choose the default.
      initial_growth_chunk_size_bytes: usize = DEFAULT,
      /// The maximum extend size if arena strategy is `kNextPowerOfTwo`.
      /// It is not an allocation limit, it is only a limit for extension when requested byte is less than the limit.
      /// When requested bytes is more than the limit, allocator will still return as requested.
      /// Use maxInt to allow ORT to choose the default 1GB for max_power_of_two_extend_bytes.
      /// Ultimately, the allocation size is determined by the allocation memory request.
      /// Further allocation sizes are governed by the arena extend strategy.
      max_power_of_two_extend_bytes: usize = DEFAULT,
      /// Maximum memory that can be allocated by the arena based allocator.
      /// Use 0 for ORT to pick the best value. Default is 0.
      max_mem: usize = 0,
      /// Use CudaMemPool based arena if available (starting with cuda 11.2)
      use_cuda_mempool: usize = DEFAULT,
      /// Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. 0 is default
      cuda_mempool_release_threshold: usize = 0,
      /// Bytes to keep on shrink for CudaMemPool, 0 (default) is to attempt to release all, allocated space not affected.
      cuda_mempool_bytes_to_keep_on_shrink: usize = 0,

      pub const ExtendStrategy = enum(usize) {
        Default = DEFAULT,
        NextPowerOfTwo = 0,
        SameAsRequested = 1,
      };
    };

    /// Create an ArenaCfg using key-value pairs of configuration options.
    pub fn init(options: OptionsV2) !*@This() {
      const created = Utils.createOptionsKVL(options, .usize);
      var self: *@This() = undefined;
      try Error.check(Api.ort.CreateArenaCfgV2.?(created.keys(), created.vals(), created.len, @ptrCast(&self)));
      return self;
    }

    /// Release the ArenaCfg object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseArenaCfg.?(@ptrCast(self));
    }
  };

  /// Memory types for allocated memory, execution provider specific types should be extended in each provider.
  pub const MemoryType = enum(c_int) {
    /// The default allocator for execution provider
    default = @bitCast(Api.c.OrtMemTypeDefault),
    /// Any CPU memory used by non-CPU execution provider
    cpu_input = @bitCast(Api.c.OrtMemTypeCPUInput),
    /// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    cpu_output = @bitCast(Api.c.OrtMemTypeCPUOutput),
  };

  pub const MemoryDevice = opaque{
    /// This mimics OrtDevice type constants so they can be returned in the API
    pub const Type = enum(c_uint) {
      CPU = @bitCast(Api.c.OrtMemoryInfoDeviceType_CPU),
      GPU = @bitCast(Api.c.OrtMemoryInfoDeviceType_GPU),
      FPGA = @bitCast(Api.c.OrtMemoryInfoDeviceType_FPGA),
      NPU = @bitCast(Api.c.OrtMemoryInfoDeviceType_NPU),
    };

    /// Compare two OrtMemoryDevice instances for equality.
    ///
    /// This is used to check if two memory devices are the same.
    /// Used to implement DataTransferImpl::CanCopy.
    ///
    /// a: The first OrtMemoryDevice instance to compare.
    /// b: The second OrtMemoryDevice instance to compare.
    /// returns True if the two OrtMemoryDevice instances are equal, false otherwise.
    ///
    /// since Version 1.23.
    pub fn equal(self: *const @This(), other: *const @This()) bool {
      return Ep.api.underlying.MemoryDevice_AreEqual.?(@ptrCast(self), @ptrCast(other));
    }

    /// Get the OrtMemoryInfoDeviceType value from an OrtMemoryDevice instance.
    ///
    /// memory_device: OrtMemoryDevice instance.
    /// returns The OrtMemoryInfoDeviceType value.
    ///
    /// since Version 1.23.
    pub fn getDeviceType(self: *const @This()) MemoryDevice.Type {
      return @enumFromInt(Ep.api.underlying.MemoryDevice_GetDeviceType.?(@ptrCast(self)));
    }

    /// Get the OrtDeviceMemoryType value from an OrtMemoryDevice instance.
    ///
    /// memory_device: OrtMemoryDevice instance.
    /// returns The OrtDeviceMemoryType value.
    ///
    /// since Version 1.23.
    pub fn getMemoryType(self: *const @This()) Allocator.DeviceMemoryType {
      return @enumFromInt(Ep.api.underlying.MemoryDevice_GetMemoryType.?(@ptrCast(self)));
    }

    /// Get the vendor ID from an OrtMemoryDevice instance.
    ///
    /// The vendor ID is used to identify the vendor of the device, and is typically set to the PCI vendor ID.
    /// If the device is not vendor specific (e.g. CPU memory) the vendor ID is 0.
    ///
    /// memory_device: OrtMemoryDevice instance.
    /// returns The vendor ID value.
    ///
    /// since Version 1.23.
    pub fn getVendorId(self: *const @This()) u32 {
      return Ep.api.underlying.MemoryDevice_GetVendorId.?(@ptrCast(self));
    }

    /// Get the device ID from an OrtMemoryDevice instance.
    ///
    /// memory_device: OrtMemoryDevice instance.
    /// returns The device ID.
    ///
    /// since Version 1.23.
    pub fn getDeviceId(self: *const @This()) u32 {
      return Ep.api.underlying.MemoryDevice_GetDeviceId.?(@ptrCast(self));
    }
  };

  /// Describes the location and traits of a memory allocation (e.g., CPU, GPU device id).
  pub const MemoryInfo = opaque {
    /// Create an ::OrtMemoryInfo
    /// name_: Arbitrary name.
    /// type: The allocator type (Device vs Arena).
    /// id_: Device ID.
    /// mem_type: Memory type (Input vs Output vs Default).
    pub fn init(name_: [*:0]const u8, alloc_type: Allocator.Type, id_: i32, mem_type: MemoryType) !*@This() {
      var self: *@This() = undefined;
      try Error.check(Api.ort.CreateMemoryInfo.?(name_, @intFromEnum(alloc_type), id_, @intFromEnum(mem_type), @ptrCast(&self)));
      return self;
    }

    /// Create an ::OrtMemoryInfo for CPU memory
    /// Special case version of OrtApi::CreateMemoryInfo for CPU based memory. 
    /// Same as using OrtApi::CreateMemoryInfo with name = "Cpu" and id = 0.
    pub fn initCpu(alloc_type: Allocator.Type, mem_type: MemoryType) !*@This() {
      var self: *@This() = undefined;
      try Error.check(Api.ort.CreateCpuMemoryInfo.?(@intFromEnum(alloc_type), @intFromEnum(mem_type), @ptrCast(&self)));
      return self;
    }

    pub const OptionsV2 = struct {
      /// Arbitrary name.
      name: [*:0]const u8,
      /// Device type (CPU, GPU, NPU, etc).
      device_type: Allocator.MemoryDevice.Type,
      /// PCI Vendor ID. Use 0 for a generic allocator.
      vendor_id: u32 = 0,
      /// Device ID if there are multiple devices of the same type.
      device_id: i32 = 0,
      /// Memory type (Default vs Host Accessible).
      mem_type: Allocator.DeviceMemoryType = .DEFAULT,
      /// Alignment of the memory if required. 0 for default.
      alignment: usize = 0,
      /// Allocator type.
      allocator_type: Allocator.Type = .device,
    };

    /// Create an ::OrtMemoryInfo (V2)
    /// This version allows specifying device types, vendor IDs, and alignment.
    pub fn initV2(options: OptionsV2) !*@This() {
      var self: *@This() = undefined;
      try Error.check(Api.ort.CreateMemoryInfo_V2.?(
        options.name,
        @intFromEnum(options.device_type),
        options.vendor_id,
        options.device_id,
        @intFromEnum(options.mem_type),
        options.alignment,
        @intFromEnum(options.allocator_type),
        @ptrCast(&self)
      ));
      return self;
    }

    /// Compare ::OrtMemoryInfo objects for equality
    /// Compares all settings of each ::OrtMemoryInfo for equality
    pub fn compare(self: *const @This(), other: *const @This()) !bool {
      var out: c_int = undefined;
      try Error.check(Api.ort.CompareMemoryInfo.?(@ptrCast(self), @ptrCast(other), &out));
      return out == 0;
    }

    /// Get name from ::OrtMemoryInfo
    /// Do NOT free the returned pointer. It is valid for the lifetime of the ::OrtMemoryInfo
    pub fn name(self: *const @This()) ![*:0]const u8 {
      var out: ?[*:0]const u8 = null;
      try Error.check(Api.ort.MemoryInfoGetName.?(@ptrCast(self), @ptrCast(&out)));
      return out.?;
    }

    /// Get the device id from ::OrtMemoryInfo
    pub fn id(self: *const @This()) !c_int {
      var out: c_int = 0;
      try Error.check(Api.ort.MemoryInfoGetId.?(@ptrCast(self), &out));
      return out;
    }

    /// Get the ::OrtMemType from ::OrtMemoryInfo
    pub fn memType(self: *const @This()) !MemoryType {
      var out: MemoryType = undefined;
      try Error.check(Api.ort.MemoryInfoGetMemType.?(@ptrCast(self), @ptrCast(&out)));
      return out;
    }

    /// Get the ::Allocator.Type from ::OrtMemoryInfo
    pub fn allocatorType(self: *const @This()) !Type {
      var out: Type = undefined;
      try Error.check(Api.ort.MemoryInfoGetType.?(@ptrCast(self), @ptrCast(&out)));
      return out;
    }

    /// Get the OrtMemoryDevice from an OrtMemoryInfo instance.
    ///
    /// This is required for OrtDataTransferImpl (which implements onnxruntime::IDataTransfer) where the OrtMemoryDevice
    /// is used in the CanCopy and CopyTensors functions.
    ///
    /// memory_info: The OrtMemoryInfo instance to get the memory device from.
    /// returns: The OrtMemoryDevice associated with the OrtMemoryInfo instance.
    ///
    /// since Version 1.23.
    pub fn getDevice(self: *const @This()) !*const Allocator.MemoryDevice {
      return @ptrCast(Ep.api.underlying.MemoryInfo_GetMemoryDevice.?(@ptrCast(self)) orelse return error.OutOfMemory);
    }

    /// Get OrtDevice type from MemoryInfo
    pub fn deviceType(self: *const @This()) !MemoryDevice.Type {
      var out: MemoryDevice.Type = undefined;
      Api.ort.MemoryInfoGetDeviceType.?(@ptrCast(self), @ptrCast(&out));
      return out;
    }

    /// Get the device memory type from ::OrtMemoryInfo
    pub fn deviceMemType(self: *const @This()) DeviceMemoryType {
      return @enumFromInt(Api.ort.MemoryInfoGetDeviceMemType.?(@ptrCast(self)));
    }

    /// Get the vendor id from ::OrtMemoryInfo
    pub fn vendorId(self: *const @This()) u32 {
      return Api.ort.MemoryInfoGetVendorId.?(@ptrCast(self));
    }

    /// Release the MemoryInfo object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseMemoryInfo.?(@ptrCast(self));
    }

    pub fn unregisterAllocator(self: *const @This()) !void {
      try Error.check(Api.ort.UnregisterAllocator.?(Api.env.underlying, @ptrCast(self)));
    }

    /// Wraps OrtApi::CreateAndRegisterAllocator
    pub fn createAndRegisterAllocator(self: *const @This(), arena_cfg: ?*const ArenaCfg) !void {
      try Error.check(Api.ort.CreateAndRegisterAllocator.?(Api.env.underlying, @ptrCast(self), @ptrCast(arena_cfg)));
    }

    pub const CreateAndRegisterOptionsV2 = struct {}; // TODO

    /// Create an allocator with specific type and register it with the ::OrtEnv
    /// This API enhance CreateAndRegisterAllocator that it can create an allocator with specific type, not just CPU allocator
    /// Enables sharing the allocator between multiple sessions that use the same env instance.
    /// Lifetime of the created allocator will be valid for the duration of the environment.
    /// Returns an error if an allocator with the same ::OrtMemoryInfo is already registered.
    ///
    /// The original comment does not specify if arena_cfg can be null but it can be null for the normal version so only makes sense to assume that it can be null here as well
    pub fn createAndRegisterAllocatorV2(self: *const @This(), provider_type: [*:0]const u8, arena_cfg: ?*const ArenaCfg, options: CreateAndRegisterOptionsV2) !void {
      const created = Utils.createOptionsKVL(options, .cstr);
      try Error.check(Api.ort.CreateAndRegisterAllocatorV2.?(
          Api.env.underlying,
          provider_type,
          @ptrCast(self),
          @ptrCast(arena_cfg),
          created.keys(),
          created.vals(),
          created.len
      ));
    }

    /// Wraps OrtApi::GetSharedAllocator
    /// Get a shared allocator from the OrtEnv.
    /// It is not an error to not find a matching allocator, in which case null is returned.
    pub fn getSharedAllocator(self: *const @This()) !?*Allocator {
      var retval: ?*Allocator = null;
      try Error.check(Api.ort.GetSharedAllocator.?(Api.env.underlying, @ptrCast(self), @ptrCast(&retval)));
      return retval;
    }
  };

  pub fn version(self: *const @This()) u32 {
    return self.underlying.version;
  }

  /// Allocates a block of memory ofthe specified size.
  /// size: Size in bytes.
  /// returns the pointer to the allocated block. null is size = 0 or allocation failed
  pub fn alloc(self: *@This(), size: usize) ?[]u8 {
    return @as([*]u8, @ptrCast(self.underlying.Alloc.?(@ptrCast(self), size) orelse return null))[0 .. size];
  }

  /// Frees a block of memory previously allocated with this allocator.
  /// p: Pointer to the memory block.
  pub fn free(self: *@This(), p: ?[*]u8) void {
    self.underlying.Free.?(@ptrCast(self), @ptrCast(p));
  }

  /// Return a pointer to an ::OrtMemoryInfo that describes this allocator
  pub fn info(self: *const @This()) *const MemoryInfo {
    return @ptrCast(self.underlying.Info.?(@ptrCast(self)).?);
  }

  /// Optional allocation function to use for memory allocations made during session initialization.
  /// Use this function if you want to separate allocations made by ORT during Run() calls from
  /// those made during session initialization. This allows for separate memory management strategies for these
  /// allocations.
  /// size: Size in bytes.
  /// returns the pointer to an allocated block of `size` bytes. null if size was 0 or allocation failed.
  pub fn reserve(self: *@This(), size: usize) ?[]u8 {
    return @as([*]u8, @ptrCast((self.underlying.Reserve orelse self.underlying.Alloc.?)(@ptrCast(self), size) orelse return null))[0 .. size];
  }

  /// Return a pointer to the OrtKeyValuePairs structure that contains the statistics of the allocator.
  /// The user should call `deinit` (wrapper around OrtApi::ReleaseKeyValuePairs) when done.
  ///
  /// Current known keys are:
  /// - Limit: Bytes limit of the allocator. -1 if no limit is set.
  /// - InUse: Number of bytes in use.
  /// - TotalAllocated: The total number of allocated bytes by the allocator.
  /// - MaxInUse: The maximum bytes in use.
  /// - NumAllocs: Number of allocations.
  /// - NumReserves: Number of reserves. (Number of calls to Reserve() in arena-based allocators)
  /// - NumArenaExtensions: Number of arena extensions (Relevant only for arena based allocators)
  /// - NumArenaShrinkages: Number of arena shrinkages (Relevant only for arena based allocators)
  /// - MaxAllocSize: The max single allocation seen.
  ///
  /// The allocator is free to add other entries as appropriate.
  ///
  /// Note: Implementation of this function is optional and GetStats may be set to a nullptr.
  ///       If the Allocator is wrapping an internal ORT allocator that does not implement GetStats
  ///       the returned OrtKeyValuePairs instance will be empty.
  pub fn stats(self: *const @This()) !?*KeyValuePairs {
    var retval: ?*KeyValuePairs = null;
    try Error.check((self.underlying.GetStats orelse return null)(@ptrCast(self), @ptrCast(&retval)));
    return retval;
  }

  /// Allocate using a stream.
  /// If the allocator is stream aware this performs allocation using a stream.
  /// Alloc will be used if this is nullptr.
  ///
  /// self: Allocator instance
  /// size: Size of the allocation in bytes. nullptr if size was 0 or allocation failed.
  /// stream: The stream to allocate on.
  ///
  /// returns the pointer to an allocated block of `size` bytes
  ///
  /// Note: Implementation of this function is optional and AllocOnStream may be set to a nullptr.
  pub fn allocOnStream(self: *@This(), size: usize, stream: *SyncStream) ?[]u8 {
    const ptr = (self.underlying.AllocOnStream orelse return self.alloc(size))(@ptrCast(self), size, @ptrCast(stream));
    return @as([*]u8, @ptrCast(ptr orelse return null))[0 .. size];
  }

  pub fn register(self: *@This()) !void {
    try Error.check(Api.ort.RegisterAllocator.?(Api.env.underlying, @ptrCast(self)));
  }

  pub fn unregister(self: *@This()) !void {
    try self.info().unregisterAllocator();
  }

  pub fn create(session: *const Session, mem_info: *const MemoryInfo) !*Allocator {
    var out: ?*Allocator = null;
    try Error.check(Api.ort.CreateAllocator.?(@ptrCast(session), @ptrCast(mem_info), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Wraps GetAllocatorWithDefaultOptions
  pub fn getDefault() !*Allocator {
    var out: ?*Allocator = null;
    try Error.check(Api.ort.GetAllocatorWithDefaultOptions.?(@ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// deinitialises the allocator
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseAllocator.?(@ptrCast(self));
  }
};

/// High-level type information for an OrtValue.
pub const TypeInfo = opaque {
  pub const Map = opaque {
    pub fn getKeyType(self: *const @This()) !Value.Sub.Tensor.ElementDataType {
      var out: Value.Sub.Tensor.ElementDataType = undefined;
      try Error.check(Api.ort.GetMapKeyType.?(@ptrCast(self), @ptrCast(&out)));
      return out;
    }

    pub fn getValueType(self: *const @This()) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.GetMapValueType.?(@ptrCast(self), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseMapTypeInfo.?(@ptrCast(self));
    }
  };

  pub const Sequence = opaque {
    pub fn getElementType(self: *const @This()) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.GetSequenceElementType.?(@ptrCast(self), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseSequenceTypeInfo.?(@ptrCast(self));
    }
  };

  pub const Optional = opaque {
    pub fn getContainedType(self: *const @This()) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.GetOptionalContainedTypeInfo.?(@ptrCast(self), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }
  };

  /// Create an TypeInfo instance for a Tensor. Asserts that the ModelEditor api is initialized
  pub fn forTensor(tensor_info: *const TensorTypeAndShapeInfo.C) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateTensorTypeInfo.?(@ptrCast(tensor_info), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for a SparseTensor.
  pub fn forSparseTensor(tensor_info: *const TensorTypeAndShapeInfo.C) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateSparseTensorTypeInfo.?(@ptrCast(tensor_info), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for a Map.
  pub fn forMap(map_key_type: Value.Sub.Tensor.ElementDataType, map_value_type: *const TypeInfo) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateMapTypeInfo.?(@intFromEnum(map_key_type), @ptrCast(map_value_type), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for a Sequence.
  pub fn forSequence(sequence_type: *const TypeInfo) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateSequenceTypeInfo.?(@ptrCast(sequence_type), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for an Optional.
  pub fn forOptional(contained_type: *const TypeInfo) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateOptionalTypeInfo.?(@ptrCast(contained_type), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get ::OrtTensorTypeAndShapeInfo from an ::OrtTypeInfo
  /// Do not free the returned value, it is valid until type_info is freed.
  pub fn toTensorTypeAndShapeInfo(self: *const @This()) !?*const TensorTypeAndShapeInfo.C {
    var out: ?*const TensorTypeAndShapeInfo.C = null;
    try Error.check(Api.ort.CastTypeInfoToTensorInfo.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get ::TypeInfo.Type from ::OrtTypeInfo
  pub fn onnxType(self: *const @This()) !Value.Type {
    var out: Value.Type = undefined;
    try Error.check(Api.ort.GetOnnxTypeFromTypeInfo.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  pub fn castToMapTypeInfo(self: *const @This()) !?*const Map {
    var out: ?*const Map = null;
    try Error.check(Api.ort.CastTypeInfoToMapTypeInfo.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  pub fn castToSequenceTypeInfo(self: *const @This()) !?*const Sequence {
    var out: ?*const Sequence = null;
    try Error.check(Api.ort.CastTypeInfoToSequenceTypeInfo.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  pub fn castToOptionalTypeInfo(self: *const @This()) !?*const Optional {
    var out: ?*const Optional = null;
    try Error.check(Api.ort.CastTypeInfoToOptionalTypeInfo.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  pub fn getDenotation(self: *const @This()) ![]const u8 {
    var denotation: ?[*:0]const u8 = null;
    var len: usize = 0;
    try Error.check(Api.ort.GetDenotationFromTypeInfo.?(@ptrCast(self), @ptrCast(&denotation), &len));
    return if (denotation) |d| d[0..len] else "";
  }

  /// Release the TypeInfo object.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseTypeInfo.?(@ptrCast(self));
  }
};

/// The ThreadingOptions used for set global threadpools' options of The Env.
pub const ThreadingOptions = struct {
  /// This configures the global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools
  /// 0 means default, 1 means invoking thread will be used and no threads will be created in the pool
  intraop_threads: c_int = 0,
  /// This configures the global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools
  /// 0 means default, 1 means invoking thread will be used and no threads will be created in the pool
  interop_threads: c_int = 0,
  /// This will configure the global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools.
  /// Allow spinning of thread pools when their queues are empty. This will set the value for both inter_op and intra_op threadpools.
  ///
  /// false = It won't spin (recommended if CPU usage is high)
  /// true = Threadpool will spin to wait for queue to become non-empty
  allow_spinning: ?bool = null,
  /// Sets global thread pool options to be used in the call to OrtApi::CreateEnvWithGlobalThreadPools.
  /// Flush-to-zero and denormal-as-zero are applied to threads in both intra and inter global thread pool.
  /// Note: This option is not needed if the models used have no denormals. Having no denormals is recommended as this option may hurt model accuracy.
  ///
  /// Know more about denormalized floats here: https://en.wikipedia.org/wiki/IEEE_754-1985#Denormalized_numbers
  denormals_as_zero: bool = false,
  /// Set custom thread creation function for global thread pools
  custom_thread_creation_interface: ?ThreadCreationInterface = null,
  /// Custom function for joining threads
  custom_thread_join_function: ?*const fn(*const Api.c.OrtCustomHandleType) callconv(.c) void = null,
  /// Set affinities for intra op threads
  /// Affinity string follows format:
  ///   logical_processor_id,logical_processor_id;logical_processor_id,logical_processor_id
  ///   Semicolon isolates configurations among threads, while comma split processors where ith thread expected to attach to.
  ///   e.g. 1,2,3;4,5
  ///   specifies affinities for two threads, with the 1st thread attach to the 1st, 2nd, and 3rd processor, and 2nd thread to the 4th and 5th.
  ///   To ease the configuration, an "interval" is also allowed:
  ///   e.g. 1-8;8-16;17-24
  ///   orders that the 1st thread runs on first eight processors, 2nd thread runs on next eight processors, and so forth.
  ///   Note:
  ///   1. Once set, the number of thread affinities must equal to intra_op_num_threads - 1,
  ///      ort does not set affinity on the main thread which is started and managed by the calling app;
  ///   2. For windows, ort will infer the group id from a logical processor id, for example, assuming there are two groups with each has 64 logical processors,
  ///      an id of 64 will be inferred as the last processor of the 1st group, while 65 will be interpreted as the 1st processor of the second group.
  ///      Hence 64-65 is an invalid configuration, because a windows thread cannot be attached to processors across group boundary.
  ///
  /// since Version 1.14
  intraop_thread_affinity: ?[*:0]const u8 = null,

  const ThreadCreationInterface = struct {
    ptr: ?*anyopaque,
    create_fn: Api.c.OrtCustomCreateThreadFn,

    /// the instances must have a `pub fn create(self: *Self, worker: *const fn(?*anyopaque), arg: ?*anyopaque) *anytype`
    /// interface can be a pointer to a non-0 sized struct or an instance of 0 sized struct
    pub fn fromContext(instance: anytype) @This() {
      const T = @TypeOf(instance);
      const Sub = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
      return .{
        .ptr = if (@bitSizeOf(Sub) == 0) undefined else @ptrCast(instance),
        .create_fn = &struct {
          pub fn create(ctx: ?*anyopaque, worker: ?*const fn(?*anyopaque) void, arg: ?*anyopaque) Api.c.OrtCustomThreadHandle {
            const original: *Sub = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @ptrCast(ctx.?);
            return @ptrCast(original.create(worker, arg));
          }
        }.create,
      };
    }
  };

  pub const C = opaque {
    /// Wraps OrtApi::CreateThreadingOptions
    pub fn init() !*@This() {
      var self: ?*@This() = undefined;
      try Error.check(Api.ort.CreateThreadingOptions.?(@ptrCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::SetGlobalIntraOpNumThreads
    pub fn intraOpNumThreads(self: *@This(), num_threads: c_int) !void {
      try Error.check(Api.ort.SetGlobalIntraOpNumThreads.?(@ptrCast(self), num_threads));
    }

    /// Wraps OrtApi::SetGlobalInterOpNumThreads
    pub fn interOpNumThreads(self: *@This(), num_threads: c_int) !void {
      try Error.check(Api.ort.SetGlobalInterOpNumThreads.?(@ptrCast(self), num_threads));
    }

    /// Wraps OrtApi::SetGlobalSpinControl
    pub fn spinControl(self: *@This(), allow_spinning: bool) !void {
      try Error.check(Api.ort.SetGlobalSpinControl.?(@ptrCast(self), @intFromBool(allow_spinning)));
    }

    /// Wraps OrtApi::SetGlobalDenormalAsZero
    pub fn denormalAsZero(self: *@This()) !void {
      try Error.check(Api.ort.SetGlobalDenormalAsZero.?(@ptrCast(self)));
    }

    pub fn customThreadCreationInterface(self: *@This(), interface: ThreadCreationInterface) !void {
      try Error.check(Api.ort.SetGlobalCustomCreateThreadFn.?(@ptrCast(self), interface.create_fn));
      try Error.check(Api.ort.SetGlobalCustomThreadCreationOptions.?(@ptrCast(self), interface.ptr));
    }

    pub fn customThreadJoinFunction(self: *@This(), f: *const fn(*const Api.c.OrtCustomHandleType) callconv(.c) void) !void {
      try Error.check(Api.ort.SetGlobalCustomJoinThreadFn.?(@ptrCast(self), @ptrCast(f)));
    }

    pub fn intraOpThreadAffinity(self: *@This(), affinity_string: [*:0]const u8) !void {
      try Error.check(Api.ort.SetGlobalIntraOpThreadAffinity.?(@ptrCast(self), affinity_string));
    }

    /// Release the ThreadingOptions object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseThreadingOptions.?(@ptrCast(self));
    }
  };

  pub fn c(self: @This()) !*C {
    var retval = try C.init();
    errdefer retval.deinit();
    if (self.intraop_threads != 0) try retval.intraOpNumThreads(self.intraop_threads);
    if (self.interop_threads != 0) try retval.interOpNumThreads(self.interop_threads);
    if (self.allow_spinning) |spinning| try retval.spinControl(spinning);
    if (self.denormals_as_zero) try retval.denormalAsZero();
    if (self.custom_thread_creation_interface) |iface| try retval.customThreadCreationInterface(iface);
    if (self.custom_thread_join_function) |f| try retval.customThreadJoinFunction(f);
    if (self.intraop_thread_affinity) |str| try retval.intraOpThreadAffinity(str);
    return retval;
  }
};

/// Information about the shape and element type of a tensor.
pub const TensorTypeAndShapeInfo = struct {
  element_type: ?Value.Sub.Tensor.ElementDataType = null,
  dimensions: ?[]const i64 = null,
  names: ?[][*:0]const u8 = null,

  pub fn c(self: @This()) !*C {
    const retval = try C.init();
    if (self.element_type) |t| try retval.setElementType(t);
    if (self.dimensions) |dims| try retval.setDimensions(dims);
    if (self.names) |names| try retval.setSymbolicDimensions(names);
    return retval;
  }

  pub const C = opaque {
    /// Create an ::OrtTensorTypeAndShapeInfo object
    pub fn init() !*@This() {
      var self: *@This() = undefined;
      try Error.check(Api.ort.CreateTensorTypeAndShapeInfo.?(@ptrCast(&self)));
      return self;
    }

    pub fn getSymbolicDimensions(self: *const @This(), gpa: std.mem.Allocator) ![][*:0]const u8 {
      const count = try self.dimensionsCount();
      const out_ptr = try gpa.alloc([*:0]const u8, count);
      errdefer gpa.free(out_ptr);

      try Error.check(Api.ort.GetSymbolicDimensions.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
      return out_ptr;
    }

    /// Set element type in ::OrtTensorTypeAndShapeInfo
    pub fn setElementType(self: *@This(), dtype: Value.Sub.Tensor.ElementDataType) !void {
      try Error.check(Api.ort.SetTensorElementType.?(@ptrCast(self), @intFromEnum(dtype)));
    }

    /// Set shape information in ::OrtTensorTypeAndShapeInfo
    pub fn setDimensions(self: *@This(), dims: []const i64) !void {
      try Error.check(Api.ort.SetDimensions.?(@ptrCast(self), dims.ptr, dims.len));
    }

    /// Set shape information in ::OrtTensorTypeAndShapeInfo
    pub fn setSymbolicDimensions(self: *@This(), names: [][*:0]const u8) !void {
      try Error.check(Api.ort.SetSymbolicDimensions.?(@ptrCast(self), @ptrCast(names.ptr), names.len));
    }

    /// Get element type in ::OrtTensorTypeAndShapeInfo
    pub fn elementType(self: *const @This()) !Value.Sub.Tensor.ElementDataType {
      var out: Value.Sub.Tensor.ElementDataType = undefined;
      try Error.check(Api.ort.GetTensorElementType.?(@ptrCast(self), @ptrCast(&out)));
      return out;
    }

    /// Get dimension count in ::OrtTensorTypeAndShapeInfo
    pub fn dimensionsCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.GetDimensionsCount.?(@ptrCast(self), &out));
      return out;
    }

    /// Get dimensions in ::OrtTensorTypeAndShapeInfo
    pub fn getDimensions(self: *const @This(), out: []i64) !void {
      try Error.check(Api.ort.GetDimensions.?(@ptrCast(self), out.ptr, out.len));
    }

    /// Get total number of elements in a tensor shape
    /// For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
    pub fn shapeElementCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.GetTensorShapeElementCount.?(@ptrCast(self), &out));
      return out;
    }

    /// Release the TensorTypeAndShapeInfo object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseTensorTypeAndShapeInfo.?(@ptrCast(self));
    }
  };
};

/// Wrapper for any data type that can be passed to or returned from an ONNX session.
pub const Value = opaque {
  pub const Info = opaque {
    /// Get the value name.
    pub fn getName(self: *const @This()) ![*:0]const u8 {
      var out: ?[*:0]const u8 = null;
      try Error.check(Api.ort.GetValueInfoName.?(@ptrCast(self), @ptrCast(&out)));
      return out orelse "";
    }

    /// Get the type information.
    /// The returned TypeInfo must be deinitialized by the caller.
    pub fn getTypeInfo(self: *const @This()) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.GetValueInfoTypeInfo.?(@ptrCast(self), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Get the OrtNode that produces this value.
    /// Returns null if not produced by a node (e.g. graph input).
    /// Also returns the output index.
    pub fn getProducer(self: *const @This()) !?struct { node: *const Node, index: usize } {
      var node: ?*const Node = null;
      var idx: usize = 0;
      try Error.check(Api.ort.ValueInfo_GetValueProducer.?(@ptrCast(self), @ptrCast(&node), &idx));
      if (node) |n| return .{ .node = n, .index = idx };
      return null;
    }

    /// Returns a boolean indicating if the given value is a graph output.
    pub fn isGraphOutput(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsGraphOutput.?(@ptrCast(self), &out));
      return out;
    }

    /// Returns a boolean indicating if the given value is a required graph input.
    pub fn isRequiredGraphInput(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsRequiredGraphInput.?(@ptrCast(self), &out));
      return out;
    }

    /// Returns a boolean indicating if the given value is an optional graph input.
    pub fn isOptionalGraphInput(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsOptionalGraphInput.?(@ptrCast(self), &out));
      return out;
    }

    /// Returns a boolean indicating if the given value is a constant initializer.
    pub fn isConstantInitializer(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsConstantInitializer.?(@ptrCast(self), &out));
      return out;
    }

    /// Get the underlying initializer value.
    /// Returns null if this is not an initializer.
    /// Note: The returned Value lifetime is tied to the Graph/ValueInfo; do not deinit it.
    pub fn getInitializerValue(self: *const @This()) !?*const Value {
      var out: ?*const Value = null;
      try Error.check(Api.ort.ValueInfo_GetInitializerValue.?(@ptrCast(self), @ptrCast(&out)));
      return out;
    }

    /// Get the number of consumers of a value as a node input.
    /// Only nodes are considered "consumers".
    /// Wraps OrtApi::ValueInfo_GetValueNumConsumers
    pub fn getNumConsumers(self: *const @This()) !usize {
      var count: usize = 0;
      try Error.check(Api.ort.ValueInfo_GetValueNumConsumers.?(@ptrCast(self), &count));
      return count;
    }

    /// Returns information for all consumer nodes that use the value as an input.
    /// Buffers 'nodes' and 'input_indices' must be pre-allocated to the size returned by getNumConsumers().
    /// Index is set to -1 for an "implicit" input to a consumer node that contains a subgraph.
    /// Wraps OrtApi::ValueInfo_GetValueConsumers
    pub fn getConsumers(self: *const @This(), nodes: []*const Node, input_indices: []i64) !void {
      std.debug.assert(nodes.len == input_indices.len);
      try Error.check(Api.ort.ValueInfo_GetValueConsumers.?(
          @ptrCast(self),
          @ptrCast(nodes.ptr),
          input_indices.ptr,
          nodes.len,
      ));
    }

    /// Get information about an external initializer (filepath, offset, size).
    /// Returns null if this ValueInfo does not represent an external initializer.
    /// The returned ExternalInitializerInfo must be freed with deinit().
    /// Wraps OrtApi::ValueInfo_GetExternalInitializerInfo
    pub fn getExternalInitializerInfo(self: *const @This()) !?*ExternalInitializerInfo {
      var out: ?*ExternalInitializerInfo = null;
      try Error.check(Api.ort.ValueInfo_GetExternalInitializerInfo.?(@ptrCast(self), @ptrCast(&out)));
      return out;
    }

    /// Returns true if the value is defined in an outer scope (parent graph).
    /// Wraps OrtApi::ValueInfo_IsFromOuterScope
    pub fn isFromOuterScope(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsFromOuterScope.?(@ptrCast(self), &out));
      return out;
    }

    /// Release the ValueInfo.
    /// Do not call this if the ValueInfo has been passed to a Graph (SetInputs/SetOutputs),
    /// as the Graph takes ownership.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseValueInfo.?(@ptrCast(self));
    }
  };

  /// High-level categorization of ONNX objects (Tensors, Maps, Sequences, etc.).
  /// Synced with onnx TypeProto oneof
  pub const Type = enum(c_uint) {
    UNKNOWN = @bitCast(Api.c.ONNX_TYPE_UNKNOWN),
    TENSOR = @bitCast(Api.c.ONNX_TYPE_TENSOR),
    SEQUENCE = @bitCast(Api.c.ONNX_TYPE_SEQUENCE),
    MAP = @bitCast(Api.c.ONNX_TYPE_MAP),
    OPAQUE = @bitCast(Api.c.ONNX_TYPE_OPAQUE),
    SPARSETENSOR = @bitCast(Api.c.ONNX_TYPE_SPARSETENSOR),
    OPTIONAL = @bitCast(Api.c.ONNX_TYPE_OPTIONAL),

    pub fn Type(comptime self: @This()) type {
      return switch (self) {
        .UNKNOWN => Value,
        .TENSOR => Sub.Tensor,
        .SEQUENCE => Sub.Sequence,
        .MAP => Sub.Map,
        .OPAQUE => Sub.Opaque,
        .SPARSETENSOR => Sub.SparseTensor,
        .OPTIONAL => Sub.Optional,
      };
    }
  };

  pub const Sub = opaque {
    pub const Tensor = opaque {
      /// Defines the underlying data type of elements within a tensor (e.g., f32, i64).
      /// @intCast(Api.c copied it from TensorProto::DataType [Refer to the Api.c comment])
      pub const ElementDataType = enum(c_uint) {
        undefined = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED),
        f32 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), // maps to c type float
        u8 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8), // maps to c type uint8_t
        i8 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8), // maps to c type int8_t
        u16 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16), // maps to c type uint16_t
        i16 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16), // maps to c type int16_t
        i32 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32), // maps to c type int32_t
        i64 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), // maps to c type int64_t
        string = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING), // maps to c++ type std::string
        bool = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
        f16 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
        f64 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE), // maps to c type double
        u32 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32), // maps to c type uint32_t
        u64 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64), // maps to c type uint64_t
        c64 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64), // complex with float32 real and imaginary components
        c128 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128), // complex with float64 real and imaginary components
        bf16 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16), // Non-IEEE floating-point format based on IEEE754 single-precision
        // float 8 types were introduced in onnx 1.14, see https://onnx.ai/onnx/technical/float8.html
        f8e4m3 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN), // Non-IEEE floating-point format based on IEEE754 single-precision
        f8e4m3uz = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ), // Non-IEEE floating-point format based on IEEE754 single-precision
        f8e5m2 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2), // Non-IEEE floating-point format based on IEEE754 single-precision
        f8e5m2uz = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ), // Non-IEEE floating-point format based on IEEE754 single-precision
        // Int4 types were introduced in ONNX 1.16. See https://onnx.ai/onnx/technical/int4.html
        u4_x2 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4), // maps to a pair of packed uint4 values (size == 1 byte)
        i4_x2 = @bitCast(Api.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4), // maps to a pair of packed int4 values (size == 1 byte)
      };

      /// Create a tensor using a supplied ::Allocator
      pub fn init(allocator: *Allocator, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateTensorAsOrtValue.?(@ptrCast(allocator), shape.ptr, shape.len, @intFromEnum(dtype), @ptrCast(&self)));
        return self orelse error.OutOfMemory;
      }

      /// Create a tensor backed by a user supplied buffer
      /// p_data is owned by caller. deinit() won't release p_data.
      pub fn initWithData(info: *const Allocator.MemoryInfo, p_data: []u8, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateTensorWithDataAsOrtValue.?(@ptrCast(info), @ptrCast(p_data.ptr), p_data.len, shape.ptr, shape.len, @intFromEnum(dtype), @ptrCast(&self)));
        return self orelse error.OutOfMemory;
      }

      /// Create an OrtValue for a Tensor that uses pre-existing memory.
      /// ORT will take ownership of the memory and free it using the provided deleter when no longer in use.
      pub fn initWithDataAndDeleter(deleter: *Allocator, p_data: []u8, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateTensorWithDataAndDeleterAsOrtValue.?(@ptrCast(deleter), @ptrCast(p_data.ptr), p_data.len, shape.ptr, shape.len, @intFromEnum(dtype), @ptrCast(&self)));
        return self orelse error.OutOfMemory;
      }

      /// Get a pointer to the raw data inside a tensor
      /// Used to read/write/modify the internal tensor data directly.
      pub fn getData(self: *@This(), comptime T: type) ![]T {
        var ptr: ?[*]T = null;
        try Error.check(Api.ort.GetTensorMutableData.?(@ptrCast(self), @ptrCast(&ptr)));
        return (ptr orelse error.OutOfMemory)[0 .. try self.getTensorSizeInBytes() / @sizeOf(T)];
      }

      /// Get a const pointer to the raw data inside a tensor
      pub fn getDataConst(self: *const @This(), comptime T: type) ![]const u8 {
        var ptr: ?[*]const u8 = null;
        try Error.check(Api.ort.GetTensorData.?(@ptrCast(self), @ptrCast(&ptr)));
        return (ptr orelse error.OutOfMemory)[0 .. try self.getTensorSizeInBytes() / @sizeOf(T)];
      }

      /// Compute total size in bytes of the tensor data contained in an OrtValue.
      pub fn getSizeInBytes(self: *@This()) !usize {
        var out: usize = 0;
        try Error.check(Api.ort.GetTensorSizeInBytes.?(@ptrCast(self), &out));
        return out;
      }

      /// Get type and shape information from a tensor ::OrtValue
      pub fn getTypeAndShape(self: *@This()) !*TensorTypeAndShapeInfo.C {
        var out: ?*TensorTypeAndShapeInfo.C = null;
        try Error.check(Api.ort.GetTensorTypeAndShape.?(@ptrCast(self), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      pub fn at(self: *@This(), indices: []i64, comptime T: type) !*T {
        var out: ?*T = null;
        try Error.check(Api.ort.TensorAt.?(@ptrCast(self), indices.ptr, indices.len, @ptrCast(&out)));
        return out.?; // returns pointer in memory so this can't be null
      }

      /// Returns a pointer to the ::OrtMemoryInfo of a Tensor
      pub fn getMemoryInfo(self: *@This()) !*const Allocator.MemoryInfo {
        var out: ?*const Allocator.MemoryInfo = null;
        try Error.check(Api.ort.GetTensorMemoryInfo.?(@ptrCast(self), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      pub const String = opaque {
        /// Get total byte length for all strings in a string tensor
        pub fn getStringDataLength(self: *@This()) !usize {
          var out: usize = 0;
          try Error.check(Api.ort.GetStringTensorDataLength.?(@ptrCast(self), &out));
          return out;
        }

        pub fn getStringContent(self: *@This(), out_bytes: []u8, offsets: []usize) !void {
          try Error.check(Api.ort.GetStringTensorContent.?(@ptrCast(self), @ptrCast(out_bytes.ptr), out_bytes.len, offsets.ptr, offsets.len));
        }

        /// Set all strings at once in a string tensor
        pub fn fillString(self: *@This(), strings: []const [*:0]const u8) !void {
          try Error.check(Api.ort.FillStringTensor.?(@ptrCast(self), @ptrCast(strings.ptr), strings.len));
        }

        pub fn getStringElementLength(self: *const @This(), index: usize) !usize {
          var out: usize = undefined;
          try Error.check(Api.ort.GetStringTensorElementLength.?(@ptrCast(self), index, &out));
          return out;
        }

        pub fn getStringElement(self: *const @This(), index: usize, out_bytes: []u8) !void {
          try Error.check(Api.ort.GetStringTensorElement.?(@ptrCast(self), out_bytes.len, index, @ptrCast(out_bytes.ptr)));
        }

        /// Set a single string in a string tensor
        pub fn fillStringElement(self: *@This(), s: [*:0]const u8, index: usize) !void {
          try Error.check(Api.ort.FillStringTensorElement.?(@ptrCast(self), s, index));
        }

        pub fn getResizedStringElementBuffer(self: *@This(), index: usize, len: usize) ![]u8 {
          var out: ?[*]u8 = null;
          try Error.check(Api.ort.GetResizedStringTensorElementBuffer.?(@ptrCast(self), index, len, @ptrCast(&out)));
          return (out orelse return error.OutOfMemory)[0 .. len];
        }
      };

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }
    };

    pub const Sequence = opaque {
      /// Create a Value representing a Sequence (ONNX_TYPE_SEQUENCE).
      pub fn init(values: []const *const Value) !*@This() {
        return @ptrCast(try Value._init(values, .SEQUENCE));
      }

      pub fn getValue(self: *@This(), index: c_int, allocator: *Allocator) !*Value {
        return self.toValue()._getValue(index, allocator);
      }

      pub fn getCount(self: *@This()) !usize {
        return self.toValue()._getCount();
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }
    };

    pub const Map = opaque {
      /// Create a Value representing a Map (ONNX_TYPE_MAP).
      /// keys: A Tensor Value containing keys.
      /// values: A Tensor Value containing values.
      /// The API ref-counts the inputs; you may deinit them after this call if you don't need them elsewhere.
      pub fn init(keys: *const Value, values: *const Value) !*@This() {
        return @ptrCast(try Value._init(&[2]*const Value{ keys, values }, .MAP));
      }

      ///  At index 0 is key, 1 is value
      pub fn getKV(self: *const @This(), allocator: *Allocator) !std.meta.Tuple(&.{*Value, *Value}) {
        const v = self.toValue();
        const keys = try v._getValue(0, allocator);
        errdefer keys.deinit();
        return .{ keys, try v._getValue(1, allocator), };
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }
    };

    pub const Opaque = opaque {
      /// Create a Value wrapping an Opaque type.
      pub fn init(domain_name: [*:0]const u8, type_name: [*:0]const u8, data: []const u8) !*@This() {
        var out: ?*@This() = null;
        try Error.check(Api.ort.CreateOpaqueValue.?(domain_name, type_name, data.ptr, data.len, @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Get data from an Opaque Value.
      /// buffer: Buffer to write data into. Must match the internal size.
      pub fn getData(self: *const @This(), comptime Out: type, out: []Out, domain_name: [*:0]const u8, type_name: [*:0]const u8) !void {
        try Error.check(Api.ort.GetOpaqueValue.?(domain_name, type_name, @ptrCast(self), @ptrCast(out.ptr), out.len * @sizeOf(Out)));
      }

      pub fn getDataAlloc(self: *const @This(), comptime Out: type, len: usize, gpa: std.mem.Allocator, domain_name: [*:0]const u8, type_name: [*:0]const u8) ![]u8 {
        const out = try gpa.alloc(Out, len);
        errdefer gpa.free(out);
        try self.getData(Out, out, domain_name, type_name);
        return out;
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }
    };

    pub const SparseTensor = opaque {
      /// Identifies the storage format for sparse tensors (COO, CSR, or Block Sparse).
      /// These types are synced with internal SparseFormatFlags
      pub const Format = enum(c_uint) {
        SPARSE_UNDEFINED = @intCast(Api.c.ORT_SPARSE_UNDEFINED),
        SPARSE_COO = @intCast(Api.c.ORT_SPARSE_COO),
        SPARSE_CSRC = @intCast(Api.c.ORT_SPARSE_CSRC),
        BLOCK_SPARSE = @intCast(Api.c.ORT_SPARSE_BLOCK_SPARSE),
      };

      /// Identifies which specific indices buffer of a sparse tensor is being queried.
      /// Enum allows to query sparse tensor indices
      pub const IndicesFormat = enum(c_uint) {
        COO_INDICES = @intCast(Api.c.ORT_SPARSE_COO_INDICES),
        CSR_INNER_INDICES = @intCast(Api.c.ORT_SPARSE_CSR_INNER_INDICES),
        CSR_OUTER_INDICES = @intCast(Api.c.ORT_SPARSE_CSR_OUTER_INDICES),
        BLOCK_SPARSE_INDICES = @intCast(Api.c.ORT_SPARSE_BLOCK_SPARSE_INDICES),
      };

      pub fn init(allocator: *Allocator, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateSparseTensorAsOrtValue.?(@ptrCast(allocator), shape.ptr, shape.len, @intFromEnum(dtype), @ptrCast(&self)));
        return self orelse error.OutOfMemory;
      }

      pub fn initWithValues(info: *const Allocator.MemoryInfo, p_data: ?[*]u8, dense_shape: []i64, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateSparseTensorWithValuesAsOrtValue.?(
            @ptrCast(info),
            @ptrCast(p_data),
            dense_shape.ptr,
            dense_shape.len,
            shape.ptr,
            shape.len,
            @intFromEnum(dtype),
            @ptrCast(&self)
        ));
        return self orelse error.OutOfMemory;
      }

      /// Fill a sparse tensor with COO (Coordinate) data.
      /// Performs allocation and copy.
      /// mem_info: Location of the source data.
      /// values_shape: Shape of the values.
      /// values: Pointer to values (or char** for strings).
      /// indices: Pointer to int64 indices.
      pub fn fillCoo(self: *@This(), mem_info: *const Allocator.MemoryInfo, values_shape: []const i64, values: [*]const u8, indices: []const i64) !void {
        try Error.check(Api.ort.FillSparseTensorCoo.?(
          @ptrCast(self),
          @ptrCast(mem_info),
          values_shape.ptr, values_shape.len,
          @ptrCast(values),
          indices.ptr, indices.len
        ));
      }

      /// Fill a sparse tensor with CSR (Compressed Sparse Row) data.
      pub fn fillCsr(
        self: *@This(),
        mem_info: *const Allocator.MemoryInfo,
        values_shape: []const i64,
        values: [*]const u8,
        inner_indices: []const i64,
        outer_indices: []const i64,
      ) !void {
        try Error.check(Api.ort.FillSparseTensorCsr.?(
          @ptrCast(self),
          @ptrCast(mem_info),
          values_shape.ptr, values_shape.len,
          @ptrCast(values),
          inner_indices.ptr, inner_indices.len,
          outer_indices.ptr, outer_indices.len
        ));
      }

      /// Fill a sparse tensor with Block Sparse data.
      pub fn fillBlockSparse(
        self: *@This(), mem_info: *const Allocator.MemoryInfo,
        values_shape: []const i64,
        values: *const anyopaque,
        indices_shape: []const i64,
        indices: []const i32,
      ) !void {
        try Error.check(Api.ort.FillSparseTensorBlockSparse.?(
          @ptrCast(self),
          @ptrCast(mem_info),
          values_shape.ptr, values_shape.len,
          @ptrCast(values),
          indices_shape.ptr, indices_shape.len,
          indices.ptr
        ));
      }

      /// Use user-supplied COO indices.
      /// indices: User buffer. Must outlive Value.
      pub fn useCooIndices(self: *@This(), indices: []i64) !void {
        try Error.check(Api.ort.UseCooIndices.?(@ptrCast(self), indices.ptr, indices.len));
      }

      /// Use user-supplied CSR indices.
      pub fn useCsrIndices(self: *@This(), inner: []i64, outer: []i64) !void {
        try Error.check(Api.ort.UseCsrIndices.?(@ptrCast(self), inner.ptr, inner.len, outer.ptr, outer.len));
      }

      /// Use user-supplied Block Sparse indices.
      pub fn useBlockSparseIndices(self: *@This(), indices_shape: []const i64, indices: []i32) !void {
        try Error.check(Api.ort.UseBlockSparseIndices.?(@ptrCast(self), indices_shape.ptr, indices_shape.len, indices.ptr));
      }

      /// Get the sparse format of the value.
      pub fn getSparseFormat(self: *const @This()) !*Format {
        var out: ?*Format = null;
        try Error.check(Api.ort.GetSparseTensorFormat.?(@ptrCast(self), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Get type info for the sparse tensor values.
      /// Caller must deinit the returned TensorTypeAndShapeInfo.
      pub fn getValuesTypeAndShape(self: *@This()) !*TensorTypeAndShapeInfo.C {
        var out: ?*TensorTypeAndShapeInfo.C = null;
        try Error.check(Api.ort.GetSparseTensorValuesTypeAndShape.?(@ptrCast(self), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Get pointer to sparse values.
      pub fn getValues(self: *@This()) ![*]const u8 {
        var ptr: ?[*]const u8 = null;
        try Error.check(Api.ort.GetSparseTensorValues.?(@ptrCast(self), @ptrCast(&ptr)));
        return ptr orelse error.OutOfMemory;
      }

      /// Get type info for the sparse tensor indices.
      /// Caller must deinit the returned TensorTypeAndShapeInfo.
      pub fn getIndicesTypeShape(self: *const @This(), format: IndicesFormat) !*TensorTypeAndShapeInfo.C {
        var out: ?*TensorTypeAndShapeInfo.C = null;
        try Error.check(Api.ort.GetSparseTensorIndicesTypeShape.?(@ptrCast(self), @intFromEnum(format), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Get pointer to sparse indices.
      pub fn getIndices(self: *const @This(), format: IndicesFormat) ![]u8 {
        var ptr: ?[*]u8 = null;
        var count: usize = 0;
        try Error.check(Api.ort.GetSparseTensorIndices.?(@ptrCast(self), @intFromEnum(format), &count, @ptrCast(&ptr)));
        return (ptr orelse return error.OutOfMemory)[0 .. count];
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }
    };

    pub const Optional = opaque {
      /// Returns true if an optional type OrtValue has an element
      pub fn hasValue(self: *@This()) !bool {
        var out: c_int = 0;
        try Error.check(Api.ort.HasValue.?(@ptrCast(self), &out));
        return out != 0;
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }
    };
  };

  pub fn _init(values: []const *const Value, t: Type) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateValue.?(@ptrCast(values.ptr), values.len, @intFromEnum(t), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn AsUnion(SelfPtr: type) type {
    return union(Type) {
      UNKNOWN: Utils.CopyPointerAttrs(SelfPtr, .one, Value),
      TENSOR: Utils.CopyPointerAttrs(SelfPtr, .one, Sub.Tensor),
      SEQUENCE: Utils.CopyPointerAttrs(SelfPtr, .one, Sub.Sequence),
      MAP: Utils.CopyPointerAttrs(SelfPtr, .one, Sub.Map),
      OPAQUE: Utils.CopyPointerAttrs(SelfPtr, .one, Sub.Opaque),
      SPARSETENSOR: Utils.CopyPointerAttrs(SelfPtr, .one, Sub.SparseTensor),
      OPTIONAL: Utils.CopyPointerAttrs(SelfPtr, .one, Sub.Optional),
    };
  }

  pub fn asUnion(self_ptr: anytype) AsUnion(@TypeOf(self_ptr)) {
    const U = AsUnion(@TypeOf(self_ptr));
    switch (try self_ptr.getType()) {
      inline else => |t| return @unionInit(U, @tagName(t), @ptrCast(self_ptr)),
    }
    unreachable;
  }

  /// You may use inline switching to get a comptime value for t and then cast to appropriate type
  pub fn asType(self: *@This(), comptime t: Type) t.Type() {
    return @ptrCast(self);
  }

  /// Get TypeInfo.Type of an ::OrtValue
  pub fn getType(self: *const @This()) !Type {
    var out: Type = undefined;
    try Error.check(Api.ort.GetValueType.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  /// Get type information of an OrtValue
  pub fn getTypeInfo(self: *@This()) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.GetTypeInfo.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get a non-tensor element from a Value (e.g., an element of a Sequence or Map).
  /// If Map: index 0 = keys, index 1 = values.
  /// If Sequence: index i = i-th element.
  /// The returned Value must be deinitialized by the caller.
  pub fn _getValue(self: *const @This(), index: c_int, allocator: *Allocator) !*@This() {
    var out: ?*Value = null;
    try Error.check(Api.ort.GetValue.?(@ptrCast(self), index, @ptrCast(allocator), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get the count of elements.
  /// If Map: returns 2.
  /// If Sequence: returns number of elements.
  pub fn _getCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.GetValueCount.?(@ptrCast(self), &out));
    return out;
  }

  /// Return true if this ::OrtValue is a tensor type
  pub fn isTensor(self: *const @This()) !bool {
    var out: c_int = 0;
    try Error.check(Api.ort.IsTensor.?(@ptrCast(self), &out));
    return out != 0;
  }

  /// Returns true if this ::OrtValue is a SparseTensor
  pub fn isSparseTensor(self: *const @This()) !bool {
    var out: c_int = 0;
    try Error.check(Api.ort.IsSparseTensor.?(@ptrCast(self), &out));
    return out != 0;
  }

  /// Release the OrtValue.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseValue.?(@ptrCast(self));
  }

  /// Warning: This uses Ep.api
  /// Get the OrtMemoryDevice from an OrtValue instance if it contains a Tensor.
  ///
  /// value: The OrtValue instance to get the memory device from.
  /// returns: Memory device if OrtValue contains a Tensor, nullptr otherwise.
  ///
  /// since Version 1.23.
  pub fn getMemoryDevice(self: *const @This()) !*const Allocator.MemoryDevice {
    return @ptrCast(Ep.api.underlying.Value_GetMemoryDevice.?(@ptrCast(self)) orelse return error.OutOfMemory);
  }
};

pub const Node = opaque {
  /// Create an OrtNode to add to an OrtGraph.
  ///
  /// Create attributes with `OpAttr.init`. OrtOpAttr instances are copied by the node
  /// creation process, so the caller retains ownership of the passed attributes.
  ///
  /// Note: This function requires the Model Editor API to be initialized in `Api.init`.
  ///
  /// param operator_name: The name of the operator (e.g. "Add", "Conv").
  /// param domain_name: The domain of the operator. Use "" for default ONNX operators.
  /// param node_name: The name of the node.
  /// param input_names: The names of the inputs.
  /// param output_names: The names of the outputs.
  /// param attributes: Slice of pointers to optional attributes. May be empty.
  pub fn init(
    operator_name: [*:0]const u8,
    domain_name: [*:0]const u8,
    node_name: [*:0]const u8,
    input_names: []const [*:0]const u8,
    output_names: []const [*:0]const u8,
    attributes: []*OpAttr,
  ) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateNode.?(
      operator_name,
      domain_name,
      node_name,
      @ptrCast(input_names.ptr),
      input_names.len,
      @ptrCast(output_names.ptr),
      output_names.len,
      @ptrCast(attributes.ptr),
      attributes.len,
      @ptrCast(&out),
    ));
    return out orelse error.OutOfMemory;
  }

  /// Returns a node's identifier.
  /// The node's identifier is only unique in the node's parent graph.
  pub fn getId(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetId.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns a node's name. Can be an empty string.
  pub fn getName(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetName.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse "";
  }

  /// Returns a node's operator type (e.g., "Conv").
  pub fn getOperatorType(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetOperatorType.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse "";
  }

  /// Returns a node's domain name.
  pub fn getDomain(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetDomain.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse "";
  }

  /// Get the opset version in which the given node's operator type was first defined.
  pub fn getSinceVersion(self: *const @This()) !c_int {
    var out: c_int = 0;
    try Error.check(Api.ort.Node_GetSinceVersion.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the number of node inputs.
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumInputs.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the node's inputs as ValueInfo instances.
  /// The returned slice is allocated using the provided allocator.
  pub fn getInputs(self: *const @This(), gpa: std.mem.Allocator) ![]*const Value.Info {
    const count = try self.getInputCount();
    const out_ptr = try gpa.alloc(*const Value.Info, count);
    errdefer gpa.free(out_ptr);
    try Error.check(Api.ort.Node_GetInputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Returns the number of node outputs.
  pub fn getOutputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumOutputs.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the node's outputs as ValueInfo instances.
  /// The returned slice is allocated using the provided allocator.
  pub fn getOutputs(self: *const @This(), gpa: std.mem.Allocator) ![]*const Value.Info {
    const count = try self.getOutputCount();
    const out_ptr = try gpa.alloc(*const Value.Info, count);
    errdefer gpa.free(out_ptr);

    try Error.check(Api.ort.Node_GetOutputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Returns the number of node implicit inputs.
  pub fn getImplicitInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumImplicitInputs.?(@ptrCast(self), &out));
    return out;
  }

  /// Get the implicit inputs, as ValueInfo instances, that are used within the given node's subgraphs.
  /// The returned slice is allocated using the provided gpa.
  pub fn getImplicitInputs(self: *const @This(), gpa: std.mem.Allocator) ![]*const Value.Info {
    const count = try self.getImplicitInputCount();
    const out_ptr = try gpa.alloc(*const Value.Info, count);
    errdefer gpa.free(out_ptr);

    try Error.check(Api.ort.Node_GetImplicitInputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Returns the number of node attributes.
  pub fn getAttributeCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumAttributes.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns a node's attributes as OpAttr instances.
  /// The returned slice is allocated using the provided gpa.
  pub fn getAttributes(self: *const @This(), gpa: std.mem.Allocator) ![]*const OpAttr {
    const count = try self.getAttributeCount();
    const out_ptr = try gpa.alloc(*const OpAttr, count);
    errdefer gpa.free(out_ptr);

    try Error.check(Api.ort.Node_GetAttributes.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Gets the Node's attribute as OpAttr by name.
  /// Returns null if the attribute is not found or is an unset optional attribute.
  pub fn getAttributeByName(self: *const @This(), name: [*:0]const u8) !?*const OpAttr {
    var out: ?*const OpAttr = null;
    // ORT returns ORT_NOT_FOUND if the attribute doesn't exist, which we translate to returning null
    // However, if we blindly check(...), it will return error.NotFound.
    // We can use Error.check, catch NotFound, and return null.
    Error.check(Api.ort.Node_GetAttributeByName.?(@ptrCast(self), name, @ptrCast(&out))) catch |err| switch (err) {
      Error.Set.OrtErrorNotFound => return null,
      else => return err,
    };

    return out;
  }

  /// Returns the number of subgraphs contained by the given node.
  pub fn getSubgraphCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumSubgraphs.?(@ptrCast(self), &out));
    return out;
  }

  pub const SubgraphInfo = struct {
    graph: *const Graph,
    attribute_name: ?[*:0]const u8,
  };

  /// Get the subgraphs, as Graph instances, contained by the given node.
  /// Also returns the attribute name associated with each subgraph.
  /// The returned slice is allocated using the provided allocator.
  pub fn getSubgraphs(self: *const @This(), allocator: std.mem.Allocator) ![]SubgraphInfo {
    const count = try self.getSubgraphCount();
    const graphs = try allocator.alloc(*const Graph, count);
    defer allocator.free(graphs);
    const names = try allocator.alloc(?[*:0]const u8, count);
    defer allocator.free(names);

    try Error.check(Api.ort.Node_GetSubgraphs.?(@ptrCast(self), @ptrCast(graphs.ptr), count, @ptrCast(names.ptr)));

    const out = try allocator.alloc(SubgraphInfo, count);
    for (0..count) |i| {
      out[i] = .{ .graph = graphs[i], .attribute_name = names[i] };
    }
    return out;
  }

  /// Get the node's parent Graph instance.
  /// Can return null if the Node was created without an owning graph.
  pub fn getGraph(self: *const @This()) !?*const Graph {
    var out: ?*const Graph = null;
    try Error.check(Api.ort.Node_GetGraph.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  /// Returns the execution provider name that this node is assigned to run on.
  /// Returns null if the node has not been assigned to any execution provider yet.
  pub fn getEpName(self: *const @This()) !?[*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetEpName.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  /// Release an OrtNode if it was not added to an OrtGraph.
  /// Do not call this if the node has been added to a graph.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseNode.?(@ptrCast(self));
  }
};

pub const Graph = opaque {
  /// Create an empty OrtGraph.
  /// Note: Requires Model Editor API initialization.
  pub fn init() !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateGraph.?(@ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Set the inputs for the OrtGraph.
  /// This will replace any existing inputs with the new values.
  /// The OrtGraph takes ownership of the Value.Info instances; do NOT call deinit on them.
  pub fn setInputs(self: *@This(), inputs: []*Value.Info) !void {
    try Error.check(Api.editor.underlying.SetGraphInputs.?(
      @ptrCast(self), 
      @ptrCast(inputs.ptr), 
      inputs.len
    ));
  }

  /// Set the outputs for the OrtGraph.
  /// This will replace any existing outputs with the new values.
  /// The OrtGraph takes ownership of the Value.Info instances; do NOT call deinit on them.
  pub fn setOutputs(self: *@This(), outputs: []*Value.Info) !void {
    try Error.check(Api.editor.underlying.SetGraphOutputs.?(
      @ptrCast(self), 
      @ptrCast(outputs.ptr), 
      outputs.len
    ));
  }

  /// Add an initializer to the OrtGraph.
  /// The OrtGraph takes ownership of the OrtValue; do NOT call deinit on it.
  /// 
  /// name: The value name for the initializer.
  /// tensor: The OrtValue instance containing the tensor data.
  /// data_is_external: Set to true if the data is external (e.g. mmap) and should not be copied.
  pub fn addInitializer(self: *@This(), name: [*:0]const u8, tensor: *Value, data_is_external: bool) !void {
    try Error.check(Api.editor.underlying.AddInitializerToGraph.?(
      @ptrCast(self),
      name,
      @ptrCast(tensor),
      data_is_external
    ));
  }

  /// Add an OrtNode to the OrtGraph.
  /// The OrtGraph takes ownership of OrtNode; do NOT call deinit on it.
  pub fn addNode(self: *@This(), node: *Node) !void {
    try Error.check(Api.editor.underlying.AddNodeToGraph.?(
      @ptrCast(self),
      @ptrCast(node)
    ));
  }

  /// Returns the graph's name.
  pub fn getName(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Graph_GetName.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse "";
  }

  /// Get the filepath to the model from which this OrtGraph was constructed.
  /// Returns empty string if unknown (e.g. created from memory).
  pub fn getModelPath(self: *const @This()) !Utils.Path {
    var out: ?Utils.Path = null;
    try Error.check(Api.ort.Graph_GetModelPath.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse @ptrCast(""); // Assuming empty string literal is compatible or mapped
  }

  /// Returns the ONNX IR version.
  pub fn getOnnxIRVersion(self: *const @This()) !i64 {
    var out: i64 = 0;
    try Error.check(Api.ort.Graph_GetOnnxIRVersion.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the number of operator sets that the graph's model uses.
  pub fn getOperatorSetCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumOperatorSets.?(@ptrCast(self), &out));
    return out;
  }

  pub const OperatorSet = struct {
    domain: [*:0]const u8,
    version: i64,
  };

  /// Returns the operator sets that the graph's model uses.
  /// The returned slice is allocated using the provided gpa.
  pub fn getOperatorSets(self: *const @This(), gpa: std.mem.Allocator) ![]OperatorSet {
    const count = try self.getOperatorSetCount();
    
    // Temporary arrays for C API
    const domains = try gpa.alloc([*:0]const u8, count);
    defer gpa.free(domains);
    const versions = try gpa.alloc(i64, count);
    defer gpa.free(versions);

    try Error.check(Api.ort.Graph_GetOperatorSets.?(
      @ptrCast(self), 
      @ptrCast(domains.ptr), 
      @ptrCast(versions.ptr), 
      count
    ));

    // Package into Zig struct
    const out = try gpa.alloc(OperatorSet, count);
    for (0..count) |i| {
      out[i] = .{ .domain = domains[i], .version = versions[i] };
    }
    return out;
  }

  /// Returns the number of graph inputs.
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumInputs.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the graph's inputs as Value.Info instances.
  /// The returned slice is allocated using the provided gpa.
  pub fn getInputs(self: *const @This(), gpa: std.mem.Allocator) ![]*const Value.Info {
    const count = try self.getInputCount();
    const out_ptr = try gpa.alloc(*const Value.Info, count);
    errdefer gpa.free(out_ptr);

    try Error.check(Api.ort.Graph_GetInputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Returns the number of graph outputs.
  pub fn getOutputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumOutputs.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the graph's outputs as Value.Info instances.
  /// The returned slice is allocated using the provided gpa.
  pub fn getOutputs(self: *const @This(), gpa: std.mem.Allocator) ![]*const Value.Info {
    const count = try self.getOutputCount();
    const out_ptr = try gpa.alloc(*const Value.Info, count);
    errdefer gpa.free(out_ptr);

    try Error.check(Api.ort.Graph_GetOutputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Returns the number of graph initializers.
  pub fn getInitializerCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumInitializers.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the graph's initializers as Value.Info instances.
  /// To get the actual data, call `Value.Info.getInitializerValue`.
  /// The returned slice is allocated using the provided gpa.
  pub fn getInitializers(self: *const @This(), gpa: std.mem.Allocator) ![]*const Value.Info {
    const count = try self.getInitializerCount();
    const out_ptr = try gpa.alloc(*const Value.Info, count);
    errdefer gpa.free(out_ptr);

    try Error.check(Api.ort.Graph_GetInitializers.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Returns the number of graph nodes.
  pub fn getNodeCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumNodes.?(@ptrCast(self), &out));
    return out;
  }

  /// Returns the graph's nodes as Node instances.
  /// The returned slice is allocated using the provided gpa.
  pub fn getNodes(self: *const @This(), gpa: std.mem.Allocator) ![]*const Node {
    const count = try self.getNodeCount();
    const out_ptr = try gpa.alloc(*const Node, count);
    errdefer gpa.free(out_ptr);

    try Error.check(Api.ort.Graph_GetNodes.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Get the parent node for the given graph, if any exists.
  /// Returns null if this is a top-level graph.
  pub fn getParentNode(self: *const @This()) !?*const Node {
    var out: ?*const Node = null;
    try Error.check(Api.ort.Graph_GetParentNode.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  /// Returns an OrtGraph that contains a subset of nodes in the source OrtGraph.
  /// The lifetime of the returned Graph view is tied to the source Graph.
  /// Note: The returned graph must be released via `deinit`.
  pub fn getGraphView(self: *const @This(), nodes: []*const Node) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.Graph_GetGraphView.?(
      @ptrCast(self),
      @ptrCast(nodes.ptr),
      nodes.len,
      @ptrCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  /// Get ModelMetadata from the OrtGraph.
  /// The returned metadata must be released by the caller.
  pub fn getModelMetadata(self: *const @This()) !*Model.Metadata {
    var out: ?*Model.Metadata = null;
    try Error.check(Api.ort.Graph_GetModelMetadata.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Release an OrtGraph.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseGraph.?(@ptrCast(self));
  }
};

pub const Model = opaque {
  /// Create an OrtModel.
  ///
  /// This can be used to build a new model, or to augment an existing model.
  /// If augmenting an existing model add additional domains/opsets if needed.
  ///
  /// domain_names: The domain names for the model (e.g. "", "ai.onnx.ml").
  /// opset_versions: The opset versions for the model.
  ///
  /// Note: Requires Model Editor API initialization.
  pub fn init(domain_names: []const [*:0]const u8, opset_versions: []const c_int) !*@This() {
    std.debug.assert(domain_names.len == opset_versions.len);
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateModel.?(
      @ptrCast(domain_names.ptr),
      @ptrCast(opset_versions.ptr),
      domain_names.len,
      @ptrCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  /// Add an OrtGraph to an OrtModel.
  /// This should be called once when creating a new model.
  /// The OrtModel takes ownership of the OrtGraph; do NOT call deinit on the graph.
  pub fn addGraph(self: *@This(), graph: *Graph) !void {
    try Error.check(Api.editor.underlying.AddGraphToModel.?(
      @ptrCast(self),
      @ptrCast(graph)
    ));
  }

  /// Release an OrtModel.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseModel.?(@ptrCast(self));
  }

  pub const Metadata = opaque {
    /// Get `producer name` from an ::OrtModelMetadata
    ///
    /// allocator: The allocator used to allocate the returned string.
    /// Returns a null terminated string allocated using `allocator`. 
    /// The caller must free the returned pointer using `allocator.free()`.
    pub fn getProducerName(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.ModelMetadataGetProducerName.?(
        @ptrCast(self),
        @ptrCast(allocator),
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Get `graph name` from an ::OrtModelMetadata
    ///
    /// allocator: The allocator used to allocate the returned string.
    /// Returns a null terminated string allocated using `allocator`. 
    /// The caller must free the returned pointer using `allocator.free()`.
    pub fn getGraphName(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.ModelMetadataGetGraphName.?(
        @ptrCast(self),
        @ptrCast(allocator),
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Get `domain` from an ::OrtModelMetadata
    ///
    /// allocator: The allocator used to allocate the returned string.
    /// Returns a null terminated string allocated using `allocator`. 
    /// The caller must free the returned pointer using `allocator.free()`.
    pub fn getDomain(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.ModelMetadataGetDomain.?(
        @ptrCast(self),
        @ptrCast(allocator),
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Get `description` from an ::OrtModelMetadata
    ///
    /// allocator: The allocator used to allocate the returned string.
    /// Returns a null terminated string allocated using `allocator`. 
    /// The caller must free the returned pointer using `allocator.free()`.
    pub fn getDescription(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.ModelMetadataGetDescription.?(
        @ptrCast(self),
        @ptrCast(allocator),
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Get the description of the graph present in the model.
    ///
    /// allocator: The allocator used to allocate the returned string.
    /// Returns a null terminated string allocated using `allocator`. 
    /// The caller must free the returned pointer using `allocator.free()`.
    pub fn getGraphDescription(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.ModelMetadataGetGraphDescription.?(
        @ptrCast(self),
        @ptrCast(allocator),
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Return data for a key in the custom metadata map.
    ///
    /// key: Null terminated string key.
    /// allocator: The allocator used to allocate the returned string.
    /// Returns a null terminated string allocated using `allocator`, or null if key not found.
    /// The caller must free the returned pointer (if not null) using `allocator.free()`.
    pub fn lookupCustomMetadataMap(self: *const @This(), allocator: *Allocator, key: [*:0]const u8) !?[*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.ModelMetadataLookupCustomMetadataMap.?(
        @ptrCast(self),
        @ptrCast(allocator),
        key,
        @ptrCast(&out)
      ));
      return out;
    }

    /// Get version number from an ::OrtModelMetadata
    pub fn getVersion(self: *const @This()) !i64 {
      var out: i64 = 0;
      try Error.check(Api.ort.ModelMetadataGetVersion.?(
        @ptrCast(self),
        &out
      ));
      return out;
    }

    /// Get keys from the custom metadata map.
    ///
    /// allocator: The allocator used to allocate the returned array and strings.
    /// Returns a slice of null-terminated strings.
    /// Note: The returned keys slice pointer AND the individual string pointers 
    /// must be freed using `allocator`.
    pub fn getCustomMetadataMapKeys(self: *const @This(), allocator: *Allocator) ![][*:0]u8 {
      var keys_ptr: ?[*][*:0]u8 = null;
      var num_keys: i64 = 0;
      
      try Error.check(Api.ort.ModelMetadataGetCustomMetadataMapKeys.?(
        @ptrCast(self),
        @ptrCast(allocator),
        @ptrCast(&keys_ptr),
        &num_keys
      ));

      if (keys_ptr) |ptr| {
        return ptr[0..@intCast(num_keys)];
      }
      return &.{};
    }

    /// Release an ::OrtModelMetadata.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseModelMetadata.?(@ptrCast(self));
    }
  };

  pub const CompilationOptions = opaque {
    /// Creates an OrtModelCompilationOptions object from an existing OrtSessionOptions object.
    ///
    /// env: The OrtEnv.
    /// session_options: The Session.Options to use as a base.
    ///
    /// Note: Requires Compile API initialization.
    pub fn init(session_options: *const Session.Options) !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.compiler.underlying.CreateModelCompilationOptionsFromSessionOptions.?(
        Api.env.underlying,
        @ptrCast(session_options), // Assuming Session.Options is opaque
        @ptrCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Sets the file path to the input ONNX model to compile.
    pub fn setInputModelPath(self: *@This(), path: Utils.Path) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetInputModelPath.?(
        @ptrCast(self),
        path
      ));
    }

    /// Sets the buffer that stores the bytes of the loaded ONNX model to compile.
    pub fn setInputModelFromBuffer(self: *@This(), buffer: []const u8) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetInputModelFromBuffer.?(
        @ptrCast(self),
        @ptrCast(buffer.ptr),
        buffer.len
      ));
    }

    /// Sets the file path for the output ONNX model.
    pub fn setOutputModelPath(self: *@This(), path: Utils.Path) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelPath.?(
        @ptrCast(self),
        path
      ));
    }

    /// Configures model compilation to store the output compiled ONNX model in a buffer.
    ///
    /// allocator: The allocator used to allocate the buffer.
    /// Returns a slice to the allocated buffer. The memory is owned by the allocator/caller 
    /// context but specifically allocated here.
    pub fn setOutputModelBuffer(self: *@This(), allocator: *Allocator) ![]u8 {
      var ptr: ?*anyopaque = null;
      var len: usize = 0;
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelBuffer.?(
        @ptrCast(self),
        @ptrCast(allocator),
        @ptrCast(&ptr),
        &len
      ));
      return @as([*]u8, @ptrCast(ptr orelse return &.{}))[0..len];
    }

    /// Enables or disables the embedding of EPContext binary data.
    pub fn setEpContextEmbedMode(self: *@This(), embed: bool) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetEpContextEmbedMode.?(
        @ptrCast(self),
        embed
      ));
    }

    /// Sets flags representing one or more boolean options to enable.
    pub fn setFlags(self: *@This(), flags: Api.compiler.Flags) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetFlags.?(
        @ptrCast(self),
        @bitCast(flags)
      ));
    }

    /// Set the graph optimization level.
    pub fn setGraphOptimizationLevel(self: *@This(), level: Session.GraphOptimizationLevel) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetGraphOptimizationLevel.?(
        @ptrCast(self),
        @intFromEnum(level)
      ));
    }

    /// Compiles an input ONNX model with the given compilation options.
    /// Note: The input/output paths must have been set on the options object.
    pub fn compile(self: *const @This()) !void {
      try Error.check(Api.compiler.underlying.CompileModel.?(
        Api.env.underlying,
        @ptrCast(self)
      ));
    }

    /// Sets information related to EP context binary file.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetEpContextBinaryInformation
    pub fn setEpContextBinaryInformation(self: *@This(), output_dir: Utils.Path, model_name: Utils.Path) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetEpContextBinaryInformation.?(
          @ptrCast(self),
          output_dir,
          model_name,
      ));
    }

    /// Optionally sets the file that should store external initializers for the compiled model.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetOutputModelExternalInitializersFile
    pub fn setOutputModelExternalInitializersFile(self: *@This(), path: Utils.Path, threshold: usize) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelExternalInitializersFile.?(
          @ptrCast(self),
          path,
          threshold,
      ));
    }

    /// Sets a custom function called by ORT to write out the output model's bytes.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetOutputModelWriteFunc
    pub fn setOutputModelWriteFunc(self: *@This(), interface: WriteInterface) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelWriteFunc.?(
          @ptrCast(self),
          interface.write_fn,
          interface.ptr,
      ));
    }

    /// Sets a custom function to specify whether initializers should be stored within the model or externally.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc
    pub fn setOutputModelGetInitializerLocationFunc(self: *@This(), interface: LocationInterface) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc.?(
          @ptrCast(self),
          interface.loc_fn,
          interface.ptr,
      ));
    }

    /// Helper for setOutputModelWriteFunc. 
    /// The context instance must have a `write(self: *Self, buffer: []const u8) !void` method.
    pub const WriteInterface = struct {
      ptr: ?*anyopaque,
      write_fn: Api.c.OrtWriteBufferFunc,

      pub fn fromContext(instance: anytype) @This() {
        const T = @TypeOf(instance);
        const Ptr = if (@typeInfo(T) == .pointer) T else *T;
        const Sub = @typeInfo(Ptr).pointer.child;

        return .{
          .ptr = if (@bitSizeOf(Sub) == 0) null else @ptrCast(instance),
          .write_fn = struct {
            fn wrapper(ctx: ?*anyopaque, buffer: ?*const anyopaque, len: usize) callconv(.c) ?*Api.c.OrtStatus {
              const self: Ptr = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @ptrCast(@alignCast(ctx.?));
              const data: []const u8 = @as([*]const u8, @ptrCast(buffer.?))[0..len];

              if (self.write(data)) |_| {
                return null; // Success
              } else |err| {
                const msg = std.fmt.allocPrintSentinel(std.heap.c_allocator, "Zig Write Error: {s}", .{@errorName(err)}, 0) catch "Zig Write Error";
                return @ptrCast(Error.Status.init(@intFromEnum(Error.Code.Fail), msg.ptr));
              }
            }
          }.wrapper,
        };
      }
    };

    /// Helper for setOutputModelGetInitializerLocationFunc.
    /// The context must have a `getLocation(self: *Self, name: [*:0]const u8, value: *const Value, info: ?*const ExternalInitializerInfo) !?*ExternalInitializerInfo` method.
    pub const LocationInterface = struct {
      ptr: ?*anyopaque,
      loc_fn: Api.c.OrtGetInitializerLocationFunc,

      pub fn fromContext(instance: anytype) @This() {
        const T = @TypeOf(instance);
        const Ptr = if (@typeInfo(T) == .pointer) T else *T;
        const Sub = @typeInfo(Ptr).pointer.child;

        return .{
          .ptr = if (@bitSizeOf(Sub) == 0) null else @ptrCast(instance),
          .loc_fn = struct {
            fn wrapper(
              ctx: ?*anyopaque,
              name: ?[*:0]const u8,
              val: ?*const Api.c.OrtValue,
              info: ?*const Api.c.ExternalInitializerInfo,
              out: ?*?*Api.c.ExternalInitializerInfo,
            ) callconv(.c) ?*Api.c.OrtStatus {
              const self: Ptr = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @ptrCast(@alignCast(ctx.?));

              // Wrap call to Zig logic
              if (self.getLocation(name.?, @ptrCast(val.?), @ptrCast(info))) |maybe_new_info| {
                out.?.* = @ptrCast(maybe_new_info);
                return null;
              } else |err| {
                return Error.Status.init(@intFromEnum(Error.Code.Fail), @errorName(err));
              }
            }
          }.wrapper,
        };
      }
    };

    /// Release the options object.
    pub fn deinit(self: *@This()) void {
      Api.compiler.underlying.ReleaseModelCompilationOptions.?(@ptrCast(self));
    }
  };
};

pub const OpAttr = opaque {
  /// Defines the type of data stored in an Operator Attribute.
  pub const Type = enum(c_uint) {
    UNDEFINED = @bitCast(Api.c.ORT_OP_ATTR_UNDEFINED),
    INT = @bitCast(Api.c.ORT_OP_ATTR_INT),
    INTS = @bitCast(Api.c.ORT_OP_ATTR_INTS),
    FLOAT = @bitCast(Api.c.ORT_OP_ATTR_FLOAT),
    FLOATS = @bitCast(Api.c.ORT_OP_ATTR_FLOATS),
    STRING = @bitCast(Api.c.ORT_OP_ATTR_STRING),
    STRINGS = @bitCast(Api.c.ORT_OP_ATTR_STRINGS),
    GRAPH = @bitCast(Api.c.ORT_OP_ATTR_GRAPH),
    TENSOR = @bitCast(Api.c.ORT_OP_ATTR_TENSOR),
  };

  /// Create an attribute of an onnxruntime operator.
  ///
  /// name: Name of the attribute.
  /// data: Pointer to data.
  /// len: Number of bytes (for string) or number of elements (for arrays). 1 for scalars.
  /// type: The type of attribute.
  pub fn init(name: [*:0]const u8, data: *const anyopaque, len: c_int, type_: Type) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateOpAttr.?(
      name, 
      data, 
      len, 
      @intFromEnum(type_), 
      @ptrCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  // Helpers for common types
  
  pub fn initInt(name: [*:0]const u8, val: i64) !*@This() {
    return init(name, &val, 1, .INT);
  }

  pub fn initFloat(name: [*:0]const u8, val: f32) !*@This() {
    return init(name, &val, 1, .FLOAT);
  }

  pub fn initString(name: [*:0]const u8, val: []const u8) !*@This() {
    return init(name, val.ptr, @intCast(val.len), .STRING);
  }

  pub fn initInts(name: [*:0]const u8, val: []const i64) !*@This() {
    return init(name, val.ptr, @intCast(val.len), .INTS);
  }

  pub fn initFloats(name: [*:0]const u8, val: []const f32) !*@This() {
    return init(name, val.ptr, @intCast(val.len), .FLOATS);
  }

  /// Get the attribute name.
  pub fn getName(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.OpAttr_GetName.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse "";
  }

  /// Get the attribute type.
  pub fn getType(self: *const @This()) !Type {
    var out: Type = undefined;
    try Error.check(Api.ort.OpAttr_GetType.?(@ptrCast(self), @ptrCast(&out)));
    return out;
  }

  /// Get the 'TENSOR' attribute as an OrtValue.
  /// Returns a new Value that must be deinitialized.
  pub fn getTensor(self: *const @This()) !*Value {
    var out: ?*Value = null;
    try Error.check(Api.ort.OpAttr_GetTensorAttributeAsOrtValue.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Read contents of an attribute to data.
  /// out_data: Buffer to write data to.
  /// Returns the number of bytes written or required.
  pub fn read(self: *const @This(), type_: Type, out_data: []u8) !usize {
    var size: usize = 0;
    try Error.check(Api.ort.ReadOpAttr.?(
        @ptrCast(self), 
        @intFromEnum(type_), 
        @ptrCast(out_data.ptr), 
        out_data.len, 
        &size
    ));
    return size;
  }

  /// Release the attribute.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseOpAttr.?(@ptrCast(self));
  }
};

pub const Session = struct {
  /// Graph optimization level
  /// Refer to https://www.onnxruntime.ai/docs/performance/graph-optimizations.html#graph-optimization-levels for an in-depth understanding of the Graph Optimization Levels.
  pub const GraphOptimizationLevel = enum(c_uint) {
    NONE = @bitCast(Api.c.ORT_DISABLE_ALL),
    BASIC = @bitCast(Api.c.ORT_ENABLE_BASIC),
    EXTENDED = @bitCast(Api.c.ORT_ENABLE_EXTENDED),
    LAYOUT = @bitCast(Api.c.ORT_ENABLE_LAYOUT),
    ALL = @bitCast(Api.c.ORT_ENABLE_ALL),
  };

  /// Dictates if operators are executed one after another or in parallel where possible.
  pub const ExecutionMode = enum(c_uint) {
    SEQUENTIAL = @bitCast(Api.c.ORT_SEQUENTIAL),
    PARALLEL = @bitCast(Api.c.ORT_PARALLEL),
  };

  pub const Options = struct {
    /// Refer to https://www.onnxruntime.ai/docs/performance/graph-optimizations.html#graph-optimization-levels
    /// for an in-depth understanding of the Graph Optimization Levels.
    ///
    /// You may want to set this to BASIC if the model is not already optimized
    optimization_level: GraphOptimizationLevel = .NONE,
    /// The model will be saved to this path after the optimizations are applied to it.
    /// If this is null, the model will not be saved. This is the case by default.
    ///
    /// This is ignored if GraphOptimizationLevel os NONE
    optimized_model_output_path: ?[*:0]const u8 = null,
    /// Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model
    /// has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance.
    /// See [docs/ONNX_Runtime_Perf_Tuning.md] in the onnxruntime repo for more details.
    execution_mode: ExecutionMode = .SEQUENTIAL,
    /// weather profiling is enabled / disabled for a session. Setting this to a non-null value will enable profiling with this profiling name
    profiling: ?[:0]const u8 = null,
    /// The idea is if the input shapes are the same, we could trace the internal memory allocation
    /// and generate a memory pattern for future request. So next time we could just do one allocation
    /// with a big chunk for all the internal memory allocation.
    /// Memory pattern optimization is only available when Sequential Execution mode is enabled (see OrtApi::SetSessionExecutionMode)
    memory_pattern_optimization: bool = false,
    /// Enable the memory arena on CPU
    /// Arena may pre-allocate memory for future usage.
    memory_arena: bool = true,
    /// Sets the logid for this session
    /// Null means default log_id is used
    log_id: ?[*:0]const u8 = null,
    /// Set session log verbosity level
    /// Applies to session load, initialization, etc
    verbosity: ?Logging.Level = null,
    /// Set session log severity level
    severity: ?Logging.Level = null,

    /// Sets the number of threads used to parallelize the execution within nodes
    /// When running a single node operation, ex. add, this sets the maximum number of threads to use.
    ///
    /// Note: If built with OpenMP, this has no effect on the number of threads used. In this case
    ///       use the OpenMP env variables to configure the number of intra op num threads.
    intraop_threads: c_int = 0, // 0 means default value

    /// Sets the number of threads used to parallelize the execution of the graph
    /// if nodes can be run in parallel, this sets the maximum number of threads to use to run them in parallel.
    ///
    /// Note: if sequential execution is enabled this value is ignored, it acts as if it was set to 1.
    interop_threads: c_int = 0, // 0 means default value

    pub fn c(self: @This()) !*C {
      try self.validate();

      const retval = try C.init();
      if (self.optimization_level != .NONE) {
        try retval.setOptimizationLevel(self.optimization_level);
        if (self.optimized_model_output_path) |p| try retval.setOptimizedModelPath(p);
      }
      try retval.setExecutionMode(self.execution_mode);
      try retval.setProfiling(self.profiling);
      try retval.setMemoryPatternOptimization(self.memory_pattern_optimization);
      try retval.setCpuMemoryArena(self.memory_arena);
      if (self.log_id) |id| try retval.setLogId(id);
      if (self.verbosity) |level| try retval.setLogVerbosity(level);
      if (self.severity) |level| try retval.setLogSeverity(level);
      if (self.intraop_threads != 0) try retval.setIntraOpThreads(self.intraop_threads);
      if (self.interop_threads != 0) try retval.setInterOpThreads(self.interop_threads);
      return retval;
    }

    pub fn validate(self: @This()) !void {
      if (self.execution_mode == .PARALLEL and self.memory_pattern_optimization) return error.InvalidOptions;
    }

    pub const C = opaque {
      pub fn init() !*@This() {
        var self: *@This() = undefined;
        try Error.check(Api.ort.CreateSessionOptions.?(@ptrCast(&self)));
        return self;
      }

      pub fn addFreeDimensionOverride(self: *@This(), denotation: [*:0]const u8, dim: i64) !void {
        try Error.check(Api.ort.AddFreeDimensionOverride.?(@ptrCast(self), denotation, dim));
      }

      /// Override symbolic dimensions by name.
      pub fn addFreeDimensionOverrideByName(self: *@This(), name: [*:0]const u8, dim: i64) !void {
        try Error.check(Api.ort.AddFreeDimensionOverrideByName.?(@ptrCast(self), name, dim));
      }

      pub fn clone(self: *const @This()) !*@This() {
        var retval: ?*@This() = null;
        try Error.check(Api.ort.CloneSessionOptions.?(@ptrCast(self), @ptrCast(&retval)));
        return retval orelse error.OutOfMemory;
      }

      pub fn deinit(self: *@This()) void {
        Api.ort.ReleaseSessionOptions.?(@ptrCast(self));
      }

      /// Enable Custom Operators (from onnxruntime-extensions).
      pub fn enableOrtCustomOps(self: *@This()) !void {
        try Error.check(Api.ort.EnableOrtCustomOps.?(@ptrCast(self)));
      }

      /// Replace initialized Tensors with external data.
      pub fn addExternalInitializers(self: *@This(), names: []const [*:0]const u8, values: []const *const Value) !void {
        std.debug.assert(names.len == values.len);
        try Error.check(Api.ort.AddExternalInitializers.?(
            @ptrCast(self),
            @ptrCast(names.ptr),
            @ptrCast(values.ptr),
            names.len
        ));
      }

      /// Checks if the given session configuration entry exists.
      pub fn hasConfigEntry(self: *const @This(), key: [*:0]const u8) !bool {
        var out: c_int = 0;
        try Error.check(Api.ort.HasSessionConfigEntry.?(@ptrCast(self), key, &out));
        return out != 0;
      }

      pub fn _getConfigEntry(self: *const @This(), key: [*:0]const u8, out_ptr: [*:0]u8, out_len: *usize) !void {
        try Error.check(Api.ort.GetSessionConfigEntry.?(@ptrCast(self), @ptrCast(key), @ptrCast(out_ptr), out_len));
      }

      /// Get a session configuration value.
      /// allocator: Used to allocate the returned string.
      pub fn getConfigEntry(self: *const @This(), key: [*:0]const u8, gpa: std.mem.Allocator) ![:0]u8 {
        var size: usize = 0;
        // First call to get size
        self._getConfigEntry(key, @ptrCast(@constCast(&[_]u8{})), &size) catch {};
        const buf = try gpa.alloc(u8, size);
        errdefer gpa.free(buf);

        try self._getConfigEntry(key, @ptrCast(buf.ptr), &size);
        std.debug.assert(buf.len == size);
        return @as([*:0]u8, @ptrCast(buf.ptr))[0 .. size - 1: 0];
      }

      /// Register custom ops using a registration function name.
      /// The library must be linked or loaded.
      pub fn registerCustomOpsUsingFunction(self: *@This(), function_name: [*:0]const u8) !void {
        try Error.check(Api.ort.RegisterCustomOpsUsingFunction.?(@ptrCast(self), function_name));
      }

      /// Disable per-session thread pools (use global env thread pools).
      pub fn disablePerSessionThreads(self: *@This()) !void {
        try Error.check(Api.ort.DisablePerSessionThreads.?(@ptrCast(self)));
      }

      pub fn setOptimizationLevel(self: *@This(), level: GraphOptimizationLevel) !void {
        try Error.check(Api.ort.SetSessionGraphOptimizationLevel.?(@ptrCast(self), @intFromEnum(level)));
      }

      pub fn setOptimizedModelPath(self: *@This(), path: [*:0]const u8) !void {
        try Error.check(Api.ort.SetOptimizedModelFilePath.?(@ptrCast(self), path));
      }

      pub fn setExecutionMode(self: *@This(), mode: ExecutionMode) !void {
        try Error.check(Api.ort.SetSessionExecutionMode.?(@ptrCast(self), @intFromEnum(mode)));
      }

      pub fn setProfiling(self: *@This(), status: ?[:0]const u8) !void {
        if (status) |str| {
          try Error.check(Api.ort.EnableProfiling.?(@ptrCast(self), str.ptr));
        } else {
          try Error.check(Api.ort.DisableProfiling.?(@ptrCast(self)));
        }
      }

      pub fn setMemoryPatternOptimization(self: *@This(), enabled: bool) !void {
        if (enabled) {
          try Error.check(Api.ort.EnableMemPattern.?(@ptrCast(self)));
        } else {
          try Error.check(Api.ort.DisableMemPattern.?(@ptrCast(self)));
        }
      }

      pub fn setCpuMemoryArena(self: *@This(), enabled: bool) !void {
        if (enabled) {
          try Error.check(Api.ort.EnableCpuMemArena.?(@ptrCast(self)));
        } else {
          try Error.check(Api.ort.DisableCpuMemArena.?(@ptrCast(self)));
        }
      }

      /// Set whether to use deterministic compute.
      pub fn setDeterministicCompute(self: *@This(), value: bool) !void {
        try Error.check(Api.ort.SetDeterministicCompute.?(@ptrCast(self), value));
      }

      /// Set user logging function.
      pub fn setUserLoggingFunction(self: *@This(), logging_interface: Logging.Interface) !void {
        try Error.check(Api.ort.SetUserLoggingFunction.?(
            @ptrCast(self),
            logging_interface.log_fn,
            logging_interface.ptr
        ));
      }

      pub fn setLogId(self: *@This(), id: [*:0]const u8) !void {
        try Error.check(Api.ort.SetSessionLogId.?(@ptrCast(self), id));
      }

      pub fn setLogVerbosity(self: *@This(), level: Logging.Level) !void {
        try Error.check(Api.ort.SetSessionLogVerbosityLevel.?(@ptrCast(self), @bitCast(@intFromEnum(level))));
      }

      pub fn setLogSeverity(self: *@This(), level: Logging.Level) !void {
        try Error.check(Api.ort.SetSessionLogSeverityLevel.?(@ptrCast(self), @bitCast(@intFromEnum(level))));
      }

      pub fn setIntraOpThreads(self: *@This(), threads: c_int) !void {
        try Error.check(Api.ort.SetIntraOpNumThreads.?(@ptrCast(self), threads));
      }

      pub fn setInterOpThreads(self: *@This(), threads: c_int) !void {
        try Error.check(Api.ort.SetInterOpNumThreads.?(@ptrCast(self), threads));
      }

      pub fn addCustomOpDomain(self: *@This(), domain: *Op.Custom.Domain) !void {
        try Error.check(Api.ort.AddCustomOpDomain.?(@ptrCast(self), @ptrCast(domain)));
      }

      /// Wraps OrtApi::AddSessionConfigEntry
      pub fn addConfigEntry(self: *@This(), key: [*:0]const u8, value: [*:0]const u8) !void {
        try Error.check(Api.ort.AddSessionConfigEntry.?(@ptrCast(self), key, value));
      }

      /// Wraps OrtApi::AddInitializer
      /// Note: The lifetime of the OrtValue and the underlying buffer must outlive the session object
      pub fn addInitializer(self: *@This(), name: [*:0]const u8, val: *const Value) !void {
        try Error.check(Api.ort.AddInitializer.?(@ptrCast(self), name, @ptrCast(val)));
      }

      /// Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA_V2
      pub fn appendExecutionProviderCUDA(self: *@This(), options: *const ProviderOptions.CUDA) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_CUDA_V2.?(@ptrCast(self), @ptrCast(options)));
      }

      /// Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT_V2
      pub fn appendExecutionProviderTensorRT(self: *@This(), options: *const ProviderOptions.TensorRT) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_TensorRT_V2.?(@ptrCast(self), @ptrCast(options)));
      }

      /// Append Execution Providers (Generic)
      /// Options can be a struct or tuple with [:0]const u8 or [*:0]const u8 values
      pub fn appendExecutionProvider(self: *@This(), name: [:0]const u8, options: anytype) !void {
        const converted = Utils.createOptionsKVL(options, .cstr);
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider.?(@ptrCast(self), name.ptr, converted.keys(), converted.vals(), converted.len));
      }

      /// Loads a shared library (.dll on windows, .so on linux, etc) named 'library_name' and looks for this entry point:
      ///     OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
      /// It then passes in the provided session options to this function along with the api base.
      ///
      /// The handle to the loaded library is automatically released by ORT when the last OrtSession that references the
      /// library handle is released. If no OrtSession is created, then the library handle is released when the provided
      /// OrtSessionOptions is released.
      pub fn registerCustomOpsLibrary(self: *@This(), path: Utils.Path) !void {
        try Error.check(Api.ort.RegisterCustomOpsLibrary_V2.?(@ptrCast(self), path));
      }

      /// Deprecated: Use registerCustomOpsLibrary
      /// Registers custom ops from a shared library and returns the library handle.
      ///
      /// Loads a shared library (dll on windows, so on linux, etc) named 'library_path' and looks for this entry point:
      ///     OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
      ///
      /// It then passes in the provided session options to this function along with the api base.
      /// The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
      /// session options are destroyed, or if an error occurs and it is non null.
      pub fn registerCustomOpsLibrary_DeprecatedV1(self: *@This(), path: Utils.Path) !*anyopaque {
        var retval: *anyopaque = undefined;
        try Error.check(Api.ort.RegisterCustomOpsLibrary.?(@ptrCast(self), path, @ptrCast(&retval)));
        return retval;
      }

      /// Replace initialized Tensors with external data from files in memory.
      ///
      /// allocator: Used to allocate the temporary C-compatible arrays required by the API.
      /// names: Slice of filenames.
      /// buffers: Slice of file contents.
      pub fn addExternalInitializersFromFilesInMemory(
        self: *@This(),
        names: []const Utils.Path,
        buffers: []const [*]const u8,
        lengths: []const [*]u8,
      ) !void {
        std.debug.assert(names.len == buffers.len);
        std.debug.assert(buffers.len == lengths.len);

        try Error.check(Api.ort.AddExternalInitializersFromFilesInMemory.?(
            @ptrCast(self),
            @ptrCast(names.ptr),
            @ptrCast(buffers.ptr),
            @ptrCast(lengths.ptr),
            names.len
        ));
      }

      pub const ExecutionProviderDevicePolicy = enum(c_uint) {
        Default = @bitCast(Api.c.OrtExecutionProviderDevicePolicy_DEFAULT),
        PreferCPU = @bitCast(Api.c.OrtExecutionProviderDevicePolicy_PREFER_CPU),
        PreferNPU = @bitCast(Api.c.OrtExecutionProviderDevicePolicy_PREFER_NPU),
        PreferGPU = @bitCast(Api.c.OrtExecutionProviderDevicePolicy_PREFER_GPU),
        MaxPerformance = @bitCast(Api.c.OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE),
        MaxEfficiency = @bitCast(Api.c.OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY),
        MinOverallPower = @bitCast(Api.c.OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER),
      };

      /// Set the execution provider selection policy for the session.
      pub fn setEpSelectionPolicy(self: *@This(), policy: ExecutionProviderDevicePolicy) !void {
        try Error.check(Api.ort.SessionOptionsSetEpSelectionPolicy.?(@ptrCast(self), @intFromEnum(policy)));
      }

      /// Set the execution provider selection policy delegate for the session.
      pub fn setEpSelectionPolicyDelegate(self: *@This(), delegate: Api.c.EpSelectionDelegate, state: ?*anyopaque) !void {
        try Error.check(Api.ort.SessionOptionsSetEpSelectionPolicyDelegate.?(@ptrCast(self), delegate, state));
      }

      /// Set custom thread creation function.
      pub fn setCustomCreateThreadFn(self: *@This(), create_fn: Api.c.OrtCustomCreateThreadFn) !void {
        try Error.check(Api.ort.SessionOptionsSetCustomCreateThreadFn.?(@ptrCast(self), create_fn));
      }

      /// Set creation options for custom thread.
      pub fn setCustomThreadCreationOptions(self: *@This(), options_ptr: ?*anyopaque) !void {
        try Error.check(Api.ort.SessionOptionsSetCustomThreadCreationOptions.?(@ptrCast(self), options_ptr));
      }

      /// Set custom thread join function.
      pub fn setCustomJoinThreadFn(self: *@This(), join_fn: Api.c.OrtCustomJoinThreadFn) !void {
        try Error.check(Api.ort.SessionOptionsSetCustomJoinThreadFn.?(@ptrCast(self), join_fn));
      }

      /// Sets load cancellation flag to abort session loading process.
      pub fn setLoadCancellationFlag(self: *@This(), cancel: bool) !void {
        try Error.check(Api.ort.SessionOptionsSetLoadCancellationFlag.?(@ptrCast(self), cancel));
      }

      /// Get Session configuration entries.
      /// Returns a new KeyValuePairs instance that must be deinitialized.
      pub fn getConfigEntries(self: *const @This()) !*KeyValuePairs {
        var out: ?*KeyValuePairs = null;
        try Error.check(Api.ort.GetSessionOptionsConfigEntries.?(@ptrCast(self), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Append the execution provider that is responsible for the selected OrtEpDevice instances.
      pub fn appendExecutionProviderV2(
        self: *@This(), 
        ep_devices: []const Ep.Device, 
        options: anytype
      ) !void {
        const converted = Utils.createOptionsKVL(options, .cstr);

        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_V2.?(
            @ptrCast(self),
            Api.env.underlying,
            @ptrCast(ep_devices.ptr),
            ep_devices.len,
            converted.keys(),
            converted.vals(),
            converted.len,
        ));
      }

      /// Append OpenVINO execution provider (V2 - String Options).
      pub fn appendExecutionProviderOpenVINO(self: *@This(), options: ?*const KeyValuePairs) !void {
        const keys: ?[*]const [*:0]const u8 = if (options) |o| o.getKeyValues()[0].ptr else null;
        const values: ?[*]const [*:0]const u8 = if (options) |o| o.getKeyValues()[1].ptr else null;
        const count = if (options) |o| o.getKeyValues()[0].len else 0;

        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_OpenVINO_V2.?(
            @ptrCast(self), keys, values, count
        ));
      }

      /// Append VitisAI provider.
      pub fn appendExecutionProviderVitisAI(self: *@This(), options: ?*const KeyValuePairs) !void {
        const keys: ?[*]const [*:0]const u8 = if (options) |o| o.getKeyValues()[0].ptr else null;
        const values: ?[*]const [*:0]const u8 = if (options) |o| o.getKeyValues()[1].ptr else null;
        const count = if (options) |o| o.getKeyValues()[0].len else 0;

        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_VitisAI.?(
            @ptrCast(self), keys, values, count
        ));
      }

      /// Append DNNL provider.
      pub fn appendExecutionProviderDnnl(self: *@This(), options: *const Api.c.OrtDnnlProviderOptions) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_Dnnl.?(@ptrCast(self), options));
      }

      /// Append CANN provider.
      pub fn appendExecutionProviderCANN(self: *@This(), options: *const Api.c.OrtCANNProviderOptions) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_CANN.?(@ptrCast(self), options));
      }

      /// Legacy: Append CUDA provider.
      pub fn appendExecutionProviderCUDALegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_CUDA.?(@ptrCast(self), device_id));
      }

      /// Legacy: Append ROCM provider.
      pub fn appendExecutionProviderROCMLegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_ROCM.?(@ptrCast(self), device_id));
      }

      /// Legacy: Append TensorRT provider.
      pub fn appendExecutionProviderTensorRTLegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_TensorRT.?(@ptrCast(self), device_id));
      }

      /// Legacy: Append MIGraphX provider.
      pub fn appendExecutionProviderMIGraphXLegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_MIGraphX.?(@ptrCast(self), device_id));
      }

      /// Legacy: Append OpenVINO provider (Struct Options).
      pub fn appendExecutionProviderOpenVINOLegacy(self: *@This(), options: *const Api.c.OrtOpenVINOProviderOptions) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_OpenVINO.?(@ptrCast(self), options));
      }
    };
  };

  pub fn initZ(path: Utils.Path, options: *const Options.C) !*@This() {
    var self: ?*@This() = null;
    try Error.check(Api.ort.CreateSession.?(
      Api.env.underlying,
      path,
      @ptrCast(options),
      @ptrCast(&self),
    ));
    return self orelse error.OutOfMemory;
  }

  pub fn initSlice(data: []const u8, options: *const Options.C) !*@This() {
    var self: ?*@This() = null;
    try Error.check(Api.ort.CreateSessionFromArray.?(
      Api.env.underlying,
      @ptrCast(data.ptr),
      data.len,
      @ptrCast(options),
      @ptrCast(&self),
    ));
    return self orelse error.OutOfMemory;
  }

  pub fn initWithPrepackedWeights(
    model_path: Utils.Path,
    options: *const Options.C,
    container: *PrepackedWeightsContainer
  ) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateSessionWithPrepackedWeightsContainer.?(
        Api.env.underlying,
        model_path,
        @ptrCast(options),
        @ptrCast(container),
        @ptrCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  /// Create session from memory with prepacked weights container.
  pub fn initFromArrayWithPrepackedWeights(
    model_data: []const u8,
    options: *const Options.C,
    container: *PrepackedWeightsContainer
  ) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateSessionFromArrayWithPrepackedWeightsContainer.?(
        Api.env.underlying,
        @ptrCast(model_data.ptr),
        model_data.len,
        @ptrCast(options),
        @ptrCast(container),
        @ptrCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  /// Set DynamicOptions for EPs.
  pub fn setEpDynamicOptions(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
    std.debug.assert(keys.len == values.len);
    try Error.check(Api.ort.SetEpDynamicOptions.?(
        @ptrCast(self),
        @ptrCast(keys.ptr),
        @ptrCast(values.ptr),
        keys.len
    ));
  }

  pub fn getInputCount(self: *const @This()) !usize {
    var retval: usize = undefined;
    try Error.check(Api.ort.SessionGetInputCount.?(@ptrCast(self), &retval));
    return retval;
  }

  pub fn getOutputCount(self: *const @This()) !usize {
    var retval: usize = undefined;
    try Error.check(Api.ort.SessionGetOutputCount.?(@ptrCast(self), &retval));
    return retval;
  }

  pub fn overridableInitializerCount(self: *const @This()) !usize {
    var retval: usize = undefined;
    try Error.check(Api.ort.SessionGetOverridableInitializerCount.?(@ptrCast(self), &retval));
    return retval;
  }

  pub fn inputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.SessionGetInputTypeInfo.?(@ptrCast(self), index, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn outputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.SessionGetOutputTypeInfo.?(@ptrCast(self), index, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn overridableInitializerTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.SessionGetOverridableInitializerTypeInfo.?(@ptrCast(self), index, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn inputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionGetInputName.?(@ptrCast(self), index, @ptrCast(allocator), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn outputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionGetOutputName.?(@ptrCast(self), index, @ptrCast(allocator), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn overridableInitializerName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionGetOverridableInitializerName.?(@ptrCast(self), index, @ptrCast(allocator), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Run the model returning results in an Ort allocated vector. Wraps OrtApi::Run
  /// 
  /// input_names: Array of null terminated strings.
  /// input_values: Array of Value objects.
  /// output_names: Array of null terminated strings.
  /// output_values: Pre-allocated array of ?*Value. If an entry is null, ORT allocates it.
  pub fn run(
    self: *@This(),
    run_options: ?*const RunOptions, 
    input_names: []const [*:0]const u8,
    input_values: []*const Value, 
    output_names: []const [*:0]const u8,
    output_values: []*Value,
  ) !void {
    std.debug.assert(input_names.len == input_values.len);
    std.debug.assert(output_names.len == output_values.len);
    try Error.check(Api.ort.Run.?(
      @ptrCast(self),
      @ptrCast(run_options),
      @ptrCast(input_names.ptr),
      @ptrCast(input_values.ptr),
      input_values.len,
      @ptrCast(output_names.ptr),
      output_values.len,
      @ptrCast(output_values.ptr),
    ));
  }

  /// Run the model asynchronously
  /// callback_ctx has a function `callback(ctx: *@TypeOf(callback_ctx), []*const Value, *Error.Status)`
  pub fn runAsync(
    self: *@This(),
    run_options: ?*const RunOptions,
    input_names: []const [*:0]const u8,
    inputs: []const *const Value,
    output_names: []const [*:0]const u8,
    outputs: []*Value,
    callback_ctx_ptr: anytype,
  ) !void {
    std.debug.assert(input_names.len == inputs.len);
    std.debug.assert(output_names.len == outputs.len);

    try Error.check(Api.ort.RunAsync.?(
        @ptrCast(self),
        @ptrCast(run_options),
        @ptrCast(input_names.ptr),
        @ptrCast(inputs.ptr),
        inputs.len,
        @ptrCast(output_names.ptr),
        output_names.len,
        @ptrCast(outputs.ptr),
        &struct {pub fn callback(ctx: ?*anyopaque, vptr: [*c]?*Api.c.OrtValue, vlen: usize, status: Api.c.OrtStatusPtr) callconv(.c) void {
          @as(@TypeOf(callback_ctx_ptr), @alignCast(@ptrCast(ctx))).callback(@as([*]Value, @ptrCast(vptr))[0 .. vlen], @as(*Error.Status, @ptrCast(status)));
        }}.callback,
        @ptrCast(callback_ctx_ptr),
    ));
  }

  /// Run the model using IoBinding. Wraps OrtApi::RunWithBinding
  pub fn runWithBinding(self: *@This(), run_options: ?*const RunOptions, binding: *const IoBinding) !void {
    try Error.check(Api.ort.RunWithBinding.?(@ptrCast(self), @ptrCast(run_options), @ptrCast(binding)));
  }

  /// End profiling and return a copy of the profiling file name.
  /// The returned string is allocated using the provided allocator and must be freed by the caller.
  pub fn endProfiling(self: *@This(), allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionEndProfiling.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get profiling start time in nanoseconds
  pub fn getProfilingStartTimeNs(self: *const @This()) !u64 {
    var out: u64 = 0;
    try Error.check(Api.ort.SessionGetProfilingStartTimeNs.?(@ptrCast(self), &out));
    return out;
  }

  /// Get model metadata
  pub fn getModelMetadata(self: *const @This()) !*Model.Metadata {
    var out: ?*Model.Metadata = null;
    try Error.check(Api.ort.SessionGetModelMetadata.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn getMemoryInfoForInputs(self: *const @This(), allocator: std.mem.Allocator) ![]*const Allocator.MemoryInfo {
    const count = try self.getInputCount();
    const out_ptr = try allocator.alloc(*const Allocator.MemoryInfo, count);
    errdefer allocator.free(out_ptr);
    try Error.check(Api.ort.SessionGetMemoryInfoForInputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  pub fn getMemoryInfoForOutputs(self: *const @This(), allocator: std.mem.Allocator) ![]*const Allocator.MemoryInfo {
    const count = try self.getOutputCount();
    const out_ptr = try allocator.alloc(*const Allocator.MemoryInfo, count);
    errdefer allocator.free(out_ptr);
    try Error.check(Api.ort.SessionGetMemoryInfoForOutputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  pub fn getEpDeviceForInputs(self: *const @This(), allocator: std.mem.Allocator) ![]?*const Ep.Device {
    const count = try self.getInputCount();
    const out_ptr = try allocator.alloc(?*const Ep.Device, count);
    errdefer allocator.free(out_ptr);
    try Error.check(Api.ort.SessionGetEpDeviceForInputs.?(@ptrCast(self), @ptrCast(out_ptr.ptr), count));
    return out_ptr;
  }

  /// Release the Session object.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseSession.?(@ptrCast(self));
  }
};

pub const RunOptions = opaque {
  pub fn init() !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateRunOptions.?(@ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn setLogVerbosityLevel(self: *@This(), level: c_int) !void {
    try Error.check(Api.ort.RunOptionsSetRunLogVerbosityLevel.?(@ptrCast(self), level));
  }

  pub fn setLogSeverityLevel(self: *@This(), level: Logging.Level) !void {
    try Error.check(Api.ort.RunOptionsSetRunLogSeverityLevel.?(@ptrCast(self), @bitCast(@intFromEnum(level))));
  }

  pub fn setRunTag(self: *@This(), tag: [*:0]const u8) !void {
    try Error.check(Api.ort.RunOptionsSetRunTag.?(@ptrCast(self), tag));
  }

  pub fn getLogVerbosityLevel(self: *const @This()) !c_int {
    var level: c_int = 0;
    try Error.check(Api.ort.RunOptionsGetRunLogVerbosityLevel.?(@ptrCast(self), &level));
    return level;
  }

  pub fn getLogSeverityLevel(self: *const @This()) !Logging.Level {
    var level: c_int = 0;
    try Error.check(Api.ort.RunOptionsGetRunLogSeverityLevel.?(@ptrCast(self), &level));
    return @enumFromInt(level);
  }

  pub fn getRunTag(self: *const @This()) !?[*:0]const u8 {
    var tag: ?[*:0]const u8 = null;
    try Error.check(Api.ort.RunOptionsGetRunTag.?(@ptrCast(self), @ptrCast(&tag)));
    return tag;
  }

  pub fn setTerminate(self: *@This()) !void {
    try Error.check(Api.ort.RunOptionsSetTerminate.?(@ptrCast(self)));
  }

  pub fn unsetTerminate(self: *@This()) !void {
    try Error.check(Api.ort.RunOptionsUnsetTerminate.?(@ptrCast(self)));
  }

  /// Wrapper around ::OrtLoraAdapter
  /// Holds a set of LoRA Parameters loaded from a single file.
  ///
  /// LoRA (Low-Rank Adaptation) allow you to modify the behavior of a base model without retraining it.
  /// This allows you to hot-swap fine-tuned weights during inference
  pub const LoraAdapter = opaque {
    /// Wraps OrtApi::CreateLoraAdapter
    /// adapter_path: Path to the LoRA adapter file.
    /// allocator: Optional allocator. If null, data stays on CPU until inference requires it on device.
    pub fn init(adapter_path: Utils.Path, allocator: *Allocator) !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateLoraAdapter.?(adapter_path, @ptrCast(allocator), @ptrCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::CreateLoraAdapterFromArray
    /// bytes: In-memory buffer of the LoRA adapter.
    pub fn initFromArray(bytes: []const u8, allocator: *Allocator) !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateLoraAdapterFromArray.?(bytes.ptr, bytes.len, @ptrCast(allocator), @ptrCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Release the LoraAdapter.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseLoraAdapter.?(@ptrCast(self));
    }
  };

  pub fn addActiveLoraAdapter(self: *@This(), adapter: *const LoraAdapter) !void {
    try Error.check(Api.ort.RunOptionsAddActiveLoraAdapter.?(@ptrCast(self), @ptrCast(adapter)));
  }

  pub fn addConfigEntry(self: *@This(), key: [*:0]const u8, value: [*:0]const u8) !void {
    try Error.check(Api.ort.AddRunConfigEntry.?(@ptrCast(self), key, value));
  }

  pub fn getConfigEntry(self: *const @This(), key: [*:0]const u8) ?[*:0]const u8 {
    return Api.ort.GetRunConfigEntry.?(@ptrCast(self), key);
  }

  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseRunOptions.?(@ptrCast(self));
  }
};

pub const IoBinding = opaque {
  pub fn init(session: *Session) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateIoBinding.?(@ptrCast(session), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn bindInput(self: *@This(), name: [*:0]const u8, value: *const Value) !void {
    try Error.check(Api.ort.BindInput.?(@ptrCast(self), name, @ptrCast(value)));
  }

  pub fn bindOutput(self: *@This(), name: [*:0]const u8, value: *const Value) !void {
    try Error.check(Api.ort.BindOutput.?(@ptrCast(self), name, @ptrCast(value)));
  }

  pub fn bindOutputToDevice(self: *@This(), name: [*:0]const u8, mem_info: *const Allocator.MemoryInfo) !void {
    try Error.check(Api.ort.BindOutputToDevice.?(@ptrCast(self), name, @ptrCast(mem_info)));
  }

  pub fn getBoundOutputNames(self: *const @This(), allocator: *Allocator) !struct { names: [*]u8, lengths: []usize } {
    var buffer: ?[*]u8 = null;
    var lengths_ptr: ?[*]usize = null;
    var count: usize = 0;
    try Error.check(Api.ort.GetBoundOutputNames.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&buffer), @ptrCast(&lengths_ptr), &count));
    if (buffer == null or lengths_ptr == null or count == 0) return error.Unbounded;
    return .{ .names = buffer.?, .lengths = lengths_ptr.?[0 .. count] };
  }

  // DO NOT FREE if length is 0
  pub fn getBoundOutputValues(self: *const @This(), allocator: *Allocator) ![]*Value {
    var output_ptr: ?[*]*Value = null;
    var count: usize = 0;
    try Error.check(Api.ort.GetBoundOutputValues.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&output_ptr), &count));
    return (output_ptr orelse return &.{})[0..count];
  }

  pub fn clearBoundInputs(self: *@This()) void {
    Api.ort.ClearBoundInputs.?(@ptrCast(self));
  }

  pub fn clearBoundOutputs(self: *@This()) void {
    Api.ort.ClearBoundOutputs.?(@ptrCast(self));
  }

  pub fn synchronizeInputs(self: *@This()) !void {
    try Error.check(Api.ort.SynchronizeBoundInputs.?(@ptrCast(self)));
  }

  pub fn synchronizeOutputs(self: *@This()) !void {
    try Error.check(Api.ort.SynchronizeBoundOutputs.?(@ptrCast(self)));
  }

  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseIoBinding.?(@ptrCast(self));
  }
};

pub const KernelContext = opaque {
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.KernelContext_GetInputCount.?(@ptrCast(self), &out));
    return out;
  }

  pub fn getOutputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.KernelContext_GetOutputCount.?(@ptrCast(self), &out));
    return out;
  }

  pub fn getInput(self: *const @This(), index: usize) !?*const Value {
    var out: ?*const Value = null;
    try Error.check(Api.ort.KernelContext_GetInput.?(@ptrCast(self), index, @ptrCast(&out)));
    return out;
  }

  pub fn getOutput(self: *@This(), index: usize, dims: []const i64) !*Value {
    var out: ?*Value = null;
    try Error.check(Api.ort.KernelContext_GetOutput.?(@ptrCast(self), index, dims.ptr, dims.len, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn getGpuComputeStream(self: *const @This()) !?*anyopaque {
    var out: ?*anyopaque = null;
    try Error.check(Api.ort.KernelContext_GetGPUComputeStream.?(@ptrCast(self), &out));
    return out;
  }
  
  pub fn getResource(self: *const @This(), version: c_int, id: c_int) !?*anyopaque {
    var out: ?*anyopaque = null;
    try Error.check(Api.ort.KernelContext_GetResource.?(@ptrCast(self), version, id, &out));
    return out;
  }
  
  pub fn getScratchBuffer(self: *const @This(), mem_info: *const Allocator.MemoryInfo, size: usize) !?*anyopaque {
    var out: ?*anyopaque = null;
    try Error.check(Api.ort.KernelContext_GetScratchBuffer.?(@ptrCast(self), @ptrCast(mem_info), size, &out));
    return out;
  }
  
  pub fn getAllocator(self: *const @This(), mem_info: *const Allocator.MemoryInfo) !*Allocator {
    var out: ?*Allocator = null;
    try Error.check(Api.ort.KernelContext_GetAllocator.?(@ptrCast(self), @ptrCast(mem_info), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get the runtime logger.
  pub fn getLogger(self: *const @This()) !*const KernelInfo.Logger {
    var out: ?*const KernelInfo.Logger = null;
    try Error.check(Api.ort.KernelContext_GetLogger.?(@ptrCast(self), @ptrCast(&out)));
    return out.?;
  }

  /// Run a function in parallel.
  pub fn parallelFor(
    self: *const @This(), 
    task: fn(*anyopaque, usize) callconv(.c) void, 
    total: usize, 
    num_batch: usize, 
    user_data: ?*anyopaque
  ) !void {
    try Error.check(Api.ort.KernelContext_ParallelFor.?(@ptrCast(self), task, total, num_batch, user_data));
  }
};

pub const KernelInfo = opaque {
  pub fn copy(self: *const @This()) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CopyKernelInfo.?(@ptrCast(self), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn _getNodeName(self: *const @This(), out_ptr: [*:0]u8, out_size: *usize) !void {
    try Error.check(Api.ort.KernelInfo_GetNodeName.?(@ptrCast(self), @ptrCast(out_ptr), out_size));
  }

  pub fn getNodeName(self: *const @This(), gpa: std.mem.Allocator) ![:0]u8 {
    var size: usize = 0;
    // First call to get size (pass null)
    self._getNodeName(@ptrCast(@constCast(&[_]u8{})), &size) catch {};
    std.debug.assert(size != 0);

    const buf = try gpa.alloc(u8, size);
    errdefer gpa.free(buf);

    try self._getNodeName(@ptrCast(buf.ptr), &size);
    std.debug.assert(buf.len == size);
    return @as([*:0]u8, @ptrCast(buf.ptr))[0 .. size - 1: 0];
  }
  
  pub fn getAttributeFloat(self: *const @This(), name: [*:0]const u8) !f32 {
    var out: f32 = 0;
    try Error.check(Api.ort.KernelInfoGetAttribute_float.?(@ptrCast(self), name, &out));
    return out;
  }
  
  pub fn getAttributeInt(self: *const @This(), name: [*:0]const u8) !i64 {
    var out: i64 = 0;
    try Error.check(Api.ort.KernelInfoGetAttribute_int64.?(@ptrCast(self), name, &out));
    return out;
  }
  
  pub fn _getAttributeString(self: *const @This(), name: [*:0]const u8, out_ptr: [*:0]u8, out_len: *usize) !void {
    try Error.check(Api.ort.KernelInfoGetAttribute_string.?(@ptrCast(self), name, @ptrCast(out_ptr), out_len));
  }

  pub fn getAttributeString(self: *const @This(), name: [*:0]const u8, gpa: std.mem.Allocator) ![:0]u8 {
    var size: usize = 0;
    // First call get size
    self._getAttributeString(name, @ptrCast(@constCast(&[_]u8{})), &size) catch {};

    const buf = try gpa.alloc(u8, size);
    errdefer gpa.free(buf);

    try self._getAttributeString(name, @ptrCast(buf.ptr), &size);
    std.debug.assert(buf.len == size);
    return @as([*:0]u8, @ptrCast(buf.ptr))[0 .. size - 1: 0];
  }
  
  pub fn getAttributeTensor(self: *const @This(), name: [*:0]const u8, allocator: *Allocator) !*Value {
    var out: ?*Value = null;
    try Error.check(Api.ort.KernelInfoGetAttribute_tensor.?(@ptrCast(self), name, @ptrCast(allocator), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get the number of inputs.
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.KernelInfo_GetInputCount.?(@ptrCast(self), &out));
    return out;
  }

  /// Get the number of outputs.
  pub fn getOutputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.KernelInfo_GetOutputCount.?(@ptrCast(self), &out));
    return out;
  }

  pub fn _getInputName(self: *const @This(), index: usize, out_ptr: [*:0]u8, out_len: *usize) !void {
    try Error.check(Api.ort.KernelInfo_GetInputName.?(@ptrCast(self), index, @ptrCast(out_ptr), out_len));
  }

  /// Get the name of an input.
  /// allocator: The allocator used to allocate the returned string buffer.
  /// Returns a null-terminated string.
  pub fn getInputName(self: *const @This(), index: usize, gpa: std.mem.Allocator) ![:0]u8 {
    var size: usize = 0;
    // First call to get the required size
    self._getInputName(index, @ptrCast(@constCast(&[_]u8{})), &size) catch {};

    const buf = try gpa.alloc(u8, size);
    errdefer gpa.free(buf);

    // Second call to fill the buffer
    try self._getInputName(index, @ptrCast(buf.ptr), &size);
    std.debug.assert(buf.len == size);
    return @as([*:0]u8, @ptrCast(buf.ptr))[0 .. size - 1: 0];
  }

  pub fn _getOutputName(self: *const @This(), index: usize, out_ptr: [*:0]u8, out_len: *usize) !void {
    try Error.check(Api.ort.KernelInfo_GetOutputName.?(@ptrCast(self), index, @ptrCast(out_ptr), out_len));
  }
  /// Get the name of an output.
  /// allocator: The allocator used to allocate the returned string buffer.
  /// Returns a null-terminated string.
  pub fn getOutputName(self: *const @This(), index: usize, gpa: std.mem.Allocator) ![:0]u8 {
    var size: usize = 0;
    // First call to get the required size
    self._getOutputName(index, @ptrCast(@constCast(&[_]u8{})), &size) catch {};
    const buf = try gpa.alloc(u8, size);
    errdefer gpa.free(buf);

    // Second call to fill the buffer
    try self._getOutputName(index, @ptrCast(buf.ptr), &size);
    std.debug.assert(buf.len == size);
    return @as([*:0]u8, @ptrCast(buf.ptr))[0 .. size - 1: 0];
  }

  /// Get the type information for an input.
  /// The returned TypeInfo must be deinitialized by the caller.
  pub fn getInputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.KernelInfo_GetInputTypeInfo.?(@ptrCast(self), index, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get the type information for an output.
  /// The returned TypeInfo must be deinitialized by the caller.
  pub fn getOutputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.KernelInfo_GetOutputTypeInfo.?(@ptrCast(self), index, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn _getAttributeArrayFloat(self: *const @This(), name: [*:0]const u8, out_ptr: [*]f32, out_len: *usize) !void {
    try Error.check(Api.ort.KernelInfoGetAttributeArray_float.?(@ptrCast(self), name, @ptrCast(out_ptr), out_len));
  }

  /// Fetch a float array stored as an attribute.
  /// allocator: The allocator used to create the slice to hold the result.
  pub fn getAttributeArrayFloat(self: *const @This(), name: [*:0]const u8, gpa: std.mem.Allocator) ![]f32 {
    var size: usize = 0;
    // First call to get element count
    self._getAttributeArrayFloat(name, @constCast(&[_]f32{}), &size) catch {};
    const buf = try gpa.alloc(f32, size);

    // Second call to fill data
    try self._getAttributeArrayFloat(name, buf.ptr, &size);
    std.debug.assert(buf.len == size);
    return buf;
  }

  pub fn _getAttributeArrayInt(self: *const @This(), name: [*:0]const u8, out_ptr: [*]i64, out_len: *usize) !void {
    try Error.check(Api.ort.KernelInfoGetAttributeArray_int64.?(@ptrCast(self), name, @ptrCast(out_ptr), out_len));
  }

  /// Fetch an int64 array stored as an attribute.
  /// allocator: The allocator used to create the slice to hold the result.
  pub fn getAttributeArrayInt(self: *const @This(), name: [*:0]const u8, gpa: std.mem.Allocator) ![]i64 {
    var size: usize = 0;
    // First call to get element count
    self._getAttributeArrayInt(name, @constCast(&[_]i64{}), &size) catch {};
    const buf = try gpa.alloc(i64, size);

    // Second call to fill data
    try self._getAttributeArrayInt(name, buf.ptr, &size);
    std.debug.assert(buf.len == size);
    return buf;
  }

  /// Get allocator from KernelInfo for a specific memory type. 
  /// Please use `Allocator.deinit` (C API ReleaseAllocator) to release the returned object.
  pub fn getAllocator(self: *const @This(), mem_type: Allocator.MemoryType) !*Allocator {
    var out: ?*Allocator = null;
    try Error.check(Api.ort.KernelInfoGetAllocator.?(@ptrCast(self), @intFromEnum(mem_type), @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub const Logger = opaque {
    /// Log a message.
    pub fn logMessage(
      self: *const @This(), 
      severity: Logging.Level, 
      message: [*:0]const u8, 
      file: Utils.Path, 
      line: c_int, 
      func: [*:0]const u8
    ) !void {
      try Error.check(Api.ort.Logger_LogMessage.?(
          @ptrCast(self), 
          @intFromEnum(severity), 
          message, 
          file, 
          line, 
          func
      ));
    }

    /// Get the logging severity level.
    pub fn getSeverityLevel(self: *const @This()) !Logging.Level {
      var out: Api.c.OrtLoggingLevel = undefined;
      try Error.check(Api.ort.Logger_GetLoggingSeverityLevel.?(@ptrCast(self), &out));
      return @enumFromInt(out);
    }
  };

  pub fn getLogger(self: *const @This()) !*const Logger {
    var out: ?*const Logger = null;
    try Error.check(Api.ort.KernelInfo_GetLogger.?(@ptrCast(self), @ptrCast(&out)));
    return out.?;
  }

  /// Get a constant input tensor.
  /// is_constant: Output bool indicating if it is constant.
  pub fn getConstantInputTensor(self: *const @This(), index: usize, is_constant: *bool) !?*const Value {
    var out: ?*const Value = null;
    var is_const_c: c_int = 0;
    try Error.check(Api.ort.KernelInfoGetConstantInput_tensor.?(@ptrCast(self), index, &is_const_c, @ptrCast(&out)));
    is_constant.* = (is_const_c != 0);
    return out;
  }

  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseKernelInfo.?(@ptrCast(self));
  }
};

/// Represents a physical hardware device (CPU, GPU, NPU) available on the system.
/// This is an opaque type managed by ORT.
pub const HardwareDevice = opaque {
  /// Maps to OrtHardwareDeviceType in the C API.
  pub const Type = enum(c_uint) {
    CPU = @bitCast(Api.c.OrtHardwareDeviceType_CPU),
    GPU = @bitCast(Api.c.OrtHardwareDeviceType_GPU),
    NPU = @bitCast(Api.c.OrtHardwareDeviceType_NPU),
  };

  /// Returns the type of hardware (CPU, GPU, or NPU).
  /// Wraps OrtApi::HardwareDevice_Type
  pub fn getType(self: *const @This()) Type {
    return @enumFromInt(Api.ort.HardwareDevice_Type.?(@ptrCast(self)));
  }

  /// Returns the hardware device's vendor identifier (e.g., PCI Vendor ID).
  /// Wraps OrtApi::HardwareDevice_VendorId
  pub fn getVendorId(self: *const @This()) u32 {
    return Api.ort.HardwareDevice_VendorId.?(@ptrCast(self));
  }

  /// Returns the hardware device's vendor name as a null-terminated string.
  /// Wraps OrtApi::HardwareDevice_Vendor
  pub fn getVendor(self: *const @This()) [*:0]const u8 {
    return @ptrCast(Api.ort.HardwareDevice_Vendor.?(@ptrCast(self)));
  }

  /// Returns the hardware device's unique identifier.
  /// Note: This identifies the specific hardware instance when combined with vendor id.
  /// Wraps OrtApi::HardwareDevice_DeviceId
  pub fn getDeviceId(self: *const @This()) u32 {
    return Api.ort.HardwareDevice_DeviceId.?(@ptrCast(self));
  }

  /// Returns an OrtKeyValuePairs instance containing additional metadata for the device.
  /// Note: ORT owns this instance; do NOT call deinit/ReleaseKeyValuePairs on it.
  /// Wraps OrtApi::HardwareDevice_Metadata
  pub fn getMetadata(self: *const @This()) *const KeyValuePairs {
    return @ptrCast(Api.ort.HardwareDevice_Metadata.?(@ptrCast(self)));
  }
};

pub const Op = opaque {
  /// Create onnxruntime native operator.
  pub fn init(
    info: *const KernelInfo,
    op_name: [*:0]const u8,
    domain: [*:0]const u8,
    version: c_int,
    type_constraint_names: [][*:0]const u8,
    type_constraint_values: []const Value.Sub.Tensor.ElementDataType,
    attrs: []const *const OpAttr,
    input_count: c_int,
    output_count: c_int
  ) !*@This() {
    std.debug.assert(type_constraint_names.len == type_constraint_values.len);
    var out: ?*@This() = null;
    
    // Convert Enum slice to C Int slice if necessary, but here we assume binary compatibility or cast
    // Since TensorElementDataType is enum(c_uint), it matches C.
    
    try Error.check(Api.ort.CreateOp.?(
      @ptrCast(info),
      op_name,
      domain,
      version,
      @ptrCast(type_constraint_names.ptr),
      @ptrCast(type_constraint_values.ptr),
      @intCast(type_constraint_names.len),
      @ptrCast(attrs.ptr),
      @intCast(attrs.len),
      input_count,
      output_count,
      @ptrCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  pub fn invoke(
    self: *const @This(),
    context: *const KernelContext,
    inputs: []const *const Value,
    outputs: []*Value
  ) !void {
    try Error.check(Api.ort.InvokeOp.?(
      @ptrCast(context),
      @ptrCast(self),
      @ptrCast(inputs.ptr),
      @intCast(inputs.len),
      @ptrCast(outputs.ptr),
      @intCast(outputs.len)
    ));
  }

  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseOp.?(@ptrCast(self));
  }

  pub const Custom = struct {
    underlying: Api.c.OrtCustomOp,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@FieldType(@This(), "underlying"))); }

    pub const InputOutputCharacteristic = enum(c_uint) {
      Required = @bitCast(Api.c.INPUT_OUTPUT_REQUIRED),
      Optional = @bitCast(Api.c.INPUT_OUTPUT_OPTIONAL),
      Variadic = @bitCast(Api.c.INPUT_OUTPUT_VARIADIC),
    };

    /// Initialize a CustomOp structure with vtables pointing to the provided OpType and KernelType.
    ///
    /// Requirements for OpType:
    /// - Must have a field `ort_op: Op.Custom` (used for @fieldParentPtr)
    /// - fn getName(self: *const OpType) [*:0]const u8
    /// - fn getExecutionProviderType(self: *const OpType) ?[*:0]const u8
    /// - fn getInputType(self: *const OpType, index: usize) Value.Sub.Tensor.ElementDataType
    /// - fn getInputTypeCount(self: *const OpType) usize
    /// - fn getOutputType(self: *const OpType, index: usize) Value.Sub.Tensor.ElementDataType
    /// - fn getOutputTypeCount(self: *const OpType) usize
    /// - fn createKernel(self: *const OpType, api: *const Api.c.OrtApi, info: *const KernelInfo) !*KernelType
    /// - (Optional) fn getInputCharacteristic(self: *const OpType, index: usize) InputOutputCharacteristic
    /// - (Optional) fn getOutputCharacteristic(self: *const OpType, index: usize) InputOutputCharacteristic
    /// - (Optional) fn getInputMemoryType(self: *const OpType, index: usize) Allocator.MemoryType
    ///
    /// Requirements for KernelType:
    /// - fn compute(self: *KernelType, context: *KernelContext) !void
    ///
    pub fn init(comptime OpType: type, comptime KernelType: type) @This() {
      const VTable = struct {
        fn getName(op: ?*const Api.c.OrtCustomOp) callconv(.c) [*:0]const u8 {
          const self: *Op.Custom = @ptrCast(op.?);
          return self.getName();
        }

        fn getExecutionProviderType(op: ?*const Api.c.OrtCustomOp) callconv(.c) ?[*:0]const u8 {
          const self: *Op.Custom = @ptrCast(op.?);
          return self.getExecutionProviderType();
        }

        fn getInputType(op: ?*const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.ONNXTensorElementDataType {
          const self: *Op.Custom = @ptrCast(op.?);
          return @intFromEnum(self.getInputType(index));
        }

        fn getInputTypeCount(op: ?*const Api.c.OrtCustomOp) callconv(.c) usize {
          const self: *Op.Custom = @ptrCast(op.?);
          return self.getInputTypeCount();
        }

        fn getOutputType(op: ?*const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.ONNXTensorElementDataType {
          const self: *Op.Custom = @ptrCast(op.?);
          return @intFromEnum(self.getOutputType(index));
        }

        fn getOutputTypeCount(op: ?*const Api.c.OrtCustomOp) callconv(.c) usize {
          const self: *Op.Custom = @ptrCast(op.?);
          return self.getOutputTypeCount();
        }

        fn createKernelV2(op: ?*const Api.c.OrtCustomOp, api: ?*const Api.c.OrtApi, info: ?*const Api.c.OrtKernelInfo, kernel: ?*?*anyopaque) callconv(.c) ?*Error.Status {
          const self: *Op.Custom = @ptrCast(op.?);
          // We forward the raw C pointers to the user implementation which should expect opaque wrappers
          // but we cast them here for convenience in the Zig wrapper logic if needed.
          const k_info: *const KernelInfo = @ptrCast(info.?);
          
          const result = self.createKernel(api.?, k_info) catch |err| {
            // Convert Zig error to OrtStatus
            // Note: Simplified error mapping. Ideally mapping specific errors to ORT codes.
            const msg = @errorName(err);
            return Error.Status.init(Api.c.ORT_FAIL, msg);
          };
          
          kernel.?.* = result;
          return null; // OK
        }

        fn kernelComputeV2(kernel: ?*anyopaque, context: ?*Api.c.OrtKernelContext) callconv(.c) ?*Error.Status {
          const self: *KernelType = @ptrCast(@alignCast(kernel.?));
          const k_ctx: *KernelContext = @ptrCast(context.?);
          
          self.compute(k_ctx) catch |err| {
            return Error.Status.init(Api.c.ORT_FAIL, @errorName(err));
          };
          return null;
        }

        fn kernelDestroy(kernel: ?*anyopaque) callconv(.c) void {
          const self: *KernelType = @ptrCast(@alignCast(kernel.?));
          // We assume the user allocated the kernel using an allocator.
          // Since `createKernel` returns a pointer, we need to know how it was allocated to free it.
          // For this generic wrapper, we assume `std.heap.c_allocator` was used or the user must manage memory manually
          // if they construct this differently. 
          // A common pattern is to use the allocator passed in init, but here we simply destroy.
          // If the user allocated with a different allocator, they might leak or crash if we enforce c_allocator.
          // Ideally, the User struct handles destruction via `deinit` or similar, but the C API 
          // provides this callback.
          // 
          // *Assumption*: The user allocates `KernelType` using `std.heap.c_allocator.create(KernelType)`.
          std.heap.c_allocator.destroy(self);
        }

        // Optional handlers (default implementations provided if missing in OpType)

        fn getInputCharacteristic(op: ?*const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.OrtCustomOpInputOutputCharacteristic {
          const self: *Op.Custom = @ptrCast(op.?);
          if (@hasDecl(OpType, "getInputCharacteristic")) {
            return @intFromEnum(self.getInputCharacteristic(index));
          }
          return Api.c.INPUT_OUTPUT_REQUIRED;
        }

        fn getOutputCharacteristic(op: ?*const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.OrtCustomOpInputOutputCharacteristic {
          const self: *Op.Custom = @ptrCast(op.?);
          if (@hasDecl(OpType, "getOutputCharacteristic")) {
            return @intFromEnum(self.getOutputCharacteristic(index));
          }
          return Api.c.INPUT_OUTPUT_REQUIRED;
        }

        fn getInputMemoryType(op: ?*const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.OrtMemType {
          const self: *Op.Custom = @ptrCast(op.?);
          if (@hasDecl(OpType, "getInputMemoryType")) {
            return @intFromEnum(self.getInputMemoryType(index));
          }
          return Api.c.OrtMemTypeDefault;
        }
      };

      return .{
        .underlying = .{
          .version = Api.c.ORT_API_VERSION,
          .CreateKernel = null, // Using V2
          .GetName = VTable.getName,
          .GetExecutionProviderType = VTable.getExecutionProviderType,
          .GetInputType = VTable.getInputType,
          .GetInputTypeCount = VTable.getInputTypeCount,
          .GetOutputType = VTable.getOutputType,
          .GetOutputTypeCount = VTable.getOutputTypeCount,
          .KernelCompute = null, // Using V2
          .KernelDestroy = VTable.kernelDestroy,
          .GetInputCharacteristic = VTable.getInputCharacteristic,
          .GetOutputCharacteristic = VTable.getOutputCharacteristic,
          .GetInputMemoryType = VTable.getInputMemoryType,
          .GetVariadicInputMinArity = null,
          .GetVariadicInputHomogeneity = null,
          .GetVariadicOutputMinArity = null,
          .GetVariadicOutputHomogeneity = null,
          .CreateKernelV2 = VTable.createKernelV2,
          .KernelComputeV2 = VTable.kernelComputeV2,
          .InferOutputShapeFn = null,
          .GetStartVersion = null,
          .GetEndVersion = null,
          .GetMayInplace = null,
          .ReleaseMayInplace = null,
          .GetAliasMap = null,
          .ReleaseAliasMap = null,
        },
      };
    }

    pub const Domain = opaque {
      /// Create a custom op domain.
      pub fn init(domain: [*:0]const u8) !*@This() {
        var out: ?*@This() = null;
        try Error.check(Api.ort.CreateCustomOpDomain.?(domain, @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Add a custom op to the domain.
      /// The Op.Custom struct must remain valid until the domain is released.
      pub fn add(self: *@This(), op: *Custom) !void {
        try Error.check(Api.ort.CustomOpDomain_Add.?(
          @ptrCast(self),
          @ptrCast(&op.underlying),
        ));
      }

      /// Release the domain.
      pub fn deinit(self: *@This()) void {
        Api.ort.ReleaseCustomOpDomain.?(@ptrCast(self));
      }
    };
  };
};

/// Information about an initializer stored in an external file (e.g., filepath, offset, size).
/// Wraps OrtExternalInitializerInfo.
pub const ExternalInitializerInfo = opaque {
  /// Creates an OrtExternalInitializerInfo instance.
  /// Wraps OrtApi::CreateExternalInitializerInfo
  pub fn init(filepath: Utils.Path, file_offset: i64, byte_size: usize) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateExternalInitializerInfo.?(
      filepath,
      file_offset,
      byte_size,
      @ptrCast(&out),
    ));
    return out orelse error.OutOfMemory;
  }

  /// Get the relative path to the file that stores the initializer's data.
  /// The path is relative to the filesystem directory where the ONNX model was stored.
  /// Note: Do NOT free this pointer. It is valid for the lifetime of this object.
  /// Wraps OrtApi::ExternalInitializerInfo_GetFilePath
  pub fn getFilePath(self: *const @This()) Utils.Path {
    return Api.ort.ExternalInitializerInfo_GetFilePath.?(@ptrCast(self));
  }

  /// Get the byte offset within the file where the initializer's data is stored.
  /// Wraps OrtApi::ExternalInitializerInfo_GetFileOffset
  pub fn getFileOffset(self: *const @This()) i64 {
    return Api.ort.ExternalInitializerInfo_GetFileOffset.?(@ptrCast(self));
  }

  /// Get the size in bytes of the initializer's data within the file.
  /// Wraps OrtApi::ExternalInitializerInfo_GetByteSize
  pub fn getByteSize(self: *const @This()) usize {
    return Api.ort.ExternalInitializerInfo_GetByteSize.?(@ptrCast(self));
  }

  /// Release an OrtExternalInitializerInfo instance.
  /// Wraps OrtApi::ReleaseExternalInitializerInfo
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseExternalInitializerInfo.?(@ptrCast(self));
  }
};

/// Wrapper around ::OrtPrepackedWeightsContainer
/// Create only and pass to Session constructor for multiple sessions to share pre-packed weights.
pub const PrepackedWeightsContainer = opaque {
  /// Wraps OrtApi::CreatePrepackedWeightsContainer
  pub fn init() !*@This() {
    var self: ?*@This() = null;
    try Error.check(Api.ort.CreatePrepackedWeightsContainer.?(@ptrCast(&self)));
    return self orelse error.OutOfMemory;
  }

  /// Release the container.
  /// Note: Instance must not be released until the sessions using it are released.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleasePrepackedWeightsContainer.?(@ptrCast(self));
  }
};

/// ShapeInferContext provides access to input metadata and attributes during the shape inference phase of an ONNX Runtime Custom Operator.
///
/// This context is passed to the custom operator's shape inference function to allow it to compute and set the output shapes based on input shapes and node attributes.
pub const ShapeInferContext = opaque {
  /// Returns the number of inputs provided to this operator node.
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.ShapeInferContext_GetInputCount.?(@ptrCast(self), &out));
    return out;
  }

  /// Get type and shape info of an input tensor.
  /// 
  /// index: The zero-based index of the input.
  ///
  /// Returns a pointer to the C representation of the tensor's type and shape.
  pub fn getInputTypeShape(self: *const @This(), index: usize) !*TensorTypeAndShapeInfo.C {
    var out: ?*TensorTypeAndShapeInfo.C = null;
    try Error.check(Api.ort.ShapeInferContext_GetInputTypeShape.?(@ptrCast(self), index, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get attribute from OrtShapeInferContext. Note that OrtShapeInferContext is a per-node context, one could only read attribute from current node.
  ///
  /// name: The null-terminated string name of the attribute as defined in the model.
  /// 
  /// Returns an optional pointer to the attribute, or null if the attribute is not found.
  pub fn getAttribute(self: *const @This(), name: [*:0]const u8) !*const OpAttr {
    var out: ?*const OpAttr = null;
    try Error.check(Api.ort.ShapeInferContext_GetAttribute.?(@ptrCast(self), name, @ptrCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Sets the inferred shape and type for a specific output of the operator.
  /// 
  /// This is the "final step" of shape inference where you provide the calculated output dimensions back to the onnx runtime.
  ///
  /// index: The zero-based index of the output to set.
  /// info: The computed type and shape information.
  pub fn setOutputTypeShape(self: *const @This(), index: usize, info: *const TensorTypeAndShapeInfo.C) !void {
    try Error.check(Api.ort.ShapeInferContext_SetOutputTypeShape.?(@ptrCast(self), index, @ptrCast(info)));
  }
};

const ProviderOptions = struct {
  /// Legacy Struct for OpenVINO. 
  /// Newer versions of ORT recommend using SessionOptionsAppendExecutionProvider_OpenVINO_V2 (which takes map)
  pub const OpenVINO = Api.c.OrtOpenVINOProviderOptions;

  /// Legacy Struct for MIGraphX
  pub const MIGraphX = Api.c.OrtMIGraphXProviderOptions;

  /// Legacy TensorRT options struct
  pub const TensorRT_V1 = Api.c.OrtTensorRTProviderOptions;

  /// Legacy CUDA options struct
  pub const CUDA_V1 = Api.c.OrtCUDAProviderOptions;

  /// TensorRT Provider Options (V2)
  pub const TensorRT = opaque {
    /// Wraps OrtApi::CreateTensorRTProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateTensorRTProviderOptions.?(@ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateTensorRTProviderOptions
    /// options: Array of keys and values.
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateTensorRTProviderOptions.?(@ptrCast(self), @ptrCast(keys.ptr), @ptrCast(values.ptr), keys.len));
    }

    /// Wraps OrtApi::UpdateTensorRTProviderOptionsWithValue
    /// Update an option where its data type is a pointer (e.g., user_compute_stream).
    pub fn updateWithValue(self: *@This(), key: [*:0]const u8, value: *anyopaque) !void {
      try Error.check(Api.ort.UpdateTensorRTProviderOptionsWithValue.?(@ptrCast(self), key, value));
    }

    /// Wraps OrtApi::GetTensorRTProviderOptionsByName
    /// Get a provider option where its data type is pointer.
    pub fn getByName(self: *const @This(), key: [*:0]const u8) !?*anyopaque {
      var out: ?*anyopaque = null;
      try Error.check(Api.ort.GetTensorRTProviderOptionsByName.?(@ptrCast(self), key, &out));
      return out;
    }

    /// Wraps OrtApi::GetTensorRTProviderOptionsAsString
    /// The returned string must be freed using the allocator.
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetTensorRTProviderOptionsAsString.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseTensorRTProviderOptions.?(@ptrCast(self));
    }
  };

  /// ROCM Provider Options
  /// Wraps ::OrtROCMProviderOptions
  pub const ROCM = struct {
    underlying: Api.c.OrtROCMProviderOptions,

    /// Wraps OrtApi::CreateROCMProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateROCMProviderOptions.?(@ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateROCMProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateROCMProviderOptions.?(@ptrCast(self), @ptrCast(keys.ptr), @ptrCast(values.ptr), keys.len));
    }

    /// Wraps OrtApi::GetROCMProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetROCMProviderOptionsAsString.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseROCMProviderOptions.?(@ptrCast(self));
    }
  };

  /// CUDA Provider Options (V2)
  pub const CUDA = opaque {
    /// Wraps OrtApi::CreateCUDAProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateCUDAProviderOptions.?(@ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateCUDAProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateCUDAProviderOptions.?(@ptrCast(self), @ptrCast(keys.ptr), @ptrCast(values.ptr), keys.len));
    }

    /// Wraps OrtApi::UpdateCUDAProviderOptionsWithValue
    pub fn updateWithValue(self: *@This(), key: [*:0]const u8, value: *anyopaque) !void {
      try Error.check(Api.ort.UpdateCUDAProviderOptionsWithValue.?(@ptrCast(self), key, value));
    }

    /// Wraps OrtApi::GetCUDAProviderOptionsByName
    /// Get a provider option where its data type is pointer.
    pub fn getByName(self: *const @This(), key: [*:0]const u8) !?*anyopaque {
      var out: ?*anyopaque = null;
      try Error.check(Api.ort.GetCUDAProviderOptionsByName.?(@ptrCast(self), key, &out));
      return out;
    }

    /// Wraps OrtApi::GetCUDAProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetCUDAProviderOptionsAsString.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseCUDAProviderOptions.?(@ptrCast(self));
    }
  };

  /// CANN Provider Options
  pub const CANN = opaque {
    /// Wraps OrtApi::CreateCANNProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateCANNProviderOptions.?(@ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateCANNProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateCANNProviderOptions.?(@ptrCast(self), @ptrCast(keys.ptr), @ptrCast(values.ptr), keys.len));
    }

    /// Wraps OrtApi::GetCANNProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetCANNProviderOptionsAsString.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseCANNProviderOptions.?(@ptrCast(self));
    }
  };

  /// DNNL Provider Options
  pub const Dnnl = opaque {
    /// Wraps OrtApi::CreateDnnlProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateDnnlProviderOptions.?(@ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateDnnlProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateDnnlProviderOptions.?(@ptrCast(self), @ptrCast(keys.ptr), @ptrCast(values.ptr), keys.len));
    }

    /// Wraps OrtApi::GetDnnlProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetDnnlProviderOptionsAsString.?(@ptrCast(self), @ptrCast(allocator), @ptrCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseDnnlProviderOptions.?(@ptrCast(self));
    }
  };
};

/// If we use std.testing.refAllDeclsRecursive, we get a compile error because c has untranslatable code, hence we use this
/// Even this touches the translated parts of the c code that we touch, but atleast not it doesn't crash
pub fn refAllDeclsRecursiveExcerptC(comptime T: type) void {
  if (!@import("builtin").is_test) return;
  inline for (comptime std.meta.declarations(T)) |decl| {
    _ = &@field(T, decl.name);
    if (@TypeOf(@field(T, decl.name)) == type) {
      if (decl.name.len == 1 and decl.name[0] == 'c') continue;
      switch (@typeInfo(@field(T, decl.name))) {
        .@"struct", .@"enum", .@"union", .@"opaque" => refAllDeclsRecursiveExcerptC(@field(T, decl.name)),
        else => {},
      }
    }
  }
}

test {
  refAllDeclsRecursiveExcerptC(@This());
}

