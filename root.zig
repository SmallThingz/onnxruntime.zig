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
              else => unreachable,
            },
            .@"enum" => @intFromEnum(v),
            .int => v,
            .bool => @intFromBool(v),
            else => unreachable,
          };
        } else {
          self._vals[self.len] = switch (@typeInfo(@TypeOf(v))) {
            .optional => |oi| switch (@typeInfo(oi.child)) {
              .pointer => |pi| switch (pi.size) {
                .many => v.?,
                .slice => v.?.ptr,
                .c, .one => unreachable,
              },
              else => unreachable,
            },
            .pointer => |pi| switch (pi.size) {
              .many => v,
              .slice => v.ptr,
              .c, .one => unreachable,
            },
            else => unreachable,
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

      pub fn keys(self: *const @This()) [*]const [*:0] const u8 { return (&self._keys).ptr; }
      pub fn vals(self: *const @This()) [*]const V { return (&self._vals).ptr; }
    };
  }

  /// Helper function to convert a struct to keys, values array and length, for V2 functions that take string arrays
  pub fn createOptionsKVL(instance: anytype, comptime V: VEnum) OptionsKVLRetvalType(@TypeOf(instance), V) {
    return .fromInstance(instance);
  }

  fn CopyPointerAttrs(From: type, size: std.builtin.Type.Pointer.Size, To: type) type {
    if (@typeInfo(To) == .@"opaque" and size != .one) {
      @compileError(std.fmt.comptimePrint("From: {s}; size: {any}; To: {s}", .{@typeName(From), size, @typeName(To)}));
    }
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

  fn ApiCast(comptime T: type) type {
    return switch (@typeInfo(T)) {
      .@"opaque", .@"struct" => T.Underlying,
      .@"enum" => @typeInfo(T).@"enum".tag_type,
      .pointer => |pi| blk: {
        std.debug.assert(pi.size != .slice);
        break :blk CopyPointerAttrs(T, if (@typeInfo(pi.child) == .@"opaque") .one else .c, ApiCast(pi.child));
      },
      .optional => |oi| blk: {
        const casted = ApiCast(oi.child);
        break :blk switch (@typeInfo(casted)) {
          .pointer => |pi| switch (pi.size) {
            .one => ?casted,
            .slice => unreachable,
            else => casted,
          },
          else => unreachable,
        };
      },
      else => unreachable,
    };
  }

  pub fn apiCast(anyptr: anytype) ApiCast(@TypeOf(anyptr)) {
    return @ptrCast(anyptr);
  }

  // fn ApiCastTo(T: type, To: type) type {
  //   return switch (@typeInfo(T)) {
  //     .@"opaque", .@"struct",  => blk: {
  //       std.debug.assert(T == To.Underlying);
  //       break :blk To.Underlying;
  //     },
  //     .@"enum" => blk: {
  //       std.debug.assert(T == @typeInfo(To).@"enum".tag_type);
  //       break :blk @typeInfo(To).@"enum".tag_type;
  //     },
  //     .pointer => |pi| switch (@typeInfo(To)) {
  //       .pointer => |topi| CopyPointerAttrs(T, topi.size, ApiCastTo(pi.child, topi.child)),
  //       .optional => |tooi| ?blk: {
  //         const topi = @typeInfo(tooi.child);
  //         break :blk CopyPointerAttrs(T, topi.size, ApiCastTo(pi.child, topi.child));
  //       },
  //       else => unreachable,
  //     },
  //     else => unreachable,
  //   };
  // }

  fn ApiCastTo(From: type, To: type) type {
    const CastedFrom = ApiCast(To);
    if (CastedFrom != From) @compileError(std.fmt.comptimePrint("From: {s}; Casted: {s}; To: {s}", .{@typeName(From), @typeName(CastedFrom), @typeName(To)}));
    return To;
  }

  pub fn apiCastTo(anyptr: anytype, comptime To: type) ApiCastTo(@TypeOf(anyptr), To) {
    return @ptrCast(anyptr);
  }

  fn SentinelStrCast(Str: type, T: type) type {
    if (Str == [*:0]const T) return [*c]const T;
    if (Str == [*:0]T) return [*c]T;
    return switch (@typeInfo(Str)) {
      .pointer => |pi| blk: {
        std.debug.assert(pi.size != .slice);
        break :blk CopyPointerAttrs(Str, .c, SentinelStrCast(pi.child, T));
      },
      .optional => |oi| SentinelStrCast(oi.child, T), // c pointers are already optional
      else => unreachable,
    };
  }

  fn SentinelStrCastTo(Str: type, T: type, To: type) type {
    if (Str == [*c]const T) {
      std.debug.assert(To == [*:0]const T or To == ?[*:0]const T);
      return To;
    }

    if (Str == [*c]T) {
      std.debug.assert(To == [*:0]T or To == ?[*:0]T);
      return To;
    }

    return switch (@typeInfo(Str)) {
      .pointer => |pi| blk: {
        std.debug.assert(pi.size != .slice);
        break :blk switch (@typeInfo(To)) {
          .pointer => |topi| blk2: {
            std.debug.assert(topi.size != .slice);
            break :blk2 CopyPointerAttrs(Str, topi.size, SentinelStrCastTo(pi.child, T, topi.child));
          },
          .optional => |tooi| ?blk2: {
            const topi = @typeInfo(tooi.child);
            break :blk2 CopyPointerAttrs(Str, topi.size, SentinelStrCastTo(pi.child, T, topi.child));
          },
          else => unreachable,
        };
      },
      else => unreachable,
    };
  }

  pub fn cStr(str: anytype) SentinelStrCast(@TypeOf(str), u8) {
    return @ptrCast(str);
  }

  pub fn cStrTo(str: anytype, comptime To: type) SentinelStrCastTo(@TypeOf(str), u8, To) {
    return @ptrCast(str);
  }

  pub fn pathCast(str: anytype) SentinelStrCast(@TypeOf(str), PathChar) {
    return @ptrCast(str);
  }

  pub fn pathCastTo(str: anytype, comptime To: type) SentinelStrCastTo(@TypeOf(str), PathChar, To) {
    return @ptrCast(str);
  }

  fn CCast(comptime T: type) type {
    return switch (@typeInfo(T)) {
      .@"enum" => @typeInfo(T).@"enum".tag_type,
      .pointer => |pi| blk: {
        std.debug.assert(pi.size != .slice);
        break :blk CopyPointerAttrs(T, if (@typeInfo(pi.child) == .@"opaque") .one else .c, CCast(pi.child));
      },
      .optional => |oi| blk: {
        const casted = CCast(oi.child);
        break :blk switch (@typeInfo(casted)) {
          .pointer => |pi| switch (pi.size) {
            .one => ?casted,
            .slice => unreachable,
            else => casted,
          },
          else => unreachable,
        };
      },
      else => T,
    };
  }

  pub fn cCast(anyptr: anytype) CCast(@TypeOf(anyptr)) {
    return @ptrCast(anyptr);
  }

  pub const empty_string = [_]u8{0};
  pub const empty_path = [_]PathChar{0};

  pub fn asPath(comptime str: []const u8) [:0]const PathChar {
    var retval: [str.len:0]PathChar = undefined;
    for (str, 0..) |c, i| retval[i] = c;
    retval[str.len] = 0;
    const const_retval = retval;
    return &const_retval;
  }
};

const apiCast = Utils.apiCast;
const apiCastTo = Utils.apiCastTo;
const cStr = Utils.cStr;
const cStrTo = Utils.cStrTo;
const pathCast = Utils.pathCast;
const pathCastTo = Utils.pathCastTo;
const cCast = Utils.cCast;

const empty_string = Utils.empty_string;
const empty_path = Utils.empty_path;

/// This is usually used by vendors
pub const Ep = struct {
  pub const api = opaque {
    pub var underlying: *const Api.c.OrtEpApi = undefined;
  };

  /// The Wrapper for OrtEp Struct
  pub const Interface = struct {
    pub const Underlying = Api.c.OrtEp;
    underlying: Underlying,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(Underlying)); }

    // Call the underlying vtable
    pub fn getName(self: *const @This()) [*:0]const u8 {
      return cStrTo(self.underlying.GetName.?(apiCast(self)), [*:0]const u8);
    }

    // Call the underlying vtable
    pub fn setDynamicOptions(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (self.underlying.SetDynamicOptions) |f| {
        try Error.check(f(apiCast(self), cStr(keys.ptr), cStr(values.ptr), keys.len));
      }
    }

    // Call the underlying vtable
    pub fn getCompiledModelCompatibilityInfo(self: *@This(), graph: *const Graph) [*:0]const u8 {
      return self.underlying.GetCompiledModelCompatibilityInfo.?(apiCast(self), apiCast(graph));
    }

    pub const DataLayout = enum(Api.c.OrtEpDataLayout) {
      NCHW = @bitCast(Api.c.OrtEpDataLayout_NCHW),
      NHWC = @bitCast(Api.c.OrtEpDataLayout_NHWC),
    };

    /// Initialize the Ep structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer (T):
    /// - Must have a field `ep: Ep.Interface`.
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
        fn getSelf(ptr: [*c]const Api.c.OrtEp) *T {
          return @constCast(@fieldParentPtr("ep", apiCastTo(ptr.?, *const Ep.Interface)));
        }

        fn getName(ptr: [*c]const Api.c.OrtEp) callconv(.c) [*c]const u8 {
          return cStr(getSelf(ptr).getName());
        }

        fn getCapability(ptr: [*c]Api.c.OrtEp, graph: ?*const Api.c.OrtGraph, info: ?*Api.c.OrtEpGraphSupportInfo) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).getCapability(apiCastTo(graph.?, *const Graph), apiCastTo(info.?, *GraphSupportInfo)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn compile(
          ptr: [*c]Api.c.OrtEp,
          graphs: [*c]?*const Api.c.OrtGraph,
          fused: [*c]?*const Api.c.OrtNode,
          count: usize,
          infos_out: [*c][*c]Api.c.OrtNodeComputeInfo,
          context_nodes_out: [*c]?*Api.c.OrtNode,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const g_slice_optional = apiCastTo(graphs, [*]?*const Graph);
          const f_slice_optional = apiCastTo(fused, [*]?*const Node);
          const i_slice_optional = apiCastTo(infos_out, [*]?*NodeCompute.Info);
          const c_slice_optional = apiCastTo(context_nodes_out, [*]?*Node);

          const g_slice = @as([*]*const Graph, @ptrCast(g_slice_optional))[0..count];
          const f_slice = @as([*]*const Node, @ptrCast(f_slice_optional))[0..count];
          const i_slice = @as([*]*NodeCompute.Info, @ptrCast(i_slice_optional))[0..count];
          const c_slice = @as([*]*Node, @ptrCast(c_slice_optional))[0..count];

          getSelf(ptr).compile(g_slice, f_slice, i_slice, c_slice) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn releaseNodeComputeInfos(ptr: [*c]Api.c.OrtEp, infos: [*c][*c]Api.c.OrtNodeComputeInfo, count: usize) callconv(.c) void {
          getSelf(ptr).releaseNodeComputeInfos(apiCastTo(infos, [*]*NodeCompute.Info)[0..count]);
        }

        fn getPreferredDataLayout(ptr: [*c]Api.c.OrtEp, layout_out: [*c]Api.c.OrtEpDataLayout) callconv(.c) ?*Api.c.OrtStatus {
          const layout = getSelf(ptr).getPreferredDataLayout() catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          layout_out.?.* = @intFromEnum(layout);
          return null;
        }

        fn shouldConvertDataLayoutForOp(ptr: [*c]Api.c.OrtEp, domain: [*:0]const u8, op_type: [*:0]const u8, layout: Api.c.OrtEpDataLayout, out: [*c]c_int) callconv(.c) ?*Api.c.OrtStatus {
          out.?.* = getSelf(ptr).shouldConvertDataLayoutForOp(domain, op_type, @enumFromInt(layout));
          return null;
        }

        fn setDynamicOptions(ptr: [*c]Api.c.OrtEp, keys: [*c]const [*:0]const u8, vals: [*c]const [*:0]const u8, count: usize) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).setDynamicOptions(keys[0..count], vals[0..count]) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn onRunStart(ptr: [*c]Api.c.OrtEp, options: [*c]const Api.c.OrtRunOptions) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).onRunStart(apiCastTo(options.?, *const RunOptions)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn onRunEnd(ptr: [*c]Api.c.OrtEp, options: [*c]const Api.c.OrtRunOptions, sync: bool) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(ptr).onRunEnd(apiCastTo(options.?, *const RunOptions), sync) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn createAllocator(ptr: [*c]Api.c.OrtEp, info: [*c]const Api.c.OrtMemoryInfo, out: [*c][*c]Api.c.OrtAllocator) callconv(.c) ?*Api.c.OrtStatus {
          const alloc = getSelf(ptr).createAllocator(apiCastTo(info.?, *const Allocator.MemoryInfo)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          out.?.* = apiCast(alloc);
          return null;
        }

        fn createSyncStreamForDevice(ptr: [*c]Api.c.OrtEp, dev: [*c]const Api.c.OrtMemoryDevice, out: [*c][*c]Api.c.OrtSyncStreamImpl) callconv(.c) ?*Api.c.OrtStatus {
          const stream = getSelf(ptr).createSyncStreamForDevice(apiCastTo(dev.?, *const Allocator.MemoryDevice)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          out.?.* = apiCast(stream);
          return null;
        }

        fn getCompiledModelCompatibilityInfo(ptr: [*c]Api.c.OrtEp, graph: ?*const Api.c.OrtGraph) callconv(.c) [*:0]const u8 {
          return getSelf(ptr).getCompiledModelCompatibilityInfo(apiCastTo(graph.?, *const Graph));
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
    pub const Underlying = Api.c.OrtDataTransferImpl;
    underlying: Underlying,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(Underlying)); }

    /// Check if the implementation can copy between the source and destination memory devices.
    /// returns true if the implementation can copy between the devices.
    /// since Version 1.23.
    pub fn canCopy(self: *const @This(), src: *const Allocator.MemoryDevice, dst: *const Allocator.MemoryDevice) bool {
      return self.underlying.CanCopy.?(apiCast(self), apiCast(src), apiCast(dst));
    }

    /// Copy tensors from src_tensors to dst_tensors using the provided streams.
    ///
    /// The implementation can use the provided streams to perform asynchronous copies if supported.
    /// If a stream is not available, the copy is performed synchronously.
    ///
    /// streams: Array of OrtSyncStream pointers for the copy operations, if the execution provider is stream aware. nullptr if it is not.
    ///
    /// since Version 1.23.
    pub fn copy(self: *@This(), src: []*const Value, dst: []*Value, streams: ?[]?*SyncStream) !void {
      std.debug.assert(src.len == dst.len);
      if (streams) |s| std.debug.assert(src.len == s.len);
      const src_ptr: [*]?*const Value = @ptrCast(src.ptr);
      const dst_ptr: [*]?*Value = @ptrCast(dst.ptr);
      try Error.check(self.underlying.CopyTensors.?(
          apiCast(self),
          apiCast(src_ptr),
          apiCast(dst_ptr),
          apiCast(if (streams) |s| s.ptr else null),
          src.len,
      ));
    }

    /// Release the OrtDataTransferImpl instance.
    ///
    /// This is called by ORT when the OrtDataTransferImpl instance is no longer needed.
    /// The implementation should release any resources held by the instance.
    ///
    /// since Version 1.23.
    pub fn deinit(self: *@This()) void {
      return self.underlying.Release.?(apiCast(self));
    }

    /// Initialize the DataTransfer structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer:
    /// - Must have a field `data_transfer: DataTransfer` as the a member, we use @fieldParentPtr on that member to get actual pointer
    /// - fn canCopy(self: *const Implementer, src: *const Allocator.MemoryDevice, dst: *const Allocator.MemoryDevice) bool
    /// - fn copyTensors(self: *Implementer, src: []const *const Value, dst: []*Value, streams: ?[]?*SyncStream) !void
    /// - (Optional) fn deinit(self: *Implementer) void
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(self: [*c]const Api.c.OrtDataTransferImpl) *T {
          return @constCast(@fieldParentPtr("data_transfer", apiCastTo(self.?, *const DataTransfer)));
        }

        fn release(self: [*c]Api.c.OrtDataTransferImpl) callconv(.c) void {
          if (@hasDecl(T, "deinit")) getSelf(self).deinit();
        }

        fn canCopy(
          self: [*c]const Api.c.OrtDataTransferImpl,
          src: ?*const Api.c.OrtMemoryDevice,
          dst: ?*const Api.c.OrtMemoryDevice,
        ) callconv(.c) bool {
          return getSelf(self).canCopy(apiCastTo(src.?, *const Allocator.MemoryDevice), apiCastTo(dst.?, *const Allocator.MemoryDevice));
        }

        fn copyTensors(
          self: [*c]Api.c.OrtDataTransferImpl,
          src: [*c]?*const Api.c.OrtValue,
          dst: [*c]?*Api.c.OrtValue,
          streams: [*c]?*Api.c.OrtSyncStream,
          num: usize,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const src_slice_optional = apiCastTo(src, [*]?*const Value);
          const dst_slice_optional = apiCastTo(dst, [*]?*Value);

          const src_slice = @as([*]const *const Value, @ptrCast(src_slice_optional))[0..num];
          const dst_slice = @as([*]*Value, @ptrCast(dst_slice_optional))[0..num];
          const streams_slice = apiCastTo(streams, [*]?*SyncStream)[0..num];

          getSelf(self).copyTensors(src_slice, dst_slice, streams_slice) catch |err| 
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
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
    pub const Underlying = Api.c.OrtSyncNotificationImpl;
    underlying: Underlying,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(Underlying)); }

    /// Called by ORT to activate the notification.
    ///
    /// since Version 1.23.
    pub fn activate(self: *@This()) !void {
      try Error.check(self.underlying.Activate.?(apiCast(self)));
    }

    /// Wait for a device to device operation to complete.
    ///
    /// this_ptr: Pointer to the OrtSyncNotificationImpl instance.
    /// stream: The OrtSyncStream instance that will wait on this notification to be activated.
    ///
    /// since Version 1.23.
    pub fn waitOnDevice(self: *@This(), stream: *SyncStream) !void {
      try Error.check(self.underlying.WaitOnDevice.?(apiCast(self), apiCast(stream)));
    }

    /// Wait for a device to host operation to complete.
    ///
    /// since Version 1.23.
    pub fn waitOnHost(self: *@This()) !void {
      try Error.check(self.underlying.WaitOnHost.?(apiCast(self)));
    }

    /// Release the OrtSyncNotificationImpl instance.
    ///
    /// This is called by ORT when the OrtSyncNotificationImpl instance is no longer needed.
    /// The implementation should release any resources held by the instance.
    ///
    /// since Version 1.23.
    pub fn deinit(self: *@This()) void {
      self.underlying.Release.?(apiCast(self));
    }

    /// Initialize the SyncNotification structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer:
    /// - Must have a field `sync_notification: SyncNotificationImpl` as a member.
    /// - fn activate(self: *Implementer) !void
    /// - fn waitOnDevice(self: *Implementer, consumer_stream: *SyncStream) !void
    /// - fn waitOnHost(self: *Implementer) !void
    /// - (Optional) fn deinit(self: *Implementer) void
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(self: [*c]Api.c.OrtSyncNotificationImpl) *T {
          return @constCast(@fieldParentPtr("sync_notification", apiCastTo(self.?, *SyncNotificationImpl)));
        }

        fn release(self: [*c]Api.c.OrtSyncNotificationImpl) callconv(.c) void {
          if (@hasDecl(T, "deinit")) getSelf(self).deinit();
        }

        fn activate(self: [*c]Api.c.OrtSyncNotificationImpl) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(self).activate() catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn waitOnDevice(self: [*c]Api.c.OrtSyncNotificationImpl, stream: ?*Api.c.OrtSyncStream) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(self).waitOnDevice(@as(*SyncStream, apiCastTo(stream.?, *SyncStream))) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn waitOnHost(self: [*c]Api.c.OrtSyncNotificationImpl) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(self).waitOnHost() catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
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

  /// Used for synchronization and async execution / data transfer etc
  pub const SyncStream = opaque {
    pub const Underlying = Api.c.OrtSyncStream;
    /// Wraps OrtApi::CreateSyncStreamForEpDevice
    /// stream_options: Optional KeyValuePairs for stream configuration.
    pub fn init(device: *const Ep.Device, stream_options: ?*const KeyValuePairs) !*@This() {
      var retval: ?*@This() = null;
      try Error.check(Api.ort.CreateSyncStreamForEpDevice.?(apiCast(device), apiCast(stream_options), apiCast(&retval)));
      return retval orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::SyncStream_GetHandle
    /// Returns the native handle
    pub fn getHandle(self: *@This()) ?*anyopaque {
      return Api.ort.SyncStream_GetHandle.?(apiCast(self));
    }

    /// Wraps OrtApi::ReleaseSyncStream
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseSyncStream.?(apiCast(self));
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
    pub fn getImpl(self: *const @This()) *const Impl {
      return apiCastTo(Ep.api.underlying.SyncStream_GetImpl.?(apiCast(self)), *const Impl);
    }

    /// Get the current sync ID for a stream.
    ///
    /// stream: The OrtSyncStream to get the sync ID for.
    /// returns Current sync ID.
    ///
    /// since Version 1.23.
    pub fn getSyncId(self: *const @This()) u64 {
      return Ep.api.underlying.SyncStream_GetSyncId.?(apiCast(self));
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
      return Ep.api.underlying.GetSyncIdForLastWaitOnSyncStream.?(apiCast(producer), apiCast(consumer));
    }

    /// Struct that an EP implements if it wishes to implement Stream support.
    /// This struct provides the overrides for onnxruntime::Stream's virtual methods.
    ///
    /// since Version 1.23.
    pub const Impl = struct {
      pub const Underlying = Api.c.OrtSyncStreamImpl;
      underlying: @This().Underlying,
      comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@This().Underlying)); }

      ///Get the handle of the stream.
      ///
      /// This returns the native handle for the stream. e.g. cudaStream_t for CUDA streams.
      ///
      /// since Version 1.23.
      pub fn handle(self: *@This()) ?*anyopaque {
        return self.underlying.GetHandle.?(apiCast(self));
      }

      /// Create an OrtSyncNotificationImpl for the OrtSyncStreamImpl instance.
      ///
      /// since Version 1.23.
      pub fn createNotification(self: *@This()) !*SyncNotificationImpl {
        var out: ?*SyncNotificationImpl = null;
        try Error.check(self.underlying.CreateNotification.?(apiCast(self), apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Notify the stream that a session run has ended.
      ///
      /// This is called by ORT to notify the stream that a session run has ended, allowing the stream to perform any
      /// necessary cleanup or finalization.
      ///
      /// since Version 1.23.
      pub fn endSessionRun(self: *@This()) !void {
        try Error.check(self.underlying.OnSessionRunEnd.?(apiCast(self)));
      }

      /// Flush the stream.
      ///
      /// This is called by ORT to flush the stream, ensuring that all operations submitted to the stream are completed.
      ///
      /// since Version 1.23.
      pub fn flush(self: *@This()) !void {
        try Error.check(self.underlying.Flush.?(apiCast(self)));
      }

      /// Release the OrtSyncStreamImpl instance.
      ///
      /// This is called by ORT when the OrtSyncStreamImpl instance is no longer needed.
      /// The implementation should release any resources held by the instance.
      ///
      /// since Version 1.23.
      pub fn deinit(self: *@This()) void {
        self.underlying.Release.?(apiCast(self));
      }

      /// Requirements for Implementer:
      /// - Must have a field `sync_stream: SyncStreamImpl`.
      /// - fn getHandle(self: *Implementer) ?*anyopaque
      /// - fn createNotification(self: *Implementer, out: **SyncNotificationImpl) !void
      /// - fn flush(self: *Implementer) !void
      /// - fn onSessionRunEnd(self: *Implementer) !void
      pub fn init(comptime T: type) @This() {
        const VTable = struct {
          fn getSelf(self: [*c]Api.c.OrtSyncStreamImpl) *T {
            return @constCast(@fieldParentPtr("sync_stream", apiCastTo(self.?, *Impl)));
          }

          fn release(self: [*c]Api.c.OrtSyncStreamImpl) callconv(.c) void {
            if (@hasDecl(T, "deinit")) getSelf(self).deinit();
          }

          fn getHandle(self: [*c]Api.c.OrtSyncStreamImpl) callconv(.c) ?*anyopaque {
            return getSelf(self).getHandle();
          }

          fn createNotification(self: [*c]Api.c.OrtSyncStreamImpl, out: [*c][*c]Api.c.OrtSyncNotificationImpl) callconv(.c) ?*Api.c.OrtStatus {
            getSelf(self).createNotification(@as(*?*SyncNotificationImpl, apiCastTo(out.?, *?*SyncNotificationImpl))) catch |err|
              return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
            return null;
          }

          fn flush(self: [*c]Api.c.OrtSyncStreamImpl) callconv(.c) ?*Api.c.OrtStatus {
            getSelf(self).flush() catch |err|
              return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
            return null;
          }

          fn onSessionRunEnd(self: [*c]Api.c.OrtSyncStreamImpl) callconv(.c) ?*Api.c.OrtStatus {
            getSelf(self).onSessionRunEnd() catch |err|
              return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
            return null;
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
  };

  pub const NodeFusionOptions = struct {
    const Underlying = Api.c.OrtNodeFusionOptions;
    underlying: Underlying,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(Underlying)); }
  };

  pub const NodeCompute = struct {
    /// Opaque to add type to createState
    pub const State = opaque {
      // This has no underlying type
    };

    pub const Context = opaque {
      pub const Underlying = Api.c.OrtNodeComputeContext;
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
        return cStrTo(api.underlying.NodeComputeContext_NodeName.?(apiCast(self)), [*:0]const u8);
      }
    };

    /// The OrtNodeComputeInfo struct provides functions that an OrtEp implements to specify the compute function for a compiled OrtGraph instance.
    /// since Version 1.23.
    pub const Info = struct {
      pub const Underlying = Api.c.OrtNodeComputeInfo;
      underlying: Underlying,
      comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(Underlying)); }

      /// Creates an opaque compute state object that is then passed to the Compute() function during inference.
      /// compute_context OrtNodeComputeContext instance that contains compiled/fused node's name and host
      ///                 memory allocation functions. Can optionally be used to build the compute state.
      /// returns: compute_state: the opaque computation state. ONNX Runtime calls ReleaseState() (after calling Compute())
      ///          to allow the implementer to release the compute state.
      ///
      /// since Version 1.23.
      pub fn createState(self: *@This(), context: *Context) !*State {
        var out: ?*State = null;
        // @ptrCast is needed because State has no underlying type
        try Error.check(self.underlying.CreateState.?(apiCast(self), apiCast(context), @ptrCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Computation function called to execute the fused node compiled by an OrtEp instance.
      ///
      /// compute_state: The opaque computation state returned by CreateState().
      /// kernel_context: The OrtKernelContext instance used to access inputs/outputs.
      ///
      /// since Version 1.23.
      pub fn compute(self: *@This(), state: *State, kernel_context: *Op.KernelContext) !void {
        // @ptrCast is needed because State has no underlying type
        try Error.check(self.underlying.Compute.?(apiCast(self), @ptrCast(state), apiCast(kernel_context)));
      }

      /// Releases the compute state returned by CreateState().
      ///
      /// compute_state: The opaque compute state returned by CreateState().
      ///
      /// since Version 1.23.
      pub fn releaseState(self: *@This(), state: *State) void {
        // @ptrCast is needed because State has no underlying type
        self.underlying.ReleaseState.?(apiCast(self), @ptrCast(state));
      }

      /// Initialize the NodeCompute.Info structure with vtables pointing to the provided Implementer type.
      ///
      /// Requirements for Implementer:
      /// - Must have a field `compute_info: Ep.NodeCompute.Info`.
      /// - fn createState(self: *Implementer, context: *Context) !*State
      /// - fn compute(self: *Implementer, state: ?*State, kernel_context: *KernelContext) !void
      /// - fn releaseState(self: *Implementer, state: ?*State) void
      pub fn init(comptime T: type) @This() {
        const VTable = struct {
          fn getSelf(self: [*c]Api.c.OrtNodeComputeInfo) *T {
            return @fieldParentPtr("compute_info", apiCastTo(self.?, *Info));
          }

          fn createState(
            self: [*c]Api.c.OrtNodeComputeInfo,
            ctx: ?*Api.c.OrtNodeComputeContext,
            state_out: ?*?*anyopaque,
          ) callconv(.c) ?*Api.c.OrtStatus {
            const result = getSelf(self).createState(apiCastTo(ctx.?, *Context)) catch |err|
              return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
            state_out.?.* = @ptrCast(result);
            return null;
          }

          fn compute(
            self: [*c]Api.c.OrtNodeComputeInfo,
            state: ?*anyopaque,
            kernel_ctx: ?*Api.c.OrtKernelContext,
          ) callconv(.c) ?*Api.c.OrtStatus {
            // @ptrCast is needed because State has no underlying type
            getSelf(self).compute(@ptrCast(state), apiCastTo(kernel_ctx.?, *Op.KernelContext)) catch |err|
              return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
            return null;
          }

          fn releaseState(self: [*c]Api.c.OrtNodeComputeInfo, state: ?*anyopaque) callconv(.c) void {
            // @ptrCast is needed because State has no underlying type
            getSelf(self).releaseState(@ptrCast(state));
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
    pub const Underlying = Api.c.OrtEpGraphSupportInfo;
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
          apiCast(self),
          apiCast(nodes.ptr),
          nodes.len,
          apiCast(options),
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
          apiCast(self),
          apiCast(node),
      ));
    }
  };

  // this is const everywhere
  /// Represents an instance of an Execution Provider mapped to a specific hardware device.
  /// Wraps OrtEpDevice.
  pub const Device = opaque {
    pub const Underlying = Api.c.OrtEpDevice;
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
          apiCast(factory),
          apiCast(hw_device),
          apiCast(ep_metadata),
          apiCast(ep_options),
          apiCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Returns the name of the execution provider (e.g., "CUDA", "CPU").
    /// Wraps OrtApi::EpDevice_EpName
    pub fn getEpName(self: *const @This()) [*:0]const u8 {
      return cStrTo(Api.ort.EpDevice_EpName.?(apiCast(self)), [*:0]const u8);
    }

    /// Returns the name of the execution provider's vendor.
    /// Wraps OrtApi::EpDevice_EpVendor
    pub fn getEpVendor(self: *const @This()) [*:0]const u8 {
      return cStrTo(Api.ort.EpDevice_EpVendor.?(apiCast(self)), [*:0]const u8);
    }

    /// Returns an OrtKeyValuePairs instance containing the metadata for the EP device.
    /// Note: ORT owns this instance; do NOT call deinit on it.
    /// Wraps OrtApi::EpDevice_EpMetadata
    pub fn getEpMetadata(self: *const @This()) *const KeyValuePairs {
      return apiCastTo(Api.ort.EpDevice_EpMetadata.?(apiCast(self)).?, *const KeyValuePairs);
    }

    /// Returns an OrtKeyValuePairs instance containing the options for the EP device.
    /// Note: ORT owns this instance; do NOT call deinit on it.
    /// Wraps OrtApi::EpDevice_EpOptions
    pub fn getEpOptions(self: *const @This()) *const KeyValuePairs {
      return apiCastTo(Api.ort.EpDevice_EpOptions.?(apiCast(self)).?, *const KeyValuePairs);
    }

    /// Returns the underlying HardwareDevice (physical CPU/GPU/NPU) instance for this EP device.
    /// Note: ORT owns this instance.
    /// Wraps OrtApi::EpDevice_Device
    pub fn getHardwareDevice(self: *const @This()) *const HardwareDevice {
      return apiCastTo(Api.ort.EpDevice_Device.?(apiCast(self)).?, *const HardwareDevice);
    }

    /// Get the OrtMemoryInfo for the device.
    /// If memory_type is DEFAULT and null is returned, the EP uses CPU memory.
    /// Wraps OrtApi::EpDevice_MemoryInfo
    pub fn getMemoryInfo(self: *const @This(), mem_type: Allocator.DeviceMemoryType) ?*const Allocator.MemoryInfo {
      return apiCastTo(Api.ort.EpDevice_MemoryInfo.?(apiCast(self), @intFromEnum(mem_type)), ?*const Allocator.MemoryInfo);
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
      try Error.check(api.underlying.EpDevice_AddAllocatorInfo.?(apiCast(self), apiCast(info)));
    }

    /// Releases the EpDevice instance. 
    /// Use only if you manually created the device via CreateEpDevice.
    /// Devices from Api.env.getEpDevices() are managed by the environment.
    pub fn deinit(self: *@This()) void {
      api.underlying.ReleaseEpDevice.?(apiCast(self));
    }
  };

  pub const Factory = struct {
    pub const Underlying = Api.c.OrtEpFactory;
    underlying: Underlying,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(Underlying)); }

    pub fn getName(self: *const @This()) [*:0]const u8 {
      return cStrTo(self.underlying.GetName.?(apiCast(self)), [*:0]const u8);
    }

    pub fn getVersion(self: *const @This()) [*:0]const u8 {
      return self.underlying.GetVersion.?(apiCast(self));
    }

    pub fn validateCompatibility(self: *@This(), devices: []const *const HardwareDevice, info: [*:0]const u8) !Api.compiler.CompiledModelCompatibility {
      var out: Api.compiler.CompiledModelCompatibility = undefined;
      try Error.check(self.underlying.ValidateCompiledModelCompatibilityInfo.?(apiCast(self), apiCast(devices.ptr), devices.len, info, apiCast(&out)));
      return out;
    }

    /// Initialize the Factory structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer (T):
    /// - Must have a field `factory: Ep.Factory`.
    /// - fn getName(self: *const T) [*:0]const u8
    /// - fn getVendor(self: *const T) [*:0]const u8
    /// - fn getSupportedDevices(self: *T, devices: []const *const HardwareDevice, ep_devices_out: []*Ep.Device) !usize (return num added)
    /// - fn createEp(self: *T, devices: []const *const HardwareDevice, metadata: []const *const KeyValuePairs, options: *const Session.Options.C, logger: *const KernelInfo.Logger) !*Ep.Interface
    /// - fn releaseEp(self: *T, ep: *Ep.Interface) void
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
        fn getSelf(ptr: [*c]const Api.c.OrtEpFactory) *T {
          return @fieldParentPtr("factory", @as(*Factory, @constCast(@ptrCast(ptr.?))));
        }

        fn getName(ptr: [*c]const Api.c.OrtEpFactory) callconv(.c) [*c]const u8 {
          return cStr(getSelf(ptr).getName());
        }

        fn getVendor(ptr: [*c]const Api.c.OrtEpFactory) callconv(.c) [*c]const u8 {
          return cStr(getSelf(ptr).getVendor());
        }

        fn getSupportedDevices(
          ptr: [*c]Api.c.OrtEpFactory,
          devices: [*c]const ?*const Api.c.OrtHardwareDevice,
          num_devices: usize,
          ep_devices_out: [*c]?*Api.c.OrtEpDevice,
          max_ep_devices: usize,
          num_ep_devices_out: ?*usize,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const d_slice_optional = apiCastTo(devices, [*]const ?*const HardwareDevice);
          const e_slice_optional = apiCastTo(ep_devices_out, [*]?*Ep.Device);

          const d_slice = @as([*]const *const HardwareDevice, @ptrCast(d_slice_optional))[0..num_devices];
          const e_slice = @as([*]*Ep.Device, @ptrCast(e_slice_optional))[0..max_ep_devices];

          const count = getSelf(ptr).getSupportedDevices(d_slice, e_slice) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          num_ep_devices_out.?.* = count;
          return null;
        }

        fn createEp(
          ptr: [*c]Api.c.OrtEpFactory,
          devices: [*c]const ?*const Api.c.OrtHardwareDevice,
          metadata: [*c]const ?*const Api.c.OrtKeyValuePairs,
          num_devices: usize,
          options: ?*const Api.c.OrtSessionOptions,
          logger: ?*const Api.c.OrtLogger,
          ep_out: ?*?*Api.c.OrtEp,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const d_slice_optional = apiCastTo(devices, [*]const ?*const HardwareDevice);
          const m_slice_optional = apiCastTo(metadata, [*]const ?*const KeyValuePairs);

          const d_slice = @as([*]const *const HardwareDevice, @ptrCast(d_slice_optional))[0..num_devices];
          const m_slice = @as([*]const *const KeyValuePairs, @ptrCast(m_slice_optional))[0..num_devices];

          const ep = getSelf(ptr).createEp(d_slice, m_slice, apiCastTo(options.?, *const Session.Options.C), apiCastTo(logger.?, *const Op.KernelInfo.Logger)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          ep_out.?.* = apiCast(ep);
          return null;
        }

        fn releaseEp(ptr: [*c]Api.c.OrtEpFactory, ep: [*c]Api.c.OrtEp) callconv(.c) void {
          getSelf(ptr).releaseEp(apiCastTo(ep.?, *Ep.Interface));
        }

        fn getVendorId(ptr: [*c]const Api.c.OrtEpFactory) callconv(.c) u32 {
          return getSelf(ptr).getVendorId();
        }

        fn getVersion(ptr: [*c]const Api.c.OrtEpFactory) callconv(.c) [*:0]const u8 {
          return getSelf(ptr).getVersion();
        }

        fn validateCompatibility(
          ptr: [*c]Api.c.OrtEpFactory,
          devices: [*c]const ?*const Api.c.OrtHardwareDevice,
          num_devices: usize,
          info: [*c]const u8,
          out: [*c]Api.c.OrtCompiledModelCompatibility,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const d_slice_optional = apiCastTo(devices, [*]const ?*const HardwareDevice);
          const d_slice = @as([*]const *const HardwareDevice, @ptrCast(d_slice_optional))[0..num_devices];
          out.?.* = @intFromEnum(getSelf(ptr).validateCompiledModelCompatibilityInfo(d_slice, info));
          return null;
        }

        fn createAllocator(
          ptr: [*c]Api.c.OrtEpFactory,
          info: ?*const Api.c.OrtMemoryInfo,
          opts: ?*const Api.c.OrtKeyValuePairs,
          out: ?*?*Api.c.OrtAllocator,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const alloc = getSelf(ptr).createAllocator(apiCastTo(info, ?*const Allocator.MemoryInfo), apiCastTo(opts, ?*const KeyValuePairs)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          out.?.* = apiCast(alloc);
          return null;
        }

        fn releaseAllocator(ptr: [*c]Api.c.OrtEpFactory, alloc: ?*Api.c.OrtAllocator) callconv(.c) void {
          getSelf(ptr).releaseAllocator(apiCastTo(alloc.?, *Allocator));
        }

        fn createDataTransfer(ptr: [*c]Api.c.OrtEpFactory, out: ?*?*Api.c.OrtDataTransferImpl) callconv(.c) ?*Api.c.OrtStatus {
          const dt = getSelf(ptr).createDataTransfer() catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          out.?.* = apiCast(dt);
          return null;
        }

        fn isStreamAware(ptr: [*c]const Api.c.OrtEpFactory) callconv(.c) bool {
          return getSelf(ptr).isStreamAware();
        }

        fn createSyncStream(
          ptr: [*c]Api.c.OrtEpFactory,
          dev: ?*const Api.c.OrtMemoryDevice,
          opts: ?*const Api.c.OrtKeyValuePairs,
          out: ?*[*c]Api.c.OrtSyncStreamImpl,
        ) callconv(.c) ?*Api.c.OrtStatus {
          const stream = getSelf(ptr).createSyncStreamForDevice(apiCastTo(dev.?, *const Allocator.MemoryDevice), apiCastTo(opts, ?*const KeyValuePairs)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          out.?.* = apiCast(stream);
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
          .CreateSyncStreamForDevice = if (@hasDecl(T, "createSyncStreamForDevice")) VTable.createSyncStream else null,
        },
      };
    }
  };
};

/// Training of models.
pub const Training = struct {
  pub const api = struct {
    pub var underlying: *const Api.c.OrtTrainingApi = undefined;
    
    /// Sets the seed used for random number generation in Onnxruntime.
    ///
    /// Use this function to generate reproducible results. It should be noted that completely reproducible
    /// results are not guaranteed.
    pub fn setSeed(seed: i64) !void {
      try Error.check(api.underlying.SetSeed.?(seed));
    }
  };

  pub const CheckpointState = opaque {
    pub const Underlying = Api.c.OrtCheckpointState;
    /// Load a checkpoint state from a file on disk into checkpoint_state.
    ///
    /// This function will parse a checkpoint file, pull relevant data and load the training
    /// state into the checkpoint_state. This checkpoint state can then be used to create the
    /// training session by invoking OrtTrainingApi::CreateTrainingSession. By doing so, the training
    /// session will resume training from the given checkpoint state.
    /// Note: the training session created with a checkpoint state uses this state to store the entire
    /// training state (including model parameters, its gradients, the optimizer states and the properties).
    /// As a result, it is required that the checkpoint state outlive the lifetime of the training session.
    /// Note: The checkpoint file can be either the complete checkpoint or the nominal checkpoint.
    ///
    /// checkpoint_path: Path to the checkpoint file
    /// checkpoint_state: Checkpoint state that contains the states of the training session.
    pub fn load(path: Utils.Path) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.LoadCheckpoint.?(pathCast(path), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Load a checkpoint state from a buffer.
    pub fn loadFromBuffer(buffer: []const u8) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.LoadCheckpointFromBuffer.?(
        cCast(buffer.ptr), 
        buffer.len, 
        apiCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    ///Save the given state to a checkpoint file on disk.
    ///
    /// This function serializes the provided checkpoint state to a file on disk.
    /// This checkpoint can later be loaded by invoking OrtTrainingApi::LoadCheckpoint to resume
    /// training from this snapshot of the state.
    ///
    /// checkpoint_state: The checkpoint state to save.
    /// checkpoint_path: Path to the checkpoint file.
    /// include_optimizer_state: Flag to indicate whether to save the optimizer state or not.
    pub fn save(self: *@This(), path: Utils.Path, include_optimizer_state: bool) !void {
      try Error.check(api.underlying.SaveCheckpoint.?(
        apiCast(self), 
        pathCast(path), 
        include_optimizer_state
      ));
    }

    /// Adds or updates the given property to/in the checkpoint state.
    ///
    /// Runtime properties such as epoch, training step, best score, and others can be added to the checkpoint
    /// state by the user by calling this function with the corresponding property name and value.
    /// The given property name must be unique to be able to successfully add the property.
    ///
    /// property_name Name of the property being added or updated.
    /// property_type Type of the property associated with the given name.
    /// property_value Property value associated with the given name.
    pub fn addProperty(self: *@This(), name: [*c]const u8, prop_type: PropertyType, value: *anyopaque) !void {
      try Error.check(api.underlying.AddProperty.?(
        apiCast(self),
        cStrTo(name, [*:0]const u8),
        @intFromEnum(prop_type),
        value
      ));
    }

    /// Gets the property value associated with the given name from the checkpoint state.
    ///
    /// Gets the property value from an existing entry in the checkpoint state. The property must
    /// exist in the checkpoint state to be able to retrieve it successfully.
    ///
    /// property_name: Name of the property being retrieved.
    /// allocator: Allocator used to allocate the memory for the property_value.
    pub fn getProperty(self: *const @This(), name: [*c]const u8, allocator: *Allocator) !PropertyUnion {
      var out_type: PropertyType = undefined;
      var out_val: ?*anyopaque = null;
      try Error.check(api.underlying.GetProperty.?(
        apiCast(self),
        cStrTo(name, [*:0]const u8),
        apiCast(allocator),
        apiCast(&out_type),
        &out_val
      ));

      const val = out_val orelse return error.OutOfMemory;
      switch (out_type) {
        // Aligncast is ok since the alignment must match that of the returned type and because anyopaque pointers have no associated alignment
        inline else => |t| return @unionInit(PropertyUnion, @tagName(t), @alignCast(@ptrCast(val))),
      }
      unreachable;
    }

    pub const PropertyType = enum(Api.c.OrtPropertyType) {
      i64 = @bitCast(Api.c.OrtIntProperty),
      f32 = @bitCast(Api.c.OrtFloatProperty),
      str = @bitCast(Api.c.OrtStringProperty),
    };

    pub const PropertyUnion = union(PropertyType) {
      i64: *i64,
      f32: *f32,
      str: [*:0]u8,

      pub fn deinit(self: @This(), allocator: *Allocator) void {
        const ptr: [*]u8 = switch (self) {
          .i64 => @ptrCast(self.i64),
          .f32 => @ptrCast(self.f32),
          .str => @ptrCast(self.str),
        };
        allocator.free(ptr);
      }
    };

    /// Retrieves the type and shape information of the parameter associated with the given parameter name.
    ///
    /// This function retrieves the type and shape of the parameter associated with the given parameter name.
    /// The parameter must exist in the checkpoint state to be able to retrieve its type and shape information successfully.
    ///
    /// parameter_name: Name of the parameter being retrieved.
    pub fn getParameterTypeAndShape(self: *const @This(), name: [*:0]const u8) !*TensorTypeAndShapeInfo.C {
      var out: ?*TensorTypeAndShapeInfo.C = null;
      try Error.check(api.underlying.GetParameterTypeAndShape.?(apiCast(self), cStr(name), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Updates the data associated with the model parameter in the checkpoint state for the given parameter name.
    ///
    /// This function updates a model parameter in the checkpoint state with the given parameter data.
    /// The training session must be already created with the checkpoint state that contains the parameter
    /// being updated. The given parameter is copied over to the registered device for the training session.
    /// The parameter must exist in the checkpoint state to be able to update it successfully.
    ///
    /// parameter_name: Name of the parameter being updated.
    /// parameter: The parameter data that should replace the existing parameter data.
    pub fn updateParameter(self: *@This(), name: [*:0]const u8, param: *Value) !void {
      try Error.check(api.underlying.UpdateParameter.?(apiCast(self), cStr(name), apiCast(param)));
    }

    /// Gets the data associated with the model parameter from the checkpoint state for the given parameter name.
    ///
    /// This function retrieves the model parameter data from the checkpoint state for the given parameter name.
    /// The parameter is copied over and returned as an OrtValue. The training session must be already created
    /// with the checkpoint state that contains the parameter being retrieved.
    /// The parameter must exist in the checkpoint state to be able to retrieve it successfully.
    ///
    /// parameter_name: Name of the parameter being retrieved.
    /// allocator: Allocator used to allocate the memory for the parameter.
    pub fn getParameter(self: *const @This(), name: [*:0]const u8, allocator: *Allocator) !*Value {
      var out: ?*Value = null;
      try Error.check(api.underlying.GetParameter.?(apiCast(self), cStr(name), apiCast(allocator), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn deinit(self: *@This()) void {
      api.underlying.ReleaseCheckpointState.?(apiCast(self));
    }
  };

  pub const TrainingSession = opaque {
    pub const Underlying = Api.c.OrtTrainingSession;
    /// Create a training session that can be used to begin or resume training.
    ///
    /// This function creates a training session based on the env and session options provided that can
    /// begin or resume training from a given checkpoint state for the given onnx models.
    /// The checkpoint state represents the parameters of the training session which will be moved
    /// to the device specified by the user through the session options (if necessary).
    /// The training session requires four training artifacts
    /// - The training onnx model
    /// - The evaluation onnx model (optional)
    /// - The optimizer onnx model
    /// - The checkpoint file
    ///
    /// These artifacts can be generated using the `onnxruntime-training` python [utility](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md).
    ///
    /// env Environment: to be used for the training session.
    /// options Session: options that the user can customize for this training session.
    /// checkpoint_state: Training states that the training session uses as a starting point for training.
    /// train_model_path: Model to be used to perform training.
    /// eval_model_path: Model to be used to perform evaluation.
    /// optimizer_model_path: Model to be used to perform gradient descent.
    /// returns the Created training session.
    pub fn init(
      options: *const Session.Options.C,
      checkpoint: *CheckpointState,
      train_model_path: Utils.Path,
      eval_model_path: Utils.Path,
      optimizer_model_path: Utils.Path,
    ) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.CreateTrainingSession.?(
        Api.env.underlying,
        apiCast(options),
        apiCast(checkpoint),
        pathCast(train_model_path),
        pathCast(eval_model_path),
        pathCast(optimizer_model_path),
        apiCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Create a training session that can be used to begin or resume training.
    /// This api provides a way to load all the training artifacts from buffers instead of files.
    ///
    /// options Session: options that the user can customize for this training session.
    /// checkpoint_state: Training states that the training session uses as a starting point for training.
    /// train_model_data: Buffer containing the model data to be used to perform training
    /// eval_model_data: Buffer containing the model data to be used to perform evaluation
    /// optim_model_data: Buffer containing the model data to be used to perform weight update
    pub fn initBuffer(
      options: *const Session.Options.C,
      checkpoint: *CheckpointState,
      train_model_data: []const u8,
      eval_model_data: []const u8,
      optimizer_model_data: []const u8,
    ) !*@This() {
      var out: ?*@This() = null;
      try Error.check(api.underlying.CreateTrainingSessionFromBuffer.?(
        Api.env.underlying,
        apiCast(options),
        apiCast(checkpoint),
        cCast(train_model_data.ptr),
        train_model_data.len,
        cCast(if (eval_model_data.len > 0) eval_model_data.ptr else null),
        eval_model_data.len,
        cCast(optimizer_model_data.ptr),
        optimizer_model_data.len,
        apiCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Retrieves the number of user outputs in the training model.
    ///
    /// This function returns the number of outputs of the training model so that the user can
    /// allocate space for the number of outputs when OrtTrainingApi::TrainStep is invoked.
    pub fn getTrainingModelOutputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(api.underlying.TrainingSessionGetTrainingModelOutputCount.?(apiCast(self), &out));
      return out;
    }

    /// Retrieves the number of user outputs in the eval model.
    ///
    /// This function returns the number of outputs of the eval model so that the user can
    /// allocate space for the number of outputs when OrtTrainingApi::EvalStep is invoked.
    pub fn getEvalModelOutputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(api.underlying.TrainingSessionGetEvalModelOutputCount.?(apiCast(self), &out));
      return out;
    }

    /// Retrieves the names of user outputs in the training model.
    ///
    /// This function returns the names of outputs of the training model that can be associated with the OrtValue(s)
    /// returned by the OrtTrainingApi::TrainStep function.
    ///
    /// index: Index of the output name requested.
    /// allocator: Allocator to use to allocate the memory for the name.
    pub fn getTrainingModelOutputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(api.underlying.TrainingSessionGetTrainingModelOutputName.?(apiCast(self), index, apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Retrieves the names of user outputs in the eval model.
    ///
    /// This function returns the names of outputs of the eval model that can be associated with the OrtValue(s) returned
    /// by the OrtTrainingApi::EvalStep function.
    ///
    /// index: Index of the output name requested.
    /// allocator: Allocator to use to allocate the memory for the name.
    pub fn getEvalModelOutputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(api.underlying.TrainingSessionGetEvalModelOutputName.?(apiCast(self), index, apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    ///Reset the gradients of all trainable parameters to zero lazily.
    ///
    /// This function sets the internal state of the training session such that the gradients of the trainable
    /// parameters in the OrtCheckpointState will be scheduled to be reset just before the new gradients are
    /// computed on the next invocation of the next OrtTrainingApi::TrainStep.
    ///
    /// Reset the gradients of all trainable parameters to zero lazily.
    pub fn lazyResetGrad(self: *@This()) !void {
      try Error.check(api.underlying.LazyResetGrad.?(apiCast(self)));
    }

    ///Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs
    ///
    /// This function performs a training step that computes the outputs of the training model and the gradients
    /// of the trainable parameters for the given inputs. The train step is performed based on the training model
    /// that was provided to the training session.
    /// The OrtTrainingApi::TrainStep is equivalent of running forward propagation and backward propagation in a single step.
    /// The gradients computed are stored inside the training session state so they can be later consumed
    /// by the OrtTrainingApi::OptimizerStep function.
    /// The gradients can be lazily reset by invoking the OrtTrainingApi::LazyResetGrad function.
    ///
    /// run_options: Run options for this training step.
    /// inputs: The user inputs to the training model.
    /// [out] outputs: User outputs computed by train step.
    pub fn trainStep(
      self: *@This(),
      run_options: ?*const RunOptions,
      inputs: []const *const Value,
      outputs: []?*Value,
    ) !void {
      try Error.check(api.underlying.TrainStep.?(
        apiCast(self),
        apiCast(run_options),
        inputs.len,
        apiCast(inputs.ptr),
        outputs.len,
        apiCast(outputs.ptr),
      ));
    }

    /// Computes the outputs for the eval model for the given inputs.
    /// Computes the outputs for the eval model for the given inputs
    ///
    /// This function performs an eval step that computes the outputs of the eval model for the given inputs.
    /// The eval step is performed based on the eval model that was provided to the training session.
    ///
    /// run_options: Run options for this eval step.
    /// inputs: The user inputs to the eval model.
    /// [out] outputs: User outputs computed by eval step.
    pub fn evalStep(
      self: *@This(),
      run_options: ?*const RunOptions,
      inputs: []const *const Value,
      outputs: []?*Value,
    ) !void {
      try Error.check(api.underlying.EvalStep.?(
        apiCast(self),
        apiCast(run_options),
        inputs.len,
        apiCast(inputs.ptr),
        outputs.len,
        apiCast(outputs.ptr),
      ));
    }

    /// Sets the learning rate for this training session.
    ///
    /// This function allows users to set the learning rate for the training session. The current
    /// learning rate is maintained by the training session and can be overwritten by invoking
    /// this function with the desired learning rate. This function should not be used when a valid
    /// learning rate scheduler is registered. It should be used either to set the learning rate
    /// derived from a custom learning rate scheduler or to set a constant learning rate to be used
    /// throughout the training session.
    /// Note: This function does not set the initial learning rate that may be needed
    /// by the predefined learning rate schedulers. To set the initial learning rate for learning
    /// rate schedulers, please look at the function OrtTrainingApi::RegisterLinearLRScheduler.
    ///
    /// learning_rate: Desired learning rate to be set.
    pub fn setLearningRate(self: *@This(), learning_rate: f32) !void {
      try Error.check(api.underlying.SetLearningRate.?(apiCast(self), learning_rate));
    }

    /// Gets the current learning rate for this training session.
    ///
    /// This function allows users to get the learning rate for the training session. The current
    /// learning rate is maintained by the training session, and users can query it for the purpose
    /// of implementing their own learning rate schedulers.
    pub fn getLearningRate(self: *@This()) !f32 {
      var out: f32 = undefined;
      try Error.check(api.underlying.GetLearningRate.?(apiCast(self), &out));
      return out;
    }

    /// Performs the weight updates for the trainable parameters using the optimizer model.
    ///
    /// This function performs the weight update step that updates the trainable parameters such that they
    /// take a step in the direction of their gradients (gradient descent). The optimizer step is performed
    /// based on the optimizer model that was provided to the training session.
    /// The updated parameters are stored inside the training state so that they can be used by the next
    /// OrtTrainingApi::TrainStep function call.
    ///
    /// run_options: Run options for this optimizer step.
    pub fn optimizerStep(self: *@This(), run_options: ?*const RunOptions) !void {
      try Error.check(api.underlying.OptimizerStep.?(apiCast(self), apiCast(run_options)));
    }

    /// Registers a linear learning rate scheduler for the training session.
    ///
    /// Register a linear learning rate scheduler that decays the learning rate by linearly updated
    /// multiplicative factor from the initial learning rate set on the training session to 0. The decay
    /// is performed after the initial warm up phase where the learning rate is linearly incremented
    /// from 0 to the initial learning rate provided.
    ///
    /// warmup_step_count Warmup steps for LR warmup.
    /// total_step_count Total step count.
    /// initial_lr The initial learning rate to be used by the training session.
    pub fn registerLinearLRScheduler(self: *@This(), warmup_step_count: i64, total_step_count: i64, initial_lr: f32) !void {
      try Error.check(api.underlying.RegisterLinearLRScheduler.?(
        apiCast(self),
        warmup_step_count,
        total_step_count,
        initial_lr
      ));
    }

    /// Update the learning rate based on the registered learning rate scheduler.
    ///
    /// Takes a scheduler step that updates the learning rate that is being used by the training session.
    /// This function should typically be called before invoking the optimizer step for each round,
    /// or as determined necessary to update the learning rate being used by the training session.
    /// Note: Please note that a valid predefined learning rate scheduler must be first registered to invoke this function.
    pub fn schedulerStep(self: *@This()) !void {
      try Error.check(api.underlying.SchedulerStep.?(apiCast(self)));
    }

    /// Retrieves the size of all the parameters.
    ///
    /// Calculates the total number of primitive (datatype of the parameters) elements of all the parameters in the training state.
    /// When trainable_only argument is true, the size is calculated for trainable params only.
    ///
    /// trainable_only: Whether to skip non-trainable parameters
    pub fn getParametersSize(self: *@This(), trainable_only: bool) !usize {
      var out: usize = undefined;
      try Error.check(api.underlying.GetParametersSize.?(apiCast(self), &out, trainable_only));
      return out;
    }


    /// Copy all parameters to a contiguous buffer held by the argument parameters_buffer
    ///
    /// The parameters_buffer has to be of the size given by GetParametersSize api call,
    /// with matching setting for the argument trainable_only. All the target parameters must be of the same
    /// datatype. The OrtValue must be pre-allocated onto
    /// the desired device. This is a complementary function to OrtTrainingApi::CopyBufferToParameters.
    /// Parameter ordering is preserved.
    /// User is responsible for allocating and freeing the resources used by the parameters_buffer.
    ///
    /// [out] out_buffer The pre-allocated OrtValue buffer to copy onto.
    /// trainable_only Whether to skip non-trainable parameters
    pub fn copyParametersToBuffer(self: *@This(), out_buffer: *Value, trainable_only: bool) !void {
      try Error.check(api.underlying.CopyParametersToBuffer.?(apiCast(self), apiCast(out_buffer), trainable_only));
    }

    /// Copy parameter values from the given contiguous buffer held by parameters_buffer to the training state
    ///
    /// The parameters_buffer argument has to be of the size given by OrtTrainingApi::GetParametersSize api call,
    /// with matching setting for trainable_only argument. All the target parameters must be of the same
    /// datatype. This is a complementary function to OrtTrainingApi::CopyParametersToBuffer
    /// and can be used to load updated buffer values onto the training state.
    /// Parameter ordering is preserved.
    /// User is responsible for allocating and freeing the resources used by the parameters_buffer.
    /// In case the training session was created with a nominal checkpoint, invoking this function is required
    /// to load the updated parameters onto the checkpoint to complete it.
    ///
    /// trainable_only Whether to skip non-trainable parameters
    pub fn copyBufferToParameters(self: *@This(), out_buffer: *Value, trainable_only: bool) !void {
      try Error.check(api.underlying.CopyBufferToParameters.?(apiCast(self), apiCast(out_buffer), trainable_only));
    }

    /// Export a model that can be used for inferencing.
    ///
    /// If the training session was provided with an eval model, the training session can generate
    /// an inference model if it knows the inference graph outputs. The input inference graph outputs
    /// are used to prune the eval model so that the inference model's outputs align with the provided outputs.
    /// The exported model is saved at the path provided and can be used for inferencing with InferenceSession.
    /// Note: the function re-loads the eval model from the path provided to OrtTrainingApi::CreateTrainingSession and expects that this path still be valid.
    ///
    /// inference_model_path: Path where the inference model should be serialized to.
    /// graph_output_names Names of the outputs that are needed in the inference model.
    pub fn exportModelForInferencing(self: *@This(), path: Utils.Path, graph_output_names: []const [*:0]const u8) !void {
      try Error.check(api.underlying.ExportModelForInferencing.?(
        apiCast(self),
        pathCast(path),
        graph_output_names.len,
        cStr(graph_output_names.ptr)
      ));
    }

    /// Retrieves the number of user inputs in the training model.
    ///
    /// This function returns the number of inputs of the training model so that the user can accordingly
    /// allocate the OrtValue(s) provided to the OrtTrainingApi::TrainStep function.
    pub fn getTrainingModelInputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(api.underlying.TrainingSessionGetTrainingModelInputCount.?(apiCast(self), &out));
      return out;
    }

    /// Retrieves the number of user inputs in the eval model.
    ///
    /// This function returns the number of inputs of the eval model so that the user can accordingly
    /// allocate the OrtValue(s) provided to the OrtTrainingApi::EvalStep function.
    pub fn getEvalModelInputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(api.underlying.TrainingSessionGetEvalModelInputCount.?(apiCast(self), &out));
      return out;
    }

    /// Retrieves the name of the user input at given index in the training model.
    ///
    /// This function returns the names of inputs of the training model that can be associated with the
    /// OrtValue(s) provided to the OrtTrainingApi::TrainStep function.
    ///
    /// index: The index of the training model input name requested.
    /// allocator: The allocator to use to allocate the memory for the requested name.
    pub fn getTrainingModelInputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(api.underlying.TrainingSessionGetTrainingModelInputName.?(apiCast(self), index, apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Retrieves the name of the user input at given index in the eval model.
    ///
    /// This function returns the names of inputs of the eval model that can be associated with the OrtValue(s) provided
    /// to the OrtTrainingApi::EvalStep function.
    ///
    /// index: The index of the eval model input name requested.
    /// allocator: The allocator to use to allocate the memory for the requested name.
    pub fn getEvalModelInputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(api.underlying.TrainingSessionGetEvalModelInputName.?(apiCast(self), index, apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn deinit(self: *@This()) void {
      api.underlying.ReleaseTrainingSession.?(apiCast(self));
    }
  };
};

/// The main api struct
pub const Api = struct {
  /// API docs: https://onnxruntime.ai/docs/api/c/struct_Api.ort.html
  pub const c = @cImport({ @cInclude("onnxruntime_training_c_api.h"); });

  pub var base: *const Api.c.OrtApiBase = undefined;
  /// a pointer to a static api struct
  pub var ort: *const Api.c.OrtApi = undefined;
  /// a pointer to a static version string
  pub var version_string: [:0]const u8 = undefined;

  /// the pointer to the Api.env used for logging n stuff.
  /// This is here because the Api.env instance is global so no point making it non_static
  /// The Env holds the logging state used by all other objects.
  pub const env = opaque {
    pub var underlying: *c.OrtEnv = undefined;

    /// Wraps OrtApi::CreateEnv, OrtApi::CreateEnvWithGlobalThreadPools, OrtApi::CreateEnvWithCustomLogger and OrtApi::CreateEnvWithCustomLoggerAndGlobalThreadPools
    /// The correct function is called depending on the provided options
    pub fn init(
      logging_level: Logging.Level,
      logid: [*:0]const u8,
      logging_interface: ?Logging.Interface,
      threading_options: ?ThreadingOptions,
    ) !void {
      var self: ?*c.OrtEnv = null;
      if (logging_interface == null and threading_options == null) {
        try Error.check(Api.ort.CreateEnv.?(@intFromEnum(logging_level), cStr(logid), &self));
      } else if (logging_interface == null) {
        const to = try threading_options.?.c();
        defer to.deinit();
        try Error.check(Api.ort.CreateEnvWithGlobalThreadPools.?(
            @intFromEnum(logging_level),
            cStr(logid),
            apiCast(to),
            &self,
        ));
      } else if (threading_options == null) {
        try Error.check(Api.ort.CreateEnvWithCustomLogger.?(
            logging_interface.?.log_fn,
            logging_interface.?.ptr,
            @intFromEnum(logging_level),
            cStr(logid),
            &self,
        ));
      } else {
        const to = try threading_options.?.c();
        defer to.deinit();
        try Error.check(Api.ort.CreateEnvWithCustomLoggerAndGlobalThreadPools.?(
            logging_interface.?.log_fn,
            logging_interface.?.ptr,
            @intFromEnum(logging_level),
            cStr(logid),
            apiCast(to),
            &self,
        ));
      }
      underlying = self orelse return error.OutOfMemory;
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
    pub const LanguageProjection = enum(Api.c.OrtLanguageProjection) {
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
    pub fn registerExecutionProviderLibrary(registration_name: [*:0]const u8, path: Utils.Path) !void {
      try Error.check(ort.RegisterExecutionProviderLibrary.?(underlying, cStr(registration_name), pathCast(path)));
    }

    /// Wraps OrtApi::UnregisterExecutionProviderLibrary
    pub fn unregisterExecutionProviderLibrary(registration_name: [*:0]const u8) !void {
      try Error.check(ort.UnregisterExecutionProviderLibrary.?(underlying, cStr(registration_name)));
    }

    /// Wraps OrtApi::GetEpDevices
    pub fn getEpDevices() ![]const *const Ep.Device {
      var retval_ptr: ?[*]const *const Ep.Device = null;
      var retval_len: usize = 0;
      try Error.check(ort.GetEpDevices.?(underlying, apiCast(&retval_ptr), &retval_len));
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
          apiCast(ep_device),
          @intFromEnum(mem_type),
          @intFromEnum(allocator_type),
          apiCast(options),
          apiCast(&retval),
      ));
      return retval orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::ReleaseSharedAllocator
    /// Release a shared allocator from the OrtEnv for the OrtEpDevice and memory type.
    /// If no shared allocator exists, this is a no-op.
    pub fn releaseSharedAllocator(ep_device: *const Ep.Device, mem_type: Allocator.DeviceMemoryType) !void {
      try Error.check(Api.ort.ReleaseSharedAllocator.?(underlying, apiCast(ep_device), @intFromEnum(mem_type)));
    }

    /// Wraps OrtApi::CopyTensors
    /// Copy OrtValue instances containing Tensors between devices.
    /// src and dst must be the same length.
    /// stream: Optional SyncStream for async copy.
    pub fn copyTensors(src: []const *const Value, dst: []const *Value, stream: ?*Ep.SyncStream) !void {
      std.debug.assert(src.len == dst.len);
      try Error.check(Api.ort.CopyTensors.?(underlying, apiCast(src.ptr), apiCast(dst.ptr), apiCast(stream), src.len));
    }

    /// Release the Env object.
    pub fn deinit() void {
      Api.ort.ReleaseEnv.?(underlying);
    }
  };

  /// The model editor api
  pub const editor = opaque {
    pub var underlying: *const c.OrtModelEditorApi = undefined;

    /// Create an OrtValueInfo for use as an OrtGraph input or output.
    pub fn createValueInfo(name: [*:0]const u8, type_info: *const TypeInfo) !*Value.Info {
      var out: ?*Value.Info = null;
      try Error.check(underlying.CreateValueInfo.?(cStr(name), apiCast(type_info), apiCast(&out)));
      return out orelse return error.OutOfMemory;
    }

    /// Create an OrtSession using the OrtModel.
    /// The OrtModel must have a Graph with inputs/outputs set.
    pub fn createSessionFromModel(model: *const Model, options: *const Session.Options.C) !*Session {
      var out: ?*Session = null;
      try Error.check(underlying.CreateSessionFromModel.?(env.underlying, apiCast(model), apiCast(options), apiCast(&out)));
      return out orelse return error.OutOfMemory;
    }

    /// Create an OrtSession to augment an existing model (from path).
    pub fn createModelEditorSession(model_path: Utils.Path, options: *const Session.Options.C) !*Session {
      var out: ?*Session = null;
      try Error.check(underlying.CreateModelEditorSession.?(env.underlying, pathCast(model_path), apiCast(options), apiCast(&out)));
      return out orelse return error.OutOfMemory;
    }

    /// Create an OrtSession to augment an existing model (from memory).
    pub fn createModelEditorSessionFromArray(data: []const u8, options: *const Session.Options.C) !*Session {
      var out: ?*Session = null;
      try Error.check(underlying.CreateModelEditorSessionFromArray.?(
        env.underlying, 
        data.ptr, 
        data.len, 
        apiCast(options), 
        apiCast(&out)
      ));
      return out orelse return error.OutOfMemory;
    }

    /// Query the session for the opset version of a domain.
    pub fn sessionGetOpsetForDomain(session: *const Session, domain: [*:0]const u8) !c_int {
      var opset: c_int = 0;
      try Error.check(underlying.SessionGetOpsetForDomain.?(apiCast(session), domain, &opset));
      return opset;
    }

    /// Apply changes to augment the ONNX model in a session.
    pub fn applyModelToModelEditorSession(session: *Session, model: *Model) !void {
      try Error.check(underlying.ApplyModelToModelEditorSession.?(apiCast(session), apiCast(model)));
    }

    /// Finalize the Model Editor session.
    pub fn finalizeModelEditorSession(session: *Session, options: *const Session.Options.C, prepacked: ?*PrepackedWeightsContainer) !void {
      try Error.check(underlying.FinalizeModelEditorSession.?(apiCast(session), apiCast(options), apiCast(prepacked)));
    }
  };

  /// The compiler api
  pub const compiler = opaque {
    pub var underlying: *const c.OrtCompileApi = undefined;

    /// Translated from OrtApi::OrtCompileApiFlags
    pub const Flags = packed struct {
      ErrorIfNoNodesCompiled: bool = false,
      ErrorIfOutputFileExists: bool = false,
      _padding: u30 = 0,
    };

    comptime {
      std.debug.assert(@bitSizeOf(Flags) == 32);
    }

    pub const CompiledModelCompatibility = enum(Api.c.OrtCompiledModelCompatibility) {
      NOT_APPLICABLE = @bitCast(Api.c.OrtCompiledModelCompatibility_EP_NOT_APPLICABLE),
      SUPPORTED_OPTIMAL = @bitCast(Api.c.OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL),
      SUPPORTED_PREFER_RECOMPILATION = @bitCast(Api.c.OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION),
      UNSUPPORTED = @bitCast(Api.c.OrtCompiledModelCompatibility_EP_UNSUPPORTED),
    };
  };

  const Options = struct {
    /// The logging level for messages generated.
    log_level: Logging.Level,
    /// This must outlive `this` and any instances created by `this` OnnxInstanceCreator.
    log_id: [*:0]const u8,
    /// The error logging function to use. This is called every time an error is occurs.
    error_log_fn: @TypeOf(Logging.errorLogFn) = null,
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
    Logging.errorLogFn = options.error_log_fn;
    base = Api.c.OrtGetApiBase();
    ort = base.GetApi.?(c.ORT_API_VERSION) orelse return error.ApiVersionMismatch;
    version_string = std.mem.sliceTo(cStrTo(base.GetVersionString.?(), ?[*:0] const u8) orelse @as([*:0] const u8, @ptrCast(&empty_string)), 0);

    editor.underlying = if (options.editor) ort.GetModelEditorApi.?() else ComptimeOptions.impl(comptime_options.editor_behavior, c.OrtModelEditorApi);
    compiler.underlying = if (options.compiler) ort.GetCompileApi.?() else ComptimeOptions.impl(comptime_options.compile_behavior, c.OrtCompileApi);
    Ep.api.underlying = if (options.ep) ort.GetEpApi.?() else ComptimeOptions.impl(comptime_options.ep_behavior, c.OrtEpApi);
    const training_fallback = ComptimeOptions.impl(comptime_options.training_behavior, c.OrtTrainingApi);
    Training.api.underlying = if (options.training) (ort.GetTrainingApi.?(c.ORT_API_VERSION) orelse return error.TrainingApiNotAvailable) else training_fallback;

    try Api.env.init(options.log_level, options.log_id, options.logging_interface, options.threading_options);
    errdefer Api.env.deinit();

    Error.Status.oom = try .init(@intFromEnum(Error.Code.RuntimeException), "Out of memory");
    errdefer Error.Status.oom.deinit();
    Error.Status.released = false;

    if (options.telemetry_events) |state| try env.setTelemetryEventsState(state);
  }

  /// Frees the resources created by the `Api.init` function
  pub fn deinit() void {
    // `Api.ort` and `version_string` are static so only env needs to be deinited
    Api.env.deinit();

    // Releasing the static oom error status
    Error.Status.oom.deinit();
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
      try Error.check(ort.ReleaseAvailableProviders.?(cStr(self.providers.ptr), @intCast(self.providers.len)));
    }
  };

  /// Returns a list of available execution providers. The caller must free the returned value using deinit
  /// Wraps OrtApi::GetAvailableProviders
  pub fn getAvailableProviders() !Providers {
    var ptrs: ?[*][*:0]u8 = null;
    var len: c_int = 0;
    try Error.check(Api.ort.GetAvailableProviders.?(cStr(&ptrs), &len));
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
    try Error.check(Api.ort.GetExecutionProviderApi.?(cStr(provider_name), version, &ptr));
    return ptr orelse error.NotFound;
  }

  /// Validate a compiled model's compatibility information for one or more EP devices.
  pub fn getModelCompatibilityForEpDevices(ep_devices: []const *const Ep.Device, compatibility_info: [*:0]const u8) !compiler.CompiledModelCompatibility {
    var out: compiler.CompiledModelCompatibility = undefined;
    try Error.check(Api.ort.GetModelCompatibilityForEpDevices.?(
        apiCast(ep_devices.ptr),
        ep_devices.len,
        cStr(compatibility_info),
        apiCast(&out)
    ));
    return out;
  }
};

/// Logging stuff and interface
pub const Logging = struct {
  /// Levels of logging verbosity, from least severe (verbose) to most severe (fatal).
  pub const Level = enum(Api.c.OrtLoggingLevel) {
    debug = @bitCast(Api.c.ORT_LOGGING_LEVEL_VERBOSE),
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
    ///   severity: Logging.Level, category: ?[*:0]const u8, logid: ?[*:0]const u8, code_location: ?[*:0]const u8, messages: ?[*:0]const u8
    ///
    /// if (@bitSizeOf(@TypeOf(context)) != 0) then `context` must outlive the instance creator and any created instances
    pub fn fromContext(context: anytype) @This() {
      const T = @TypeOf(context);
      const Sub = blk: {
        const Temp = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
        break :blk if (@typeInfo(Temp) == .pointer) @typeInfo(Temp).pointer.child else Temp;
      };
      return .{
        .ptr = if (@bitSizeOf(Sub) == 0) null else @ptrCast(context),
        .log_fn = &struct {
          fn log_fn(
            ctx: ?*anyopaque,
            severity: Api.c.OrtLoggingLevel,
            category: [*c]const u8,
            logid: [*c]const u8,
            code_location: [*c]const u8,
            messages: [*c]const u8
          ) callconv(.c) void {
            const original: *Sub = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @alignCast(@ptrCast(ctx.?));
            original.log(@as(Logging.Level, @enumFromInt(severity)), cStrTo(category, ?[*:0]const u8), cStrTo(logid, ?[*:0]const u8), cStrTo(code_location, ?[*:0]const u8), cStrTo(messages, ?[*:0]const u8));
          }
        }.log_fn,
      };
    }
  };

  pub var errorLogFn: ?*const fn(*Error.Status) void = null;
};

/// Error handling things
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
  pub const Code = enum(Api.c.OrtErrorCode) {
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
    pub const Underlying = Api.c.OrtStatus;
    /// create a new status, this function uses c's allocator
    pub fn init(code: c_uint, msg: [*:0]const u8) !*@This() {
      return apiCastTo(Api.ort.CreateStatus.?(code, msg) orelse return error.OutOfMemory, *@This());
    }

    /// get the error messages from the status
    pub fn getErrorMessage(self: *const @This()) [*:0]const u8 {
      return cStrTo(Api.ort.GetErrorMessage.?(apiCast(self)), [*:0]const u8);
    }

    /// Get the error code from the status
    pub fn getErrorCode(self: *const @This()) c_uint {
      return Api.ort.GetErrorCode.?(apiCast(self));
    }

    var oom: *Status = undefined;
    var released: bool = true;

    /// release the status
    pub fn deinit(self: *@This()) void {
      if (@intFromPtr(self) == @intFromPtr(oom)) {
        released = false;
      } else {
        Api.ort.ReleaseStatus.?(apiCast(self));
      }
    }
    
    pub fn initInfallible(code: c_uint, msg: [*:0]const u8) *@This() {
      return init(code, msg) catch {
        // This may only happen if the status is released outside of this wrapper and we oom twice but is still possible
        if (released) @panic("OOM");
        released = true;
        return oom;
      };
    }
  };

  /// A simple error checking function.
  /// This function is a no-op if onnx_status is null
  /// If `onnx_status` is NOT null, this function
  ///   calls the on_error_fn and returns it's error or returns `OnnxError` if it is null
  pub fn check(ort_status: ?*Api.c.OrtStatus) Set!void {
    if (apiCastTo(ort_status, ?*Status)) |status| {
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

      if (Logging.errorLogFn) |f| f(status);
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

/// Key-Value Pairs, used in a lot of things in this librarym but mainly for options k-v pairs
pub const KeyValuePairs = opaque {
  pub const Underlying = Api.c.OrtKeyValuePairs;
  /// Wraps OrtApi::CreateKeyValuePairs
  pub fn init() !*@This() {
    var retval: ?*@This() = null;
    Api.ort.CreateKeyValuePairs.?(apiCast(&retval));
    return retval orelse error.OutOfMemory;
  }

  /// Wraps OrtApi::AddKeyValuePair.
  pub fn add(self: *@This(), key: [*:0]const u8, value: [*:0]const u8) void {
    // Since AddKeyValuePair returns nothing, we don't know if it errored; hence this function returns no error
    // We technically could use `get` but no need for another virtual call if library authors thought this was not an issue
    Api.ort.AddKeyValuePair.?(apiCast(self), key, value);
  }

  /// Wraps OrtApi::GetKeyValue
  pub fn get(self: *const @This(), key: [*:0]const u8) ?[*:0]const u8 {
    return cStrTo(Api.ort.GetKeyValue.?(apiCast(self), key), ?[*:0]const u8);
  }

  /// Wraps OrtApi::RemoveKeyValuePair
  pub fn remove(self: *@This(), key: [*:0]const u8) void {
    Api.ort.RemoveKeyValuePair.?(apiCast(self), key);
  }

  /// Wraps OrtApi::GetKeyValuePairs
  /// Returns slices of keys and values. These pointers are valid as long as the KeyValuePairs object is valid. No need to free these
  pub fn getKeyValues(self: *const @This()) std.meta.Tuple(&.{ []const [*:0]const u8, []const [*:0]const u8 }) {
    const mt: []const [*:0]const u8 = &[_][*:0]const u8{};
    var keys_ptr = mt.ptr;
    var values_ptr = mt.ptr;
    var count: usize = 0;
    Api.ort.GetKeyValuePairs.?(apiCast(self), cStr(&keys_ptr), cStr(&values_ptr), &count);
    return .{ keys_ptr[0..count], values_ptr[0..count] };
  }

  /// Wraps OrtApi::ReleaseKeyValuePairs
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseKeyValuePairs.?(apiCast(self));
  }
};

/// Structure of function pointers that defines a memory allocator.
/// Used for both internal and custom user-defined memory management.
pub const Allocator = struct {
  pub const Underlying = Api.c.OrtAllocator;
  /// The underlying C allocator pointer.
  underlying: Underlying,
  comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(Underlying)); }

  /// Types of memory allocators (Device-local, Arena-wrapped, or Read-only).
  pub const Type = enum(Api.c.OrtAllocatorType) {
    invalid = @bitCast(Api.c.OrtInvalidAllocator),
    device = @bitCast(Api.c.OrtDeviceAllocator),
    arena = @bitCast(Api.c.OrtArenaAllocator),
    readonly = @bitCast(Api.c.OrtReadOnlyAllocator),
  };

  /// This matches OrtDevice::MemoryType values
  /// Specific memory access traits for devices (Default vs Host Accessible).
  pub const DeviceMemoryType = enum(Api.c.OrtDeviceMemoryType) {
    DEFAULT = @bitCast(Api.c.OrtDeviceMemoryType_DEFAULT),
    HOST_ACCESSIBLE = @bitCast(Api.c.OrtDeviceMemoryType_HOST_ACCESSIBLE),
  };

  /// Configuration for an arena-based allocator.
  /// Defines maximum memory limits and growth strategies.
  pub const ArenaCfg = opaque {
    pub const Underlying = Api.c.OrtArenaCfg;
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
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateArenaCfg.?(
        options.max_mem,
        @intFromEnum(options.extend_strategy),
        options.initial_chunk,
        options.max_dead,
        apiCast(&self)
      ));
      return self orelse error.OutOfMemory;
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
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateArenaCfgV2.?(created.keys(), created.vals(), created.len, apiCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Release the ArenaCfg object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseArenaCfg.?(apiCast(self));
    }
  };

  /// Memory types for allocated memory, execution provider specific types should be extended in each provider.
  pub const MemoryType = enum(Api.c.OrtMemType) {
    /// The default allocator for execution provider
    default = @bitCast(Api.c.OrtMemTypeDefault),
    /// Any CPU memory used by non-CPU execution provider
    cpu_input = @bitCast(Api.c.OrtMemTypeCPUInput),
    /// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    cpu_output = @bitCast(Api.c.OrtMemTypeCPUOutput),
  };

  pub const MemoryDevice = opaque {
    pub const Underlying = Api.c.OrtMemoryDevice;
    /// This mimics OrtDevice type constants so they can be returned in the API
    pub const Type = enum(Api.c.OrtMemoryInfoDeviceType) {
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
      return Ep.api.underlying.MemoryDevice_AreEqual.?(apiCast(self), apiCast(other));
    }

    /// Get the OrtMemoryInfoDeviceType value from an OrtMemoryDevice instance.
    ///
    /// memory_device: OrtMemoryDevice instance.
    /// returns The OrtMemoryInfoDeviceType value.
    ///
    /// since Version 1.23.
    pub fn getDeviceType(self: *const @This()) MemoryDevice.Type {
      return @enumFromInt(Ep.api.underlying.MemoryDevice_GetDeviceType.?(apiCast(self)));
    }

    /// Get the OrtDeviceMemoryType value from an OrtMemoryDevice instance.
    ///
    /// memory_device: OrtMemoryDevice instance.
    /// returns The OrtDeviceMemoryType value.
    ///
    /// since Version 1.23.
    pub fn getMemoryType(self: *const @This()) Allocator.DeviceMemoryType {
      return @enumFromInt(Ep.api.underlying.MemoryDevice_GetMemoryType.?(apiCast(self)));
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
      return Ep.api.underlying.MemoryDevice_GetVendorId.?(apiCast(self));
    }

    /// Get the device ID from an OrtMemoryDevice instance.
    ///
    /// memory_device: OrtMemoryDevice instance.
    /// returns The device ID.
    ///
    /// since Version 1.23.
    pub fn getDeviceId(self: *const @This()) u32 {
      return Ep.api.underlying.MemoryDevice_GetDeviceId.?(apiCast(self));
    }
  };

  /// Describes the location and traits of a memory allocation (e.g., CPU, GPU device id).
  pub const MemoryInfo = opaque {
    pub const Underlying = Api.c.OrtMemoryInfo;
    /// Create an ::OrtMemoryInfo
    /// name_: Arbitrary name.
    /// type: The allocator type (Device vs Arena).
    /// id_: Device ID.
    /// mem_type: Memory type (Input vs Output vs Default).
    pub fn init(name_: [*:0]const u8, alloc_type: Allocator.Type, id_: i32, mem_type: MemoryType) !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateMemoryInfo.?(cStr(name_), @intFromEnum(alloc_type), id_, @intFromEnum(mem_type), apiCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Create an ::OrtMemoryInfo for CPU memory
    /// Special case version of OrtApi::CreateMemoryInfo for CPU based memory. 
    /// Same as using OrtApi::CreateMemoryInfo with name = "Cpu" and id = 0.
    pub fn initCpu(alloc_type: Allocator.Type, mem_type: MemoryType) !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateCpuMemoryInfo.?(@intFromEnum(alloc_type), @intFromEnum(mem_type), apiCast(&self)));
      return self orelse error.OutOfMemory;
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
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateMemoryInfo_V2.?(
        cStr(options.name),
        @intFromEnum(options.device_type),
        options.vendor_id,
        options.device_id,
        @intFromEnum(options.mem_type),
        options.alignment,
        @intFromEnum(options.allocator_type),
        apiCast(&self)
      ));
      return self orelse error.OutOfMemory;
    }

    /// Compare ::OrtMemoryInfo objects for equality
    /// Compares all settings of each ::OrtMemoryInfo for equality
    pub fn compare(self: *const @This(), other: *const @This()) !bool {
      var out: c_int = undefined;
      try Error.check(Api.ort.CompareMemoryInfo.?(apiCast(self), apiCast(other), &out));
      return out == 0;
    }

    /// Get name from ::OrtMemoryInfo
    /// Do NOT free the returned pointer. It is valid for the lifetime of the ::OrtMemoryInfo
    pub fn name(self: *const @This()) ![*:0]const u8 {
      var out: ?[*:0]const u8 = null;
      try Error.check(Api.ort.MemoryInfoGetName.?(apiCast(self), cStr(&out)));
      return out.?;
    }

    /// Get the device id from ::OrtMemoryInfo
    pub fn id(self: *const @This()) !c_int {
      var out: c_int = 0;
      try Error.check(Api.ort.MemoryInfoGetId.?(apiCast(self), &out));
      return out;
    }

    /// Get the ::OrtMemType from ::OrtMemoryInfo
    pub fn memType(self: *const @This()) !MemoryType {
      var out: MemoryType = undefined;
      try Error.check(Api.ort.MemoryInfoGetMemType.?(apiCast(self), apiCast(&out)));
      return out;
    }

    /// Get the ::Allocator.Type from ::OrtMemoryInfo
    pub fn allocatorType(self: *const @This()) !Type {
      var out: Type = undefined;
      try Error.check(Api.ort.MemoryInfoGetType.?(apiCast(self), apiCast(&out)));
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
      return apiCastTo(Ep.api.underlying.MemoryInfo_GetMemoryDevice.?(apiCast(self)) orelse return error.OutOfMemory, *const Allocator.MemoryDevice);
    }

    /// Get OrtDevice type from MemoryInfo
    pub fn deviceType(self: *const @This()) !MemoryDevice.Type {
      var out: MemoryDevice.Type = undefined;
      Api.ort.MemoryInfoGetDeviceType.?(apiCast(self), apiCast(&out));
      return out;
    }

    /// Get the device memory type from ::OrtMemoryInfo
    pub fn deviceMemType(self: *const @This()) DeviceMemoryType {
      return @enumFromInt(Api.ort.MemoryInfoGetDeviceMemType.?(apiCast(self)));
    }

    /// Get the vendor id from ::OrtMemoryInfo
    pub fn vendorId(self: *const @This()) u32 {
      return Api.ort.MemoryInfoGetVendorId.?(apiCast(self));
    }

    /// Release the MemoryInfo object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseMemoryInfo.?(apiCast(self));
    }

    pub fn unregisterAllocator(self: *const @This()) !void {
      try Error.check(Api.ort.UnregisterAllocator.?(Api.env.underlying, apiCast(self)));
    }

    /// Wraps OrtApi::CreateAndRegisterAllocator
    pub fn createAndRegisterAllocator(self: *const @This(), arena_cfg: ?*const ArenaCfg) !void {
      try Error.check(Api.ort.CreateAndRegisterAllocator.?(Api.env.underlying, apiCast(self), apiCast(arena_cfg)));
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
          apiCast(self),
          apiCast(arena_cfg),
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
      try Error.check(Api.ort.GetSharedAllocator.?(Api.env.underlying, apiCast(self), apiCast(&retval)));
      return retval;
    }
  };

  pub fn version(self: *const @This()) u32 {
    return self.underlying.version;
  }

  pub fn _alloc(self: *@This(), size: usize) ?[]u8 {
    std.debug.assert(size != 0);
    return @as([*]u8, @ptrCast(self.underlying.Alloc.?(apiCast(self), size) orelse return null))[0 .. size];
  }

  /// Allocates a block of memory ofthe specified size.
  /// size: Size in bytes. asserts size is not 0.
  /// returns the pointer to the allocated block.
  pub fn alloc(self: *@This(), comptime T: type, size: usize) ![]T {
    std.debug.assert(@alignOf(T) <= 16);
    std.debug.assert(size != 0);
    const mem = self._alloc(@sizeOf(T) * size) orelse return error.OutOfMemory;
    return @as([*]T, @alignCast(@ptrCast(mem.ptr)))[0 .. size];
  }

  pub fn _free(self: *@This(), p: ?[*]u8) void {
    self.underlying.Free.?(apiCast(self), @ptrCast(p));
  }

  /// Frees a block of memory previously allocated with this allocator.
  /// p: Pointer to the memory block.
  pub fn free(self: *@This(), ptr: anytype) void {
    switch (@typeInfo(@TypeOf(ptr))) {
      .optional => if (ptr) |v| self.free(v),
      .pointer => |pi| switch (pi.size) {
        .one, .c, .many => self._free(@ptrCast(ptr)),
        .slice => self._free(@ptrCast(ptr.ptr)),
      },
      else => unreachable,
    }
  }

  /// Return a pointer to an ::OrtMemoryInfo that describes this allocator
  pub fn info(self: *const @This()) *const MemoryInfo {
    return apiCastTo(self.underlying.Info.?(apiCast(self)).?, *const MemoryInfo);
  }

  /// Optional allocation function to use for memory allocations made during session initialization.
  /// Use this function if you want to separate allocations made by ORT during Run() calls from
  /// those made during session initialization. This allows for separate memory management strategies for these
  /// allocations.
  /// size: Size in bytes. asserts size is not 0
  /// returns the pointer to an allocated block of `size` bytes.
  pub fn reserve(self: *@This(), size: usize) ![]u8 {
    std.debug.assert(size != 0);
    return @as([*]u8, @ptrCast((self.underlying.Reserve orelse self.underlying.Alloc.?)(apiCast(self), size) orelse return error.OutOfMemory))[0 .. size];
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
    try Error.check((self.underlying.GetStats orelse return null)(apiCast(self), apiCast(&retval)));
    return retval;
  }

  /// Allocate using a stream.
  /// If the allocator is stream aware this performs allocation using a stream.
  /// Alloc will be used if this is nullptr.
  ///
  /// self: Allocator instance
  /// size: Size of the allocation in bytes. asserts size is not 0.
  /// stream: The stream to allocate on.
  ///
  /// returns the pointer to an allocated block of `size` bytes
  ///
  /// Note: Implementation of this function is optional and AllocOnStream may be set to a nullptr.
  pub fn allocOnStream(self: *@This(), size: usize, stream: *Ep.SyncStream) ![]u8 {
    std.debug.assert(size != 0);
    const ptr = (self.underlying.AllocOnStream orelse return self._alloc(size) orelse error.OutOfMemory)(apiCast(self), size, apiCast(stream));
    return @as([*]u8, @ptrCast(ptr orelse return error.OutOfMemory))[0 .. size];
  }

  pub fn register(self: *@This()) !void {
    try Error.check(Api.ort.RegisterAllocator.?(Api.env.underlying, apiCast(self)));
  }

  pub fn unregister(self: *@This()) !void {
    try self.info().unregisterAllocator();
  }

  pub fn create(session: *const Session, mem_info: *const MemoryInfo) !*Allocator {
    var out: ?*Allocator = null;
    try Error.check(Api.ort.CreateAllocator.?(apiCast(session), apiCast(mem_info), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Wraps GetAllocatorWithDefaultOptions
  pub fn getDefault() !*Allocator {
    var out: ?*Allocator = null;
    try Error.check(Api.ort.GetAllocatorWithDefaultOptions.?(apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// deinitialises the allocator
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseAllocator.?(apiCast(self));
  }
};

/// High-level type information for an OrtValue.
pub const TypeInfo = opaque {
  pub const Underlying = Api.c.OrtTypeInfo;
  pub const Map = opaque {
    pub const Underlying = Api.c.OrtMapTypeInfo;
    pub fn getKeyType(self: *const @This()) !Value.Sub.Tensor.ElementDataType {
      var out: Value.Sub.Tensor.ElementDataType = undefined;
      try Error.check(Api.ort.GetMapKeyType.?(apiCast(self), apiCast(&out)));
      return out;
    }

    pub fn getValueType(self: *const @This()) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.GetMapValueType.?(apiCast(self), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseMapTypeInfo.?(apiCast(self));
    }
  };

  pub const Sequence = opaque {
    pub const Underlying = Api.c.OrtSequenceTypeInfo;
    pub fn getElementType(self: *const @This()) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.GetSequenceElementType.?(apiCast(self), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseSequenceTypeInfo.?(apiCast(self));
    }
  };

  pub const Optional = opaque {
    pub const Underlying = Api.c.OrtOptionalTypeInfo;
    pub fn getContainedType(self: *const @This()) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.GetOptionalContainedTypeInfo.?(apiCast(self), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }
  };

  /// Create an TypeInfo instance for a Tensor. Asserts that the ModelEditor api is initialized
  pub fn forTensor(tensor_info: *const TensorTypeAndShapeInfo.C) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateTensorTypeInfo.?(apiCast(tensor_info), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for a SparseTensor.
  pub fn forSparseTensor(tensor_info: *const TensorTypeAndShapeInfo.C) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateSparseTensorTypeInfo.?(apiCast(tensor_info), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for a Map.
  pub fn forMap(map_key_type: Value.Sub.Tensor.ElementDataType, map_value_type: *const TypeInfo) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateMapTypeInfo.?(@intFromEnum(map_key_type), apiCast(map_value_type), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for a Sequence.
  pub fn forSequence(sequence_type: *const TypeInfo) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateSequenceTypeInfo.?(apiCast(sequence_type), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Create an TypeInfo instance for an Optional.
  pub fn forOptional(contained_type: *const TypeInfo) !*TypeInfo {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateOptionalTypeInfo.?(apiCast(contained_type), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get ::OrtTensorTypeAndShapeInfo from an ::OrtTypeInfo
  /// Do not free the returned value, it is valid until type_info is freed.
  pub fn toTensorTypeAndShapeInfo(self: *const @This()) !?*const TensorTypeAndShapeInfo.C {
    var out: ?*const TensorTypeAndShapeInfo.C = null;
    try Error.check(Api.ort.CastTypeInfoToTensorInfo.?(apiCast(self), apiCast(&out)));
    return out;
  }

  /// Get ::TypeInfo.Type from ::OrtTypeInfo
  pub fn onnxType(self: *const @This()) !Value.Type {
    var out: Value.Type = undefined;
    try Error.check(Api.ort.GetOnnxTypeFromTypeInfo.?(apiCast(self), apiCast(&out)));
    return out;
  }

  pub fn castToMapTypeInfo(self: *const @This()) !?*const Map {
    var out: ?*const Map = null;
    try Error.check(Api.ort.CastTypeInfoToMapTypeInfo.?(apiCast(self), apiCast(&out)));
    return out;
  }

  pub fn castToSequenceTypeInfo(self: *const @This()) !?*const Sequence {
    var out: ?*const Sequence = null;
    try Error.check(Api.ort.CastTypeInfoToSequenceTypeInfo.?(apiCast(self), apiCast(&out)));
    return out;
  }

  pub fn castToOptionalTypeInfo(self: *const @This()) !?*const Optional {
    var out: ?*const Optional = null;
    try Error.check(Api.ort.CastTypeInfoToOptionalTypeInfo.?(apiCast(self), apiCast(&out)));
    return out;
  }

  pub fn getDenotation(self: *const @This()) ![]const u8 {
    var denotation: ?[*:0]const u8 = null;
    var len: usize = 0;
    try Error.check(Api.ort.GetDenotationFromTypeInfo.?(apiCast(self), cStr(&denotation), &len));
    return if (denotation) |d| d[0..len] else "";
  }

  /// Release the TypeInfo object.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseTypeInfo.?(apiCast(self));
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
  custom_thread_creation_interface: ?ThreadingInterface = null,
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

  pub const ThreadingInterface = struct {
    ptr: ?*anyopaque,
    create_fn: @typeInfo(Api.c.OrtCustomCreateThreadFn).optional.child,
    join_fn: @typeInfo(Api.c.OrtCustomJoinThreadFn).optional.child,

    /// the instances must have
    ///   - `pub fn create(self: *Self, worker: *const fn(?*anyopaque), arg: ?*anyopaque) ?*const anyopaque`
    ///   - `pub fn join(handle: *const anyopaque) void` interface
    /// interface can be a pointer to a non-0 sized struct or an instance of 0 sized struct
    pub fn fromContext(instance: anytype) @This() {
      const T = @TypeOf(instance);
      const Sub = blk: {
        const Temp = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
        break :blk if (@typeInfo(Temp) == .pointer) @typeInfo(Temp).pointer.child else Temp;
      };
      return .{
        .ptr = if (@bitSizeOf(Sub) == 0) null else apiCast(instance),
        .create_fn = &struct {
          pub fn create(ctx: ?*anyopaque, worker: ?*const fn(?*anyopaque) callconv(.c) void, arg: ?*anyopaque) callconv(.c) Api.c.OrtCustomThreadHandle {
            const original: *Sub = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @ptrCast(ctx.?);
            return @ptrCast(original.create(worker, arg));
          }
        }.create,
        .join_fn = &struct {
          pub fn join(handle: Api.c.OrtCustomThreadHandle) callconv(.c) void {
            Sub.join(@ptrCast(handle));
          }
        }.join,
      };
    }
  };

  pub const C = opaque {
    pub const Underlying = Api.c.OrtThreadingOptions;
    /// Wraps OrtApi::CreateThreadingOptions
    pub fn init() !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateThreadingOptions.?(apiCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::SetGlobalIntraOpNumThreads
    pub fn intraOpNumThreads(self: *@This(), num_threads: c_int) !void {
      try Error.check(Api.ort.SetGlobalIntraOpNumThreads.?(apiCast(self), num_threads));
    }

    /// Wraps OrtApi::SetGlobalInterOpNumThreads
    pub fn interOpNumThreads(self: *@This(), num_threads: c_int) !void {
      try Error.check(Api.ort.SetGlobalInterOpNumThreads.?(apiCast(self), num_threads));
    }

    /// Wraps OrtApi::SetGlobalSpinControl
    pub fn spinControl(self: *@This(), allow_spinning: bool) !void {
      try Error.check(Api.ort.SetGlobalSpinControl.?(apiCast(self), @intFromBool(allow_spinning)));
    }

    /// Wraps OrtApi::SetGlobalDenormalAsZero
    pub fn denormalAsZero(self: *@This()) !void {
      try Error.check(Api.ort.SetGlobalDenormalAsZero.?(apiCast(self)));
    }

    pub fn customThreadingInterface(self: *@This(), interface: ThreadingInterface) !void {
      try Error.check(Api.ort.SetGlobalCustomCreateThreadFn.?(apiCast(self), interface.create_fn));
      if (interface.ptr) |p| try Error.check(Api.ort.SetGlobalCustomThreadCreationOptions.?(apiCast(self), p));
      try Error.check(Api.ort.SetGlobalCustomJoinThreadFn.?(apiCast(self), interface.join_fn));
    }

    pub fn intraOpThreadAffinity(self: *@This(), affinity_string: [*:0]const u8) !void {
      try Error.check(Api.ort.SetGlobalIntraOpThreadAffinity.?(apiCast(self), affinity_string));
    }

    /// Release the ThreadingOptions object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseThreadingOptions.?(apiCast(self));
    }
  };

  pub fn c(self: @This()) !*C {
    var retval = try C.init();
    errdefer retval.deinit();
    if (self.intraop_threads != 0) try retval.intraOpNumThreads(self.intraop_threads);
    if (self.interop_threads != 0) try retval.interOpNumThreads(self.interop_threads);
    if (self.allow_spinning) |spinning| try retval.spinControl(spinning);
    if (self.denormals_as_zero) try retval.denormalAsZero();
    if (self.custom_thread_creation_interface) |iface| try retval.customThreadingInterface(iface);
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
    errdefer retval.deinit();
    if (self.element_type) |t| try retval.setElementType(t);
    if (self.dimensions) |dims| try retval.setDimensions(dims);
    if (self.names) |names| try retval.setSymbolicDimensions(names);
    return retval;
  }

  pub const C = opaque {
    pub const Underlying = Api.c.OrtTensorTypeAndShapeInfo;
    /// Create an ::OrtTensorTypeAndShapeInfo object
    pub fn init() !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateTensorTypeAndShapeInfo.?(apiCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Set element type in ::OrtTensorTypeAndShapeInfo
    pub fn setElementType(self: *@This(), dtype: Value.Sub.Tensor.ElementDataType) !void {
      try Error.check(Api.ort.SetTensorElementType.?(apiCast(self), @intFromEnum(dtype)));
    }

    /// Set shape information in ::OrtTensorTypeAndShapeInfo
    pub fn setDimensions(self: *@This(), dims: []const i64) !void {
      try Error.check(Api.ort.SetDimensions.?(apiCast(self), dims.ptr, dims.len));
    }

    /// Set shape information in ::OrtTensorTypeAndShapeInfo
    pub fn setSymbolicDimensions(self: *@This(), names: [][*:0]const u8) !void {
      try Error.check(Api.ort.SetSymbolicDimensions.?(apiCast(self), cStr(names.ptr), names.len));
    }

    /// Get element type in ::OrtTensorTypeAndShapeInfo
    pub fn elementType(self: *const @This()) !Value.Sub.Tensor.ElementDataType {
      var out: Value.Sub.Tensor.ElementDataType = undefined;
      try Error.check(Api.ort.GetTensorElementType.?(apiCast(self), apiCast(&out)));
      return out;
    }

    /// Get dimension count in ::OrtTensorTypeAndShapeInfo
    pub fn dimensionsCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.GetDimensionsCount.?(apiCast(self), &out));
      return out;
    }

    pub fn getSymbolicDimensions(self: *const @This(), out: [][*:0]const u8) !void {
      try Error.check(Api.ort.GetSymbolicDimensions.?(apiCast(self), cStr(out.ptr), out.len));
    }

    /// Get dimensions in ::OrtTensorTypeAndShapeInfo
    pub fn getDimensions(self: *const @This(), out: []i64) !void {
      try Error.check(Api.ort.GetDimensions.?(apiCast(self), out.ptr, out.len));
    }

    /// Get total number of elements in a tensor shape
    /// For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
    pub fn shapeElementCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.GetTensorShapeElementCount.?(apiCast(self), &out));
      return out;
    }

    /// Release the TensorTypeAndShapeInfo object.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseTensorTypeAndShapeInfo.?(apiCast(self));
    }
  };
};

/// Wrapper for any data type that can be passed to or returned from an ONNX session.
pub const Value = opaque {
  pub const Underlying = Api.c.OrtValue;
  pub const Info = opaque {
    pub const Underlying = Api.c.OrtValueInfo;
    /// Get the value name.
    pub fn getName(self: *const @This()) ![*:0]const u8 {
      var out: ?[*:0]const u8 = null;
      try Error.check(Api.ort.GetValueInfoName.?(apiCast(self), cStr(&out)));
      return out orelse "";
    }

    /// Get the type information.
    /// The returned TypeInfo must be deinitialized by the caller.
    pub fn getTypeInfo(self: *const @This()) !*const TypeInfo {
      var out: ?*const TypeInfo = null;
      try Error.check(Api.ort.GetValueInfoTypeInfo.?(apiCast(self), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Get the OrtNode that produces this value.
    /// Returns null if not produced by a node (e.g. graph input).
    /// Also returns the output index.
    pub fn getProducer(self: *const @This()) !?struct { node: *const Node, index: usize } {
      var node: ?*const Node = null;
      var idx: usize = 0;
      try Error.check(Api.ort.ValueInfo_GetValueProducer.?(apiCast(self), apiCast(&node), &idx));
      if (node) |n| return .{ .node = n, .index = idx };
      return null;
    }

    /// Returns a boolean indicating if the given value is a graph output.
    pub fn isGraphOutput(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsGraphOutput.?(apiCast(self), &out));
      return out;
    }

    /// Returns a boolean indicating if the given value is a required graph input.
    pub fn isRequiredGraphInput(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsRequiredGraphInput.?(apiCast(self), &out));
      return out;
    }

    /// Returns a boolean indicating if the given value is an optional graph input.
    pub fn isOptionalGraphInput(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsOptionalGraphInput.?(apiCast(self), &out));
      return out;
    }

    /// Returns a boolean indicating if the given value is a constant initializer.
    pub fn isConstantInitializer(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsConstantInitializer.?(apiCast(self), &out));
      return out;
    }

    /// Get the underlying initializer value.
    /// Returns null if this is not an initializer.
    /// Note: The returned Value lifetime is tied to the Graph/ValueInfo; do not deinit it.
    pub fn getInitializerValue(self: *const @This()) !?*const Value {
      var out: ?*const Value = null;
      try Error.check(Api.ort.ValueInfo_GetInitializerValue.?(apiCast(self), apiCast(&out)));
      return out;
    }

    /// Get the number of consumers of a value as a node input.
    /// Only nodes are considered "consumers".
    /// Wraps OrtApi::ValueInfo_GetValueNumConsumers
    pub fn getNumConsumers(self: *const @This()) !usize {
      var count: usize = 0;
      try Error.check(Api.ort.ValueInfo_GetValueNumConsumers.?(apiCast(self), &count));
      return count;
    }

    /// Returns information for all consumer nodes that use the value as an input.
    /// Buffers 'nodes' and 'input_indices' must be pre-allocated to the size returned by getNumConsumers().
    /// Index is set to -1 for an "implicit" input to a consumer node that contains a subgraph.
    /// Wraps OrtApi::ValueInfo_GetValueConsumers
    pub fn getConsumers(self: *const @This(), nodes: []*const Node, input_indices: []i64) !void {
      std.debug.assert(nodes.len == input_indices.len);
      const nodes_ptr: [*]?*const Node = @ptrCast(nodes.ptr);
      try Error.check(Api.ort.ValueInfo_GetValueConsumers.?(
          apiCast(self),
          apiCast(nodes_ptr),
          cCast(input_indices.ptr),
          nodes.len,
      ));
    }

    /// Get information about an external initializer (filepath, offset, size).
    /// Returns null if this ValueInfo does not represent an external initializer.
    /// The returned ExternalInitializerInfo must be freed with deinit().
    /// Wraps OrtApi::ValueInfo_GetExternalInitializerInfo
    pub fn getExternalInitializerInfo(self: *const @This()) !?*ExternalInitializerInfo {
      var out: ?*ExternalInitializerInfo = null;
      try Error.check(Api.ort.ValueInfo_GetExternalInitializerInfo.?(apiCast(self), apiCast(&out)));
      return out;
    }

    /// Returns true if the value is defined in an outer scope (parent graph).
    /// Wraps OrtApi::ValueInfo_IsFromOuterScope
    pub fn isFromOuterScope(self: *const @This()) !bool {
      var out: bool = false;
      try Error.check(Api.ort.ValueInfo_IsFromOuterScope.?(apiCast(self), &out));
      return out;
    }

    /// Release the ValueInfo.
    /// Do not call this if the ValueInfo has been passed to a Graph (SetInputs/SetOutputs),
    /// as the Graph takes ownership.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseValueInfo.?(apiCast(self));
    }
  };

  /// High-level categorization of ONNX objects (Tensors, Maps, Sequences, etc.).
  /// Synced with onnx TypeProto oneof
  pub const Type = enum(Api.c.ONNXType) {
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
      pub const Underlying = Api.c.OrtValue;
      /// Defines the underlying data type of elements within a tensor (e.g., f32, i64).
      /// @intCast(Api.c copied it from TensorProto::DataType [Refer to the Api.c comment])
      pub const ElementDataType = enum(Api.c.ONNXTensorElementDataType) {
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
        try Error.check(Api.ort.CreateTensorAsOrtValue.?(apiCast(allocator), shape.ptr, shape.len, @intFromEnum(dtype), apiCast(&self)));
        return self orelse error.OutOfMemory;
      }

      /// Create a tensor backed by a user supplied buffer
      /// p_data is owned by caller. deinit() won't release p_data.
      pub fn initWithData(info: *const Allocator.MemoryInfo, p_data: []u8, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateTensorWithDataAsOrtValue.?(apiCast(info), cCast(p_data.ptr), p_data.len, shape.ptr, shape.len, @intFromEnum(dtype), apiCast(&self)));
        return self orelse error.OutOfMemory;
      }

      /// Create an OrtValue for a Tensor that uses pre-existing memory.
      /// ORT will take ownership of the memory and free it using the provided deleter when no longer in use.
      pub fn initWithDataAndDeleter(deleter: *Allocator, p_data: []u8, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateTensorWithDataAndDeleterAsOrtValue.?(
            apiCast(deleter),
            cCast(p_data.ptr),
            p_data.len,
            shape.ptr,
            shape.len,
            @intFromEnum(dtype),
            apiCast(&self)
        ));
        return self orelse error.OutOfMemory;
      }

      /// Get a pointer to the raw data inside a tensor
      /// Used to read/write/modify the internal tensor data directly.
      pub fn getData(self: *@This(), comptime T: type) ![]T {
        var ptr: ?[*]T = null;
        try Error.check(Api.ort.GetTensorMutableData.?(apiCast(self), @ptrCast(&ptr)));
        return (ptr orelse return error.OutOfMemory)[0 .. try self.getSizeInBytes() / @sizeOf(T)];
      }

      /// Get a const pointer to the raw data inside a tensor
      pub fn getDataConst(self: *const @This(), comptime T: type) ![]const u8 {
        var ptr: ?[*]const u8 = null;
        try Error.check(Api.ort.GetTensorData.?(apiCast(self), @ptrCast(&ptr)));
        return (ptr orelse return error.OutOfMemory)[0 .. try self.getSizeInBytes() / @sizeOf(T)];
      }

      /// Compute total size in bytes of the tensor data contained in an OrtValue.
      pub fn getSizeInBytes(self: *const @This()) !usize {
        var out: usize = 0;
        try Error.check(Api.ort.GetTensorSizeInBytes.?(apiCast(self), &out));
        return out;
      }

      /// Get type and shape information from a tensor ::OrtValue
      pub fn getTypeAndShape(self: *const @This()) !*TensorTypeAndShapeInfo.C {
        var out: ?*TensorTypeAndShapeInfo.C = null;
        try Error.check(Api.ort.GetTensorTypeAndShape.?(apiCast(self), apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      pub fn at(self: *@This(), indices: []const i64, comptime T: type) !*T {
        var out: ?*T = null;
        try Error.check(Api.ort.TensorAt.?(apiCast(self), indices.ptr, indices.len, @ptrCast(&out)));
        return out.?; // returns pointer in memory so this can't be null
      }

      /// Returns a pointer to the ::OrtMemoryInfo of a Tensor
      pub fn getMemoryInfo(self: *@This()) !*const Allocator.MemoryInfo {
        var out: ?*const Allocator.MemoryInfo = null;
        try Error.check(Api.ort.GetTensorMemoryInfo.?(apiCast(self), apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      pub const String = opaque {
        pub const Underlying = Api.c.OrtValue;
        /// Get total byte length for all strings in a string tensor
        pub fn getStringDataLength(self: *@This()) !usize {
          var out: usize = 0;
          try Error.check(Api.ort.GetStringTensorDataLength.?(apiCast(self), &out));
          return out;
        }

        pub fn getStringContent(self: *@This(), out_bytes: []u8, offsets: []usize) !void {
          try Error.check(Api.ort.GetStringTensorContent.?(apiCast(self), @ptrCast(out_bytes.ptr), out_bytes.len, offsets.ptr, offsets.len));
        }

        /// Set all strings at once in a string tensor
        pub fn fillString(self: *@This(), strings: []const [*:0]const u8) !void {
          try Error.check(Api.ort.FillStringTensor.?(apiCast(self), cStr(strings.ptr), strings.len));
        }

        pub fn getStringElementLength(self: *const @This(), index: usize) !usize {
          var out: usize = undefined;
          try Error.check(Api.ort.GetStringTensorElementLength.?(apiCast(self), index, &out));
          return out;
        }

        pub fn getStringElement(self: *const @This(), index: usize, out_bytes: []u8) !void {
          try Error.check(Api.ort.GetStringTensorElement.?(apiCast(self), out_bytes.len, index, @ptrCast(out_bytes.ptr)));
        }

        /// Set a single string in a string tensor
        pub fn fillStringElement(self: *@This(), s: [*:0]const u8, index: usize) !void {
          try Error.check(Api.ort.FillStringTensorElement.?(apiCast(self), s, index));
        }

        pub fn getResizedStringElementBuffer(self: *@This(), index: usize, len: usize) ![]u8 {
          var out: ?[*]u8 = null;
          try Error.check(Api.ort.GetResizedStringTensorElementBuffer.?(apiCast(self), index, len, cCast(&out)));
          return (out orelse return error.OutOfMemory)[0 .. len];
        }
      };

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }

      pub fn deinit(self: *@This()) void {
        self.toValue().deinit();
      }
    };

    pub const Sequence = opaque {
      pub const Underlying = Api.c.OrtValue;
      /// Create a Value representing a Sequence (ONNX_TYPE_SEQUENCE).
      pub fn init(values: []const *const Value) !*@This() {
        return (try Value._init(values, .SEQUENCE)).asType(.SEQUENCE);
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

      pub fn deinit(self: *@This()) void {
        self.toValue().deinit();
      }
    };

    pub const Map = opaque {
      pub const Underlying = Api.c.OrtValue;
      /// Create a Value representing a Map (ONNX_TYPE_MAP).
      /// keys: A Tensor Value containing keys.
      /// values: A Tensor Value containing values.
      /// The API ref-counts the inputs; you may deinit them after this call if you don't need them elsewhere.
      pub fn init(keys: *const Value, values: *const Value) !*@This() {
        return (try Value._init(&[2]*const Value{ keys, values }, .MAP)).asType(.MAP);
      }

      ///  At index 0 is key, 1 is value
      ///  You must deinit the returned values
      pub fn getKV(self: *const @This(), allocator: *Allocator) !std.meta.Tuple(&.{*Value, *Value}) {
        const v = self.toValue();
        const keys = try v._getValue(0, allocator);
        errdefer keys.deinit();
        return .{ keys, try v._getValue(1, allocator), };
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }

      pub fn deinit(self: *@This()) void {
        self.toValue().deinit();
      }
    };

    pub const Opaque = opaque {
      pub const Underlying = Api.c.OrtValue;
      /// Create a Value wrapping an Opaque type.
      pub fn init(domain_name: [*:0]const u8, type_name: [*:0]const u8, data: []const u8) !*@This() {
        var out: ?*@This() = null;
        try Error.check(Api.ort.CreateOpaqueValue.?(cStr(domain_name), cStr(type_name), data.ptr, data.len, apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Get data from an Opaque Value.
      /// buffer: Buffer to write data into. Must match the internal size.
      pub fn getData(self: *const @This(), comptime Out: type, out: []Out, domain_name: [*:0]const u8, type_name: [*:0]const u8) !void {
        try Error.check(Api.ort.GetOpaqueValue.?(cStr(domain_name), cStr(type_name), apiCast(self), @ptrCast(out.ptr), out.len * @sizeOf(Out)));
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }

      pub fn deinit(self: *@This()) void {
        self.toValue().deinit();
      }
    };

    pub const SparseTensor = opaque {
      pub const Underlying = Api.c.OrtValue;
      /// Identifies the storage format for sparse tensors (COO, CSR, or Block Sparse).
      /// These types are synced with internal SparseFormatFlags
      pub const Format = enum(Api.c.OrtSparseFormat) {
        SPARSE_UNDEFINED = @intCast(Api.c.ORT_SPARSE_UNDEFINED),
        SPARSE_COO = @intCast(Api.c.ORT_SPARSE_COO),
        SPARSE_CSRC = @intCast(Api.c.ORT_SPARSE_CSRC),
        BLOCK_SPARSE = @intCast(Api.c.ORT_SPARSE_BLOCK_SPARSE),
      };

      /// Identifies which specific indices buffer of a sparse tensor is being queried.
      /// Enum allows to query sparse tensor indices
      pub const IndicesFormat = enum(Api.c.OrtSparseIndicesFormat) {
        COO_INDICES = @intCast(Api.c.ORT_SPARSE_COO_INDICES),
        CSR_INNER_INDICES = @intCast(Api.c.ORT_SPARSE_CSR_INNER_INDICES),
        CSR_OUTER_INDICES = @intCast(Api.c.ORT_SPARSE_CSR_OUTER_INDICES),
        BLOCK_SPARSE_INDICES = @intCast(Api.c.ORT_SPARSE_BLOCK_SPARSE_INDICES),

        pub fn Type(comptime self: @This()) type {
          return switch (self) {
            .COO_INDICES, .CSR_INNER_INDICES, .CSR_OUTER_INDICES => i64,
            .BLOCK_SPARSE_INDICES => i32,
          };
        }
      };

      pub fn init(allocator: *Allocator, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateSparseTensorAsOrtValue.?(apiCast(allocator), shape.ptr, shape.len, @intFromEnum(dtype), apiCast(&self)));
        return self orelse error.OutOfMemory;
      }

      pub fn initWithValues(info: *const Allocator.MemoryInfo, p_data: ?[*]u8, dense_shape: []i64, shape: []const i64, dtype: Value.Sub.Tensor.ElementDataType) !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateSparseTensorWithValuesAsOrtValue.?(
            apiCast(info),
            cCast(p_data),
            dense_shape.ptr,
            dense_shape.len,
            shape.ptr,
            shape.len,
            @intFromEnum(dtype),
            apiCast(&self)
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
          apiCast(self),
          apiCast(mem_info),
          values_shape.ptr, values_shape.len,
          @as(*const anyopaque, @ptrCast(values)),
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
          apiCast(self),
          apiCast(mem_info),
          values_shape.ptr, values_shape.len,
          @as(*const anyopaque, @ptrCast(values)),
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
          apiCast(self),
          apiCast(mem_info),
          values_shape.ptr, values_shape.len,
          values,
          indices_shape.ptr, indices_shape.len,
          indices.ptr
        ));
      }

      /// Use user-supplied COO indices.
      /// indices: User buffer. Must outlive Value.
      ///
      /// if indices_data == null, a fully sparse tensor is created.
      pub fn useCooIndices(self: *@This(), indices_data: ?[*]i64, indices_data_len: usize) !void {
        try Error.check(Api.ort.UseCooIndices.?(apiCast(self), indices_data, indices_data_len));
      }

      /// Use user-supplied CSR indices.
      /// if inner_data == null, a fully sparse tensor is created.
      /// if outer_data == null, a fully sparse tensor is created.
      pub fn useCsrIndices(self: *@This(), inner_data: ?[*]i64, inner_data_len: usize, outer_data: ?[*]i64, outer_data_len: usize) !void {
        try Error.check(Api.ort.UseCsrIndices.?(apiCast(self), inner_data, inner_data_len, outer_data, outer_data_len));
      }

      /// Use user-supplied Block Sparse indices.
      ///
      /// if indices_data == null, a fully sparse tensor is created.
      pub fn useBlockSparseIndices(self: *@This(), indices_shape: []const i64, indices_data: ?[*]i32) !void {
        try Error.check(Api.ort.UseBlockSparseIndices.?(apiCast(self), indices_shape.ptr, indices_shape.len, indices_data));
      }

      /// Get the sparse format of the value.
      pub fn getSparseFormat(self: *const @This()) !Format {
        var out: Format = undefined;
        try Error.check(Api.ort.GetSparseTensorFormat.?(apiCast(self), apiCast(&out)));
        return out;
      }

      /// Get type info for the sparse tensor values.
      /// Caller must deinit the returned TensorTypeAndShapeInfo.
      pub fn getValuesTypeAndShape(self: *@This()) !*TensorTypeAndShapeInfo.C {
        var out: ?*TensorTypeAndShapeInfo.C = null;
        try Error.check(Api.ort.GetSparseTensorValuesTypeAndShape.?(apiCast(self), apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Get pointer to sparse values.
      pub fn getValues(self: *@This()) ![*]const u8 {
        var ptr: ?[*]const u8 = null;
        try Error.check(Api.ort.GetSparseTensorValues.?(apiCast(self), cCast(&ptr)));
        return ptr orelse error.OutOfMemory;
      }

      /// Get type info for the sparse tensor indices.
      /// Caller must deinit the returned TensorTypeAndShapeInfo.
      pub fn getIndicesTypeShape(self: *const @This(), format: IndicesFormat) !*TensorTypeAndShapeInfo.C {
        var out: ?*TensorTypeAndShapeInfo.C = null;
        try Error.check(Api.ort.GetSparseTensorIndicesTypeShape.?(apiCast(self), @intFromEnum(format), apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Get pointer to sparse indices. a slice of i64's or i32's depending on IndicesFormat.
      /// You must @ptrCast(@alignCast) to get the correct type.
      pub fn getIndices(self: *const @This(), comptime format: IndicesFormat) ![]const format.Type() {
        var ptr: ?[*]const format.Type() = null;
        var count: usize = 0;
        try Error.check(Api.ort.GetSparseTensorIndices.?(apiCast(self), @intFromEnum(format), &count, @ptrCast(&ptr)));
        return (ptr orelse return error.OutOfMemory)[0 .. count];
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }

      pub fn deinit(self: *@This()) void {
        self.toValue().deinit();
      }
    };

    pub const Optional = opaque {
      pub const Underlying = Api.c.OrtValue;
      /// Returns true if an optional type OrtValue has an element
      pub fn hasValue(self: *@This()) !bool {
        var out: c_int = 0;
        try Error.check(Api.ort.HasValue.?(apiCast(self), &out));
        return out != 0;
      }

      pub fn toValue(self_ptr: anytype) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, Value) {
        return @ptrCast(self_ptr);
      }

      pub fn deinit(self: *@This()) void {
        self.toValue().deinit();
      }
    };
  };

  pub fn _init(values: []const *const Value, t: Type) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateValue.?(apiCast(values.ptr), values.len, @intFromEnum(t), apiCast(&out)));
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

  pub fn asUnion(self_ptr: anytype) !AsUnion(@TypeOf(self_ptr)) {
    const U = AsUnion(@TypeOf(self_ptr));
    switch (try self_ptr.getType()) {
      inline else => |t| return @unionInit(U, @tagName(t), @ptrCast(self_ptr)),
    }
    return error.InvalidType;
  }

  /// You may use inline switching to get a comptime value for t and then cast to appropriate type
  pub fn asType(self_ptr: anytype, comptime t: Type) Utils.CopyPointerAttrs(@TypeOf(self_ptr), .one, t.Type()) {
    // Ptrcasting is safe since destination will have the same underlying type
    return @ptrCast(self_ptr);
  }

  /// Get TypeInfo.Type of an ::OrtValue
  pub fn getType(self: *const @This()) !Type {
    var out: Type = undefined;
    try Error.check(Api.ort.GetValueType.?(apiCast(self), apiCast(&out)));
    return out;
  }

  /// Get type information of an OrtValue
  pub fn getTypeInfo(self: *@This()) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.GetTypeInfo.?(apiCast(self), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get a non-tensor element from a Value (e.g., an element of a Sequence or Map).
  /// If Map: index 0 = keys, index 1 = values.
  /// If Sequence: index i = i-th element.
  /// The returned Value must be deinitialized by the caller.
  pub fn _getValue(self: *const @This(), index: c_int, allocator: *Allocator) !*@This() {
    var out: ?*Value = null;
    try Error.check(Api.ort.GetValue.?(apiCast(self), index, apiCast(allocator), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get the count of elements.
  /// If Map: returns 2.
  /// If Sequence: returns number of elements.
  pub fn _getCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.GetValueCount.?(apiCast(self), &out));
    return out;
  }

  /// Return true if this ::OrtValue is a tensor type
  pub fn isTensor(self: *const @This()) !bool {
    var out: c_int = 0;
    try Error.check(Api.ort.IsTensor.?(apiCast(self), &out));
    return out != 0;
  }

  /// Returns true if this ::OrtValue is a SparseTensor
  pub fn isSparseTensor(self: *const @This()) !bool {
    var out: c_int = 0;
    try Error.check(Api.ort.IsSparseTensor.?(apiCast(self), &out));
    return out != 0;
  }

  /// Release the OrtValue.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseValue.?(apiCast(self));
  }

  /// Warning: This uses Ep.api
  /// Get the OrtMemoryDevice from an OrtValue instance if it contains a Tensor.
  ///
  /// value: The OrtValue instance to get the memory device from.
  /// returns: Memory device if OrtValue contains a Tensor, nullptr otherwise.
  ///
  /// since Version 1.23.
  pub fn getMemoryDevice(self: *const @This()) !*const Allocator.MemoryDevice {
    return apiCastTo(Ep.api.underlying.Value_GetMemoryDevice.?(apiCast(self)) orelse return error.OutOfMemory, *const Allocator.MemoryDevice);
  }
};

/// Represenrs a single mathematical operation. Graph is made from these
pub const Node = opaque {
  pub const Underlying = Api.c.OrtNode;
  /// Create an OrtNode to add to an OrtGraph.
  ///
  /// Create attributes with `Op.Attr.init`. OrtOpAttr instances are copied by the node
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
    attributes: []*Op.Attr,
  ) !*@This() {
    var out: ?*@This() = null;
    const attributes_ptr: [*]?*Op.Attr = @ptrCast(attributes.ptr);
    try Error.check(Api.editor.underlying.CreateNode.?(
      cStr(operator_name),
      cStr(domain_name),
      cStr(node_name),
      cStr(input_names.ptr),
      input_names.len,
      cStr(output_names.ptr),
      output_names.len,
      apiCast(attributes_ptr),
      attributes.len,
      apiCast(&out),
    ));
    return out orelse error.OutOfMemory;
  }

  /// Returns a node's identifier.
  /// The node's identifier is only unique in the node's parent graph.
  pub fn getId(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetId.?(apiCast(self), &out));
    return out;
  }

  /// Returns a node's name. Can be an empty string.
  pub fn getName(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetName.?(apiCast(self), cStr(&out)));
    return out orelse "";
  }

  /// Returns a node's operator type (e.g., "Conv").
  pub fn getOperatorType(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetOperatorType.?(apiCast(self), cStr(&out)));
    return out orelse "";
  }

  /// Returns a node's domain name.
  pub fn getDomain(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetDomain.?(apiCast(self), cStr(&out)));
    return out orelse "";
  }

  /// Get the opset version in which the given node's operator type was first defined.
  pub fn getSinceVersion(self: *const @This()) !c_int {
    var out: c_int = 0;
    try Error.check(Api.ort.Node_GetSinceVersion.?(apiCast(self), &out));
    return out;
  }

  /// Returns the number of node inputs.
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumInputs.?(apiCast(self), &out));
    return out;
  }

  /// Returns the node's inputs as ValueInfo instances.
  pub fn getInputs(self: *const @This(), out: []*const Value.Info) !void {
    const out_ptr: [*]?*const Value.Info = @ptrCast(out.ptr);
    try Error.check(Api.ort.Node_GetInputs.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  /// Returns the number of node outputs.
  pub fn getOutputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumOutputs.?(apiCast(self), &out));
    return out;
  }

  /// Returns the node's outputs as ValueInfo instances.
  pub fn getOutputs(self: *const @This(), out: []?*const Value.Info) !void {
    try Error.check(Api.ort.Node_GetOutputs.?(apiCast(self), apiCast(out.ptr), out.len));
  }

  /// Returns the number of node implicit inputs.
  pub fn getImplicitInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumImplicitInputs.?(apiCast(self), &out));
    return out;
  }

  /// Get the implicit inputs, as ValueInfo instances, that are used within the given node's subgraphs.
  pub fn getImplicitInputs(self: *const @This(), out: []*const Value.Info) !void {
    // we need to use @ptrCast here because the api tales optional pointers but never sets to null
    const out_ptr: [*]?*const Value.Info = @ptrCast(out.ptr);
    try Error.check(Api.ort.Node_GetImplicitInputs.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  /// Returns the number of node attributes.
  pub fn getAttributeCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumAttributes.?(apiCast(self), &out));
    return out;
  }

  /// Returns a node's attributes as Op.Attr instances.
  pub fn getAttributes(self: *const @This(), out: []*const Op.Attr) !void {
    // we need to use @ptrCast here because the api tales optional pointers but never sets to null
    const out_ptr: [*]?*const Op.Attr = @ptrCast(out.ptr);
    try Error.check(Api.ort.Node_GetAttributes.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  /// Gets the Node's attribute as Op.Attr by name.
  /// Returns null if the attribute is not found or is an unset optional attribute.
  pub fn getAttributeByName(self: *const @This(), name: [*:0]const u8) !?*const Op.Attr {
    var out: ?*const Op.Attr = null;
    // ORT returns ORT_NOT_FOUND if the attribute doesn't exist, which we translate to returning null
    // However, if we blindly check(...), it will return error.NotFound.
    // We can use Error.check, catch NotFound, and return null.
    Error.check(Api.ort.Node_GetAttributeByName.?(apiCast(self), cStr(name), apiCast(&out))) catch |err| switch (err) {
      Error.Set.OrtErrorNotFound => return null,
      else => return err,
    };

    return out;
  }

  /// Returns the number of subgraphs contained by the given node.
  pub fn getSubgraphCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Node_GetNumSubgraphs.?(apiCast(self), &out));
    return out;
  }

  pub const SubgraphInfo = struct {
    graph: []*const Graph,
    attribute_name: []?[*:0]const u8,
  };

  /// Get the subgraphs, as Graph instances, contained by the given node.
  /// Also returns the attribute name associated with each subgraph.
  pub fn getSubgraphs(self: *const @This(), out_graph: []*const Graph, out_attribute_name: []?[*:0]const u8) !void {
    std.debug.assert(out_graph.len == out_attribute_name.len);
    const out_graph_ptr: [*]?*const Graph = @ptrCast(out_graph.ptr);
    try Error.check(Api.ort.Node_GetSubgraphs.?(apiCast(self), apiCast(out_graph_ptr), out_graph.len, cStr(out_attribute_name.ptr)));
  }

  /// Get the node's parent Graph instance.
  /// Can return null if the Node was created without an owning graph.
  pub fn getGraph(self: *const @This()) !?*const Graph {
    var out: ?*const Graph = null;
    try Error.check(Api.ort.Node_GetGraph.?(apiCast(self), apiCast(&out)));
    return out;
  }

  /// Returns the execution provider name that this node is assigned to run on.
  /// Returns null if the node has not been assigned to any execution provider yet.
  pub fn getEpName(self: *const @This()) !?[*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Node_GetEpName.?(apiCast(self), cStr(&out)));
    return out;
  }

  /// Release an OrtNode if it was not added to an OrtGraph.
  /// Do not call this if the node has been added to a graph.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseNode.?(apiCast(self));
  }
};

/// Represents directed acyclic graph (DAG) of the model
pub const Graph = opaque {
  pub const Underlying = Api.c.OrtGraph;
  /// Create an empty OrtGraph.
  /// Note: Requires Model Editor API initialization.
  pub fn init() !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.editor.underlying.CreateGraph.?(apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Set the inputs for the OrtGraph.
  /// This will replace any existing inputs with the new values.
  /// The OrtGraph takes ownership of the Value.Info instances; do NOT call deinit on them.
  pub fn setInputs(self: *@This(), inputs: []*Value.Info) !void {
    const inputs_ptr: [*]?*Value.Info = @ptrCast(inputs.ptr);
    try Error.check(Api.editor.underlying.SetGraphInputs.?(
      apiCast(self), 
      apiCast(inputs_ptr), 
      inputs.len
    ));
  }

  /// Set the outputs for the OrtGraph.
  /// This will replace any existing outputs with the new values.
  /// The OrtGraph takes ownership of the Value.Info instances; do NOT call deinit on them.
  pub fn setOutputs(self: *@This(), outputs: []*Value.Info) !void {
    const outputs_ptr: [*]?*Value.Info = @ptrCast(outputs.ptr);
    try Error.check(Api.editor.underlying.SetGraphOutputs.?(
      apiCast(self), 
      apiCast(outputs_ptr), 
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
      apiCast(self),
      cStr(name),
      apiCast(tensor),
      data_is_external
    ));
  }

  /// Add an OrtNode to the OrtGraph.
  /// The OrtGraph takes ownership of OrtNode; do NOT call deinit on it.
  pub fn addNode(self: *@This(), node: *Node) !void {
    try Error.check(Api.editor.underlying.AddNodeToGraph.?(
      apiCast(self),
      apiCast(node)
    ));
  }

  /// Returns the graph's name.
  pub fn getName(self: *const @This()) ![*:0]const u8 {
    var out: ?[*:0]const u8 = null;
    try Error.check(Api.ort.Graph_GetName.?(apiCast(self), cStr(&out)));
    return out orelse "";
  }

  /// Get the filepath to the model from which this OrtGraph was constructed.
  /// Returns empty string if unknown (static pointer).
  /// Do NOT free the returned string.
  pub fn getModelPath(self: *const @This()) !Utils.Path {
    var out: ?Utils.Path = null;
    try Error.check(Api.ort.Graph_GetModelPath.?(apiCast(self), pathCast(&out)));
    // in error case, we return a pointer to a static empty string
    return out orelse @ptrCast(@constCast(&empty_path));
  }

  /// Returns the ONNX IR version.
  pub fn getOnnxIRVersion(self: *const @This()) !i64 {
    var out: i64 = 0;
    try Error.check(Api.ort.Graph_GetOnnxIRVersion.?(apiCast(self), &out));
    return out;
  }

  /// Returns the number of operator sets that the graph's model uses.
  pub fn getOperatorSetCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumOperatorSets.?(apiCast(self), &out));
    return out;
  }

  pub const OperatorSet = struct {
    domain: [*:0]const u8,
    version: i64,
  };

  /// Returns the operator sets that the graph's model uses.
  pub fn getOperatorSets(self: *const @This(), out_domains: [][*:0]const u8, out_versions: []i64) !void {
    std.debug.assert(out_domains.len == out_versions.len);
    try Error.check(Api.ort.Graph_GetOperatorSets.?(
      apiCast(self), 
      cStr(out_domains.ptr), 
      cCast(out_versions.ptr), 
      out_domains.len,
    ));
  }

  /// Returns the number of graph inputs.
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumInputs.?(apiCast(self), &out));
    return out;
  }

  /// Returns the graph's inputs as Value.Info instances.
  pub fn getInputs(self: *const @This(), out: []*const Value.Info) !void {
    const out_ptr: [*]?*const Value.Info = @ptrCast(out.ptr);
    try Error.check(Api.ort.Graph_GetInputs.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  /// Returns the number of graph outputs.
  pub fn getOutputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumOutputs.?(apiCast(self), &out));
    return out;
  }

  /// Returns the graph's outputs as Value.Info instances.
  pub fn getOutputs(self: *const @This(), out: []*const Value.Info) !void {
    const out_ptr: [*]?*const Value.Info = @ptrCast(out.ptr);
    try Error.check(Api.ort.Graph_GetOutputs.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  /// Returns the number of graph initializers.
  pub fn getInitializerCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumInitializers.?(apiCast(self), &out));
    return out;
  }

  /// Returns the graph's initializers as Value.Info instances.
  /// To get the actual data, call `Value.Info.getInitializerValue`.
  pub fn getInitializers(self: *const @This(), out: []*const Value.Info) !void {
    const out_ptr: [*]?*const Value.Info = @ptrCast(out.ptr);
    try Error.check(Api.ort.Graph_GetInitializers.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  /// Returns the number of graph nodes.
  pub fn getNodeCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.Graph_GetNumNodes.?(apiCast(self), &out));
    return out;
  }

  /// Returns the graph's nodes as Node instances.
  ///
  /// NOTE: The C API doesn't return the count, so you MUST ensure the slice is exactly the size of Graph_GetNumNodes.
  pub fn getNodes(self: *const @This(), out: []*const Node) !void {
    const out_ptr: [*]?*const Node = @ptrCast(out.ptr);
    try Error.check(Api.ort.Graph_GetNodes.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  /// Get the parent node for the given graph, if any exists.
  /// Returns null if this is a top-level graph.
  pub fn getParentNode(self: *const @This()) !?*const Node {
    var out: ?*const Node = null;
    try Error.check(Api.ort.Graph_GetParentNode.?(apiCast(self), apiCast(&out)));
    return out;
  }

  /// Returns an OrtGraph that contains a subset of nodes in the source OrtGraph.
  /// The lifetime of the returned Graph view is tied to the source Graph.
  /// Note: The returned graph must be released via `deinit`.
  pub fn getGraphView(self: *const @This(), nodes: []*const Node) !*@This() {
    var out: ?*@This() = null;
    const nodes_ptr: [*]?*const Node = @ptrCast(nodes.ptr);
    try Error.check(Api.ort.Graph_GetGraphView.?(
      apiCast(self),
      apiCast(nodes_ptr),
      nodes.len,
      apiCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  /// Get ModelMetadata from the OrtGraph.
  /// The returned metadata must be released by the caller.
  pub fn getModelMetadata(self: *const @This()) !*Model.Metadata {
    var out: ?*Model.Metadata = null;
    try Error.check(Api.ort.Graph_GetModelMetadata.?(apiCast(self), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Release an OrtGraph.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseGraph.?(apiCast(self));
  }
};

/// Represents the top-level .onnx file structure
pub const Model = opaque {
  pub const Underlying = Api.c.OrtModel;
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
      cStr(domain_names.ptr),
      @as([*c]const c_int, @ptrCast(opset_versions.ptr)), // @ptrCast is needed because c_int has no underlying type
      domain_names.len,
      apiCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  /// Add an OrtGraph to an OrtModel.
  /// This should be called once when creating a new model.
  /// The OrtModel takes ownership of the OrtGraph; do NOT call deinit on the graph.
  pub fn addGraph(self: *@This(), graph: *Graph) !void {
    try Error.check(Api.editor.underlying.AddGraphToModel.?(
      apiCast(self),
      apiCast(graph)
    ));
  }

  /// Release an OrtModel.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseModel.?(apiCast(self));
  }

  pub const Metadata = opaque {
    pub const Underlying = Api.c.OrtModelMetadata;
    /// Get `producer name` from an ::OrtModelMetadata
    ///
    /// allocator: The allocator used to allocate the returned string.
    /// Returns a null terminated string allocated using `allocator`. 
    /// The caller must free the returned pointer using `allocator.free()`.
    pub fn getProducerName(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.ModelMetadataGetProducerName.?(
        apiCast(self),
        apiCast(allocator),
        cStr(&out)
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
        apiCast(self),
        apiCast(allocator),
        cStr(&out)
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
        apiCast(self),
        apiCast(allocator),
        cStr(&out)
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
        apiCast(self),
        apiCast(allocator),
        cStr(&out)
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
        apiCast(self),
        apiCast(allocator),
        cStr(&out)
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
        apiCast(self),
        apiCast(allocator),
        cStr(key),
        cStr(&out)
      ));
      return out;
    }

    /// Get version number from an ::OrtModelMetadata
    pub fn getVersion(self: *const @This()) !i64 {
      var out: i64 = 0;
      try Error.check(Api.ort.ModelMetadataGetVersion.?(
        apiCast(self),
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
    pub fn getCustomMetadataMapKeys(self: *const @This(), allocator: *Allocator) !struct {
      keys: ?[][*:0]u8,

      pub fn deinit(self_: *@This(), allocator_: *Allocator) void {
        if (self_.keys) |keys| {
          for (keys) |key| {
            allocator_.free(key);
          }
          allocator_.free(keys);
        }
      }
    } {
      var keys_ptr: ?[*][*:0]u8 = null;
      var num_keys: i64 = 0;
      
      try Error.check(Api.ort.ModelMetadataGetCustomMetadataMapKeys.?(
        apiCast(self),
        apiCast(allocator),
        cStr(&keys_ptr),
        &num_keys
      ));

      return .{ .keys = if (keys_ptr) |ptr| ptr[0..@intCast(num_keys)] else null };
    }

    /// Release an ::OrtModelMetadata.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseModelMetadata.?(apiCast(self));
    }
  };

  pub const CompilationOptions = opaque {
    pub const Underlying = Api.c.OrtModelCompilationOptions;
    /// Creates an OrtModelCompilationOptions object from an existing OrtSessionOptions object.
    ///
    /// env: The OrtEnv.
    /// session_options: The Session.Options to use as a base.
    ///
    /// Note: Requires Compile API initialization.
    pub fn init(session_options: *const Session.Options.C) !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.compiler.underlying.CreateModelCompilationOptionsFromSessionOptions.?(
        Api.env.underlying,
        apiCast(session_options), // Assuming Session.Options is opaque
        apiCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    /// Sets the file path to the input ONNX model to compile.
    pub fn setInputModelPath(self: *@This(), path: Utils.Path) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetInputModelPath.?(
        apiCast(self),
        pathCast(path),
      ));
    }

    /// Sets the buffer that stores the bytes of the loaded ONNX model to compile.
    pub fn setInputModelFromBuffer(self: *@This(), buffer: []const u8) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetInputModelFromBuffer.?(
        apiCast(self),
        cCast(buffer.ptr),
        buffer.len
      ));
    }

    /// Sets the file path for the output ONNX model.
    pub fn setOutputModelPath(self: *@This(), path: Utils.Path) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelPath.?(
        apiCast(self),
        pathCast(path),
      ));
    }

    pub const ModelOutputBuffer = struct {
      ptr: ?*anyopaque = null,
      len: usize = 0,
    };

    /// Configures model compilation to store the output compiled ONNX model in a buffer.
    ///
    /// allocator: The allocator used to allocate the buffer.
    /// Returns a slice to the allocated buffer. The memory is owned by the allocator/caller 
    /// context but specifically allocated here.
    ///
    /// WARNING: the `out` struct must be pinned in memory until Model.compile is called. They are populated during that call automatically
    pub fn setOutputModelBuffer(self: *@This(), allocator: *Allocator, out: *ModelOutputBuffer) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelBuffer.?(
        apiCast(self),
        apiCast(allocator),
        @ptrCast(&out.ptr), // cast to anyopaque c pointer which has no `Underlying`
        &out.len
      ));
    }

    /// Enables or disables the embedding of EPContext binary data.
    pub fn setEpContextEmbedMode(self: *@This(), embed: bool) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetEpContextEmbedMode.?(
        apiCast(self),
        embed
      ));
    }

    /// Sets flags representing one or more boolean options to enable.
    pub fn setFlags(self: *@This(), flags: Api.compiler.Flags) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetFlags.?(
        apiCast(self),
        @bitCast(flags)
      ));
    }

    /// Set the graph optimization level.
    pub fn setGraphOptimizationLevel(self: *@This(), level: Session.GraphOptimizationLevel) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetGraphOptimizationLevel.?(
        apiCast(self),
        @intFromEnum(level)
      ));
    }

    /// Compiles an input ONNX model with the given compilation options.
    /// Note: The input/output paths must have been set on the options object.
    pub fn compile(self: *const @This()) !void {
      try Error.check(Api.compiler.underlying.CompileModel.?(
        Api.env.underlying,
        apiCast(self)
      ));
    }

    /// Sets information related to EP context binary file.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetEpContextBinaryInformation
    pub fn setEpContextBinaryInformation(self: *@This(), output_dir: Utils.Path, model_name: Utils.Path) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetEpContextBinaryInformation.?(
          apiCast(self),
          pathCast(output_dir),
          pathCast(model_name),
      ));
    }

    /// Optionally sets the file that should store external initializers for the compiled model.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetOutputModelExternalInitializersFile
    pub fn setOutputModelExternalInitializersFile(self: *@This(), path: Utils.Path, threshold: usize) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelExternalInitializersFile.?(
          apiCast(self),
          pathCast(path),
          threshold,
      ));
    }

    /// Sets a custom function called by ORT to write out the output model's bytes.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetOutputModelWriteFunc
    pub fn setOutputModelWriteFunc(self: *@This(), interface: WriteInterface) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelWriteFunc.?(
          apiCast(self),
          interface.write_fn,
          interface.ptr,
      ));
    }

    /// Sets a custom function to specify whether initializers should be stored within the model or externally.
    /// Wraps OrtCompileApi::ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc
    pub fn setOutputModelGetInitializerLocationFunc(self: *@This(), interface: LocationInterface) !void {
      try Error.check(Api.compiler.underlying.ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc.?(
          apiCast(self),
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
        const Ptr = blk: {
          const Temp = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
          break :blk if (@typeInfo(Temp) == .pointer) Temp else *Temp;
        };
        const Sub = blk: {
          const Temp = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
          break :blk if (@typeInfo(Temp) == .pointer) @typeInfo(Temp).pointer.child else Temp;
        };

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
                return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), msg.ptr));
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
      loc_fn: @typeInfo(Api.c.OrtGetInitializerLocationFunc).optional.child,

      pub fn fromContext(instance: anytype) @This() {
        const T = @TypeOf(instance);
        const Ptr = blk: {
          const Temp = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
          break :blk if (@typeInfo(Temp) == .pointer) Temp else *Temp;
        };
        const Sub = blk: {
          const Temp = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
          break :blk if (@typeInfo(Temp) == .pointer) @typeInfo(Temp).pointer.child else Temp;
        };

        return .{
          .ptr = if (@bitSizeOf(Sub) == 0) null else @ptrCast(instance),
          .loc_fn = struct {
            fn wrapper(
              ctx: ?*anyopaque,
              name: [*c]const u8,
              val: ?*const Api.c.OrtValue,
              info: ?*const Api.c.OrtExternalInitializerInfo,
              out: [*c]?*Api.c.OrtExternalInitializerInfo,
            ) callconv(.c) ?*Api.c.OrtStatus {
              const self: Ptr = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @ptrCast(@alignCast(ctx.?));

              // Wrap call to Zig logic
              if (self.getLocation(cStrTo(name, [*:0]const u8), apiCastTo(val.?, *const Value), apiCastTo(info, ?*const ExternalInitializerInfo))) |maybe_new_info| {
                out.?.* = apiCast(maybe_new_info);
                return null;
              } else |err| {
                return Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err));
              }
            }
          }.wrapper,
        };
      }
    };

    /// Release the options object.
    pub fn deinit(self: *@This()) void {
      Api.compiler.underlying.ReleaseModelCompilationOptions.?(apiCast(self));
    }
  };
};

/// Session is usd for running inference on a model.
pub const Session = opaque {
  pub const Underlying = Api.c.OrtSession;
  /// Graph optimization level
  /// Refer to https://www.onnxruntime.ai/docs/performance/graph-optimizations.html#graph-optimization-levels for an in-depth understanding of the Graph Optimization Levels.
  pub const GraphOptimizationLevel = enum(Api.c.GraphOptimizationLevel) {
    NONE = @bitCast(Api.c.ORT_DISABLE_ALL),
    BASIC = @bitCast(Api.c.ORT_ENABLE_BASIC),
    EXTENDED = @bitCast(Api.c.ORT_ENABLE_EXTENDED),
    LAYOUT = @bitCast(Api.c.ORT_ENABLE_LAYOUT),
    ALL = @bitCast(Api.c.ORT_ENABLE_ALL),
  };

  /// Dictates if operators are executed one after another or in parallel where possible.
  pub const ExecutionMode = enum(Api.c.ExecutionMode) {
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
    profiling: ?Utils.Path = null,
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
      pub const Underlying = Api.c.OrtSessionOptions;
      pub fn init() !*@This() {
        var self: ?*@This() = null;
        try Error.check(Api.ort.CreateSessionOptions.?(apiCast(&self)));
        return self orelse error.OutOfMemory;
      }

      pub fn addFreeDimensionOverride(self: *@This(), denotation: [*:0]const u8, dim: i64) !void {
        try Error.check(Api.ort.AddFreeDimensionOverride.?(apiCast(self), denotation, dim));
      }

      /// Override symbolic dimensions by name.
      pub fn addFreeDimensionOverrideByName(self: *@This(), name: [*:0]const u8, dim: i64) !void {
        try Error.check(Api.ort.AddFreeDimensionOverrideByName.?(apiCast(self), cStr(name), dim));
      }

      pub fn clone(self: *const @This()) !*@This() {
        var retval: ?*@This() = null;
        try Error.check(Api.ort.CloneSessionOptions.?(apiCast(self), apiCast(&retval)));
        return retval orelse error.OutOfMemory;
      }

      pub fn deinit(self: *@This()) void {
        Api.ort.ReleaseSessionOptions.?(apiCast(self));
      }

      /// Enable Custom Operators (from onnxruntime-extensions).
      pub fn enableOrtCustomOps(self: *@This()) !void {
        try Error.check(Api.ort.EnableOrtCustomOps.?(apiCast(self)));
      }

      /// Replace initialized Tensors with external data.
      pub fn addExternalInitializers(self: *@This(), names: []const [*:0]const u8, values: []const *const Value) !void {
        std.debug.assert(names.len == values.len);
        try Error.check(Api.ort.AddExternalInitializers.?(
            apiCast(self),
            cStr(names.ptr),
            apiCast(values.ptr),
            names.len
        ));
      }

      /// Checks if the given session configuration entry exists.
      pub fn hasConfigEntry(self: *const @This(), key: [*:0]const u8) !bool {
        var out: c_int = 0;
        try Error.check(Api.ort.HasSessionConfigEntry.?(apiCast(self), key, &out));
        return out != 0;
      }

      pub fn _getConfigEntry(self: *const @This(), key: [*:0]const u8, out_ptr: [*:0]u8, out_len: *usize) !void {
        try Error.check(Api.ort.GetSessionConfigEntry.?(apiCast(self), cStr(key), cStr(out_ptr), out_len));
      }

      /// The returned size does NOT include the null terminator.
      pub fn getConfigEntryCount(self: *const C, key: [*:0]const u8) usize {
        var out: usize = 0;
        self._getConfigEntry(key, @ptrCast(@constCast(&empty_string)), &out) catch {};
        return if (out > 0) out - 1 else 0;
      }

      /// Get a session configuration value.
      /// allocator: Used to allocate the returned string.
      pub fn getConfigEntry(self: *const @This(), key: [*:0]const u8, out: [:0]u8) !void {
        var size: usize = out.len + 1;
        try self._getConfigEntry(key, out.ptr, &size);
        std.debug.assert(out.len + 1 == size);
      }

      /// Register custom ops using a registration function name.
      /// The library must be linked or loaded.
      pub fn registerCustomOpsUsingFunction(self: *@This(), function_name: [*:0]const u8) !void {
        try Error.check(Api.ort.RegisterCustomOpsUsingFunction.?(apiCast(self), cStr(function_name)));
      }

      /// Disable per-session thread pools (use global env thread pools).
      pub fn disablePerSessionThreads(self: *@This()) !void {
        try Error.check(Api.ort.DisablePerSessionThreads.?(apiCast(self)));
      }

      pub fn setOptimizationLevel(self: *@This(), level: GraphOptimizationLevel) !void {
        try Error.check(Api.ort.SetSessionGraphOptimizationLevel.?(apiCast(self), @intFromEnum(level)));
      }

      pub fn setOptimizedModelPath(self: *@This(), path: [*:0]const u8) !void {
        try Error.check(Api.ort.SetOptimizedModelFilePath.?(apiCast(self), path));
      }

      pub fn setExecutionMode(self: *@This(), mode: ExecutionMode) !void {
        try Error.check(Api.ort.SetSessionExecutionMode.?(apiCast(self), @intFromEnum(mode)));
      }

      pub fn setProfiling(self: *@This(), status: ?Utils.Path) !void {
        if (status) |str| {
          try Error.check(Api.ort.EnableProfiling.?(apiCast(self), pathCast(str)));
        } else {
          try Error.check(Api.ort.DisableProfiling.?(apiCast(self)));
        }
      }

      pub fn setMemoryPatternOptimization(self: *@This(), enabled: bool) !void {
        if (enabled) {
          try Error.check(Api.ort.EnableMemPattern.?(apiCast(self)));
        } else {
          try Error.check(Api.ort.DisableMemPattern.?(apiCast(self)));
        }
      }

      pub fn setCpuMemoryArena(self: *@This(), enabled: bool) !void {
        if (enabled) {
          try Error.check(Api.ort.EnableCpuMemArena.?(apiCast(self)));
        } else {
          try Error.check(Api.ort.DisableCpuMemArena.?(apiCast(self)));
        }
      }

      /// Set whether to use deterministic compute.
      pub fn setDeterministicCompute(self: *@This(), value: bool) !void {
        try Error.check(Api.ort.SetDeterministicCompute.?(apiCast(self), value));
      }

      /// Set user logging function.
      pub fn setUserLoggingFunction(self: *@This(), logging_interface: Logging.Interface) !void {
        try Error.check(Api.ort.SetUserLoggingFunction.?(
            apiCast(self),
            logging_interface.log_fn,
            logging_interface.ptr
        ));
      }

      pub fn setLogId(self: *@This(), id: [*:0]const u8) !void {
        try Error.check(Api.ort.SetSessionLogId.?(apiCast(self), cStr(id)));
      }

      pub fn setLogVerbosity(self: *@This(), level: Logging.Level) !void {
        try Error.check(Api.ort.SetSessionLogVerbosityLevel.?(apiCast(self), @bitCast(@intFromEnum(level))));
      }

      pub fn setLogSeverity(self: *@This(), level: Logging.Level) !void {
        try Error.check(Api.ort.SetSessionLogSeverityLevel.?(apiCast(self), @bitCast(@intFromEnum(level))));
      }

      pub fn setIntraOpThreads(self: *@This(), threads: c_int) !void {
        try Error.check(Api.ort.SetIntraOpNumThreads.?(apiCast(self), threads));
      }

      pub fn setInterOpThreads(self: *@This(), threads: c_int) !void {
        try Error.check(Api.ort.SetInterOpNumThreads.?(apiCast(self), threads));
      }

      pub fn addCustomOpDomain(self: *@This(), domain: *Op.Custom.Domain) !void {
        try Error.check(Api.ort.AddCustomOpDomain.?(apiCast(self), apiCast(domain)));
      }

      /// Wraps OrtApi::AddSessionConfigEntry
      pub fn addConfigEntry(self: *@This(), key: [*:0]const u8, value: [*:0]const u8) !void {
        try Error.check(Api.ort.AddSessionConfigEntry.?(apiCast(self), cStr(key), value));
      }

      /// Wraps OrtApi::AddInitializer
      /// Note: The lifetime of the OrtValue and the underlying buffer must outlive the session object
      pub fn addInitializer(self: *@This(), name: [*:0]const u8, val: *const Value) !void {
        try Error.check(Api.ort.AddInitializer.?(apiCast(self), cStr(name), apiCast(val)));
      }

      /// Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA_V2
      pub fn appendExecutionProviderCUDA(self: *@This(), options: *const ProviderOptions.CUDA) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_CUDA_V2.?(apiCast(self), apiCast(options)));
      }

      /// Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT_V2
      pub fn appendExecutionProviderTensorRT(self: *@This(), options: *const ProviderOptions.TensorRT) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_TensorRT_V2.?(apiCast(self), apiCast(options)));
      }

      /// Append Execution Providers (Generic)
      /// Options can be a struct or tuple with [:0]const u8 or [*:0]const u8 values
      pub fn appendExecutionProvider(self: *@This(), name: [:0]const u8, options: anytype) !void {
        const converted = Utils.createOptionsKVL(options, .cstr);
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider.?(apiCast(self), cStr(name.ptr), cStr(converted.keys()), cStr(converted.vals()), converted.len));
      }

      /// Loads a shared library (.dll on windows, .so on linux, etc) named 'library_name' and looks for this entry point:
      ///     OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
      /// It then passes in the provided session options to this function along with the api base.
      ///
      /// The handle to the loaded library is automatically released by ORT when the last OrtSession that references the
      /// library handle is released. If no OrtSession is created, then the library handle is released when the provided
      /// OrtSessionOptions is released.
      pub fn registerCustomOpsLibrary(self: *@This(), path: Utils.Path) !void {
        try Error.check(Api.ort.RegisterCustomOpsLibrary_V2.?(apiCast(self), pathCast(path)));
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
        var retval: ?*anyopaque = undefined;
        try Error.check(Api.ort.RegisterCustomOpsLibrary.?(apiCast(self), pathCast(path), @ptrCast(&retval))); // @ptrCast is needed because anyopaque has no underlying type
        return retval orelse error.OutOfMemory;
      }

      /// Replace initialized Tensors with external data from files in memory.
      ///
      /// allocator: Used to allocate the temporary C-compatible arrays required by the API.
      /// names: Slice of filenames.
      /// buffers: Slice of file contents.
      pub fn addExternalInitializersFromFilesInMemory(
        self: *@This(),
        names: []const Utils.Path,
        buffers: []const [*]u8,
        lengths: []const usize,
      ) !void {
        std.debug.assert(names.len == buffers.len);
        std.debug.assert(buffers.len == lengths.len);

        try Error.check(Api.ort.AddExternalInitializersFromFilesInMemory.?(
            apiCast(self),
            pathCast(names.ptr),
            cCast(buffers.ptr),
            cCast(lengths.ptr),
            names.len
        ));
      }

      pub const ExecutionProviderDevicePolicy = enum(Api.c.OrtExecutionProviderDevicePolicy) {
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
        try Error.check(Api.ort.SessionOptionsSetEpSelectionPolicy.?(apiCast(self), @intFromEnum(policy)));
      }

      /// Set the execution provider selection policy delegate for the session.
      pub fn setEpSelectionPolicyDelegate(self: *@This(), delegate: Api.c.EpSelectionDelegate, state: ?*anyopaque) !void {
        try Error.check(Api.ort.SessionOptionsSetEpSelectionPolicyDelegate.?(apiCast(self), delegate, state));
      }

      /// Set custom thread creation and join functions.
      pub fn customThreadingInterface(self: *@This(), interface: ThreadingOptions.ThreadingInterface) !void {
        try Error.check(Api.ort.SessionOptionsSetCustomCreateThreadFn.?(apiCast(self), interface.create_fn));
        if (interface.ptr) |p| try Error.check(Api.ort.SessionOptionsSetCustomThreadCreationOptions.?(apiCast(self), p));
        try Error.check(Api.ort.SessionOptionsSetCustomJoinThreadFn.?(apiCast(self), interface.join_fn));
      }

      /// Sets load cancellation flag to abort session loading process.
      pub fn setLoadCancellationFlag(self: *@This(), cancel: bool) !void {
        try Error.check(Api.ort.SessionOptionsSetLoadCancellationFlag.?(apiCast(self), cancel));
      }

      /// Get Session configuration entries.
      /// Returns a new KeyValuePairs instance that must be deinitialized.
      pub fn getConfigEntries(self: *const @This()) !*KeyValuePairs {
        var out: ?*KeyValuePairs = null;
        try Error.check(Api.ort.GetSessionOptionsConfigEntries.?(apiCast(self), apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Append the execution provider that is responsible for the selected OrtEpDevice instances.
      pub fn appendExecutionProviderV2(
        self: *@This(), 
        ep_devices: []const *const Ep.Device, 
        options: anytype
      ) !void {
        const converted = Utils.createOptionsKVL(options, .cstr);

        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_V2.?(
            apiCast(self),
            Api.env.underlying,
            apiCast(ep_devices.ptr),
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
            apiCast(self), keys, values, count
        ));
      }

      /// Append VitisAI provider.
      pub fn appendExecutionProviderVitisAI(self: *@This(), options: ?*const KeyValuePairs) !void {
        const keys: ?[*]const [*:0]const u8 = if (options) |o| o.getKeyValues()[0].ptr else null;
        const values: ?[*]const [*:0]const u8 = if (options) |o| o.getKeyValues()[1].ptr else null;
        const count = if (options) |o| o.getKeyValues()[0].len else 0;

        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_VitisAI.?(
            apiCast(self), keys, values, count
        ));
      }

      /// Append DNNL provider.
      pub fn appendExecutionProviderDnnl(self: *@This(), options: *const Api.c.OrtDnnlProviderOptions) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_Dnnl.?(apiCast(self), options));
      }

      /// Append CANN provider.
      pub fn appendExecutionProviderCANN(self: *@This(), options: *const Api.c.OrtCANNProviderOptions) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_CANN.?(apiCast(self), options));
      }

      /// Legacy: Append CUDA provider.
      pub fn appendExecutionProviderCUDALegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_CUDA.?(apiCast(self), device_id));
      }

      /// Legacy: Append ROCM provider.
      pub fn appendExecutionProviderROCMLegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_ROCM.?(apiCast(self), device_id));
      }

      /// Legacy: Append TensorRT provider.
      pub fn appendExecutionProviderTensorRTLegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_TensorRT.?(apiCast(self), device_id));
      }

      /// Legacy: Append MIGraphX provider.
      pub fn appendExecutionProviderMIGraphXLegacy(self: *@This(), device_id: c_int) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_MIGraphX.?(apiCast(self), device_id));
      }

      /// Legacy: Append OpenVINO provider (Struct Options).
      pub fn appendExecutionProviderOpenVINOLegacy(self: *@This(), options: *const Api.c.OrtOpenVINOProviderOptions) !void {
        try Error.check(Api.ort.SessionOptionsAppendExecutionProvider_OpenVINO.?(apiCast(self), options));
      }
    };
  };

  pub fn initZ(path: Utils.Path, options: *const Options.C) !*@This() {
    var self: ?*@This() = null;
    try Error.check(Api.ort.CreateSession.?(
      Api.env.underlying,
      pathCast(path),
      apiCast(options),
      apiCast(&self),
    ));
    return self orelse error.OutOfMemory;
  }

  pub fn initSlice(data: []const u8, options: *const Options.C) !*@This() {
    var self: ?*@This() = null;
    try Error.check(Api.ort.CreateSessionFromArray.?(
      Api.env.underlying,
      @as([*c]const u8, @ptrCast(data.ptr)), // @ptrCast is needed because u8 has no underlying type
      data.len,
      apiCast(options),
      apiCast(&self),
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
        pathCast(model_path),
        apiCast(options),
        apiCast(container),
        apiCast(&out)
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
        @as([*c]const u8, @ptrCast(model_data.ptr)), // @ptrCast is needed because u8 has no underlying type
        model_data.len,
        apiCast(options),
        apiCast(container),
        apiCast(&out)
    ));
    return out orelse error.OutOfMemory;
  }

  /// Set DynamicOptions for EPs.
  pub fn setEpDynamicOptions(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
    std.debug.assert(keys.len == values.len);
    try Error.check(Api.ort.SetEpDynamicOptions.?(
        apiCast(self),
        cStr(keys.ptr),
        cStr(values.ptr),
        keys.len
    ));
  }

  pub fn getInputCount(self: *const @This()) !usize {
    var retval: usize = undefined;
    try Error.check(Api.ort.SessionGetInputCount.?(apiCast(self), &retval));
    return retval;
  }

  pub fn getOutputCount(self: *const @This()) !usize {
    var retval: usize = undefined;
    try Error.check(Api.ort.SessionGetOutputCount.?(apiCast(self), &retval));
    return retval;
  }

  pub fn overridableInitializerCount(self: *const @This()) !usize {
    var retval: usize = undefined;
    try Error.check(Api.ort.SessionGetOverridableInitializerCount.?(apiCast(self), &retval));
    return retval;
  }

  pub fn inputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.SessionGetInputTypeInfo.?(apiCast(self), index, apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn outputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.SessionGetOutputTypeInfo.?(apiCast(self), index, apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn overridableInitializerTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
    var out: ?*TypeInfo = null;
    try Error.check(Api.ort.SessionGetOverridableInitializerTypeInfo.?(apiCast(self), index, apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn inputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionGetInputName.?(apiCast(self), index, apiCast(allocator), cStr(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn outputName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionGetOutputName.?(apiCast(self), index, apiCast(allocator), cStr(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn overridableInitializerName(self: *const @This(), index: usize, allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionGetOverridableInitializerName.?(apiCast(self), index, apiCast(allocator), cStr(&out)));
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
    output_values: []?*Value,
  ) !void {
    std.debug.assert(input_names.len == input_values.len);
    std.debug.assert(output_names.len == output_values.len);
    try Error.check(Api.ort.Run.?(
      apiCast(self),
      apiCast(run_options),
      cStr(input_names.ptr),
      apiCast(input_values.ptr),
      input_values.len,
      cStr(output_names.ptr),
      output_values.len,
      apiCast(output_values.ptr),
    ));
  }

  /// Run the model asynchronously
  /// callback_ctx has a function `callback(ctx: *@TypeOf(callback_ctx), []?*Value, ?*Error.Status)`
  ///
  /// WARNING: the `callback_ctx_ptr` should be static or pinned in memory until the end of the program.
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

    const output_names_ptr: [*]const ?[*:0]const u8 = @ptrCast(output_names.ptr);
    const outputs_ptr: [*]?*Value = @ptrCast(outputs.ptr);

    try Error.check(Api.ort.RunAsync.?(
        apiCast(self),
        apiCast(run_options),
        cStr(input_names.ptr),
        apiCast(inputs.ptr),
        inputs.len,
        cStr(output_names_ptr),
        output_names.len,
        apiCast(outputs_ptr),
        &struct {pub fn callback(ctx: ?*anyopaque, vptr: [*c]?*Api.c.OrtValue, vlen: usize, status: Api.c.OrtStatusPtr) callconv(.c) void {
          @as(@TypeOf(callback_ctx_ptr), @alignCast(@ptrCast(ctx))).callback(apiCastTo(vptr, [*]?*Value)[0 .. vlen], apiCastTo(status, ?*Error.Status));
        }}.callback,
        @ptrCast(callback_ctx_ptr),
    ));
  }

  /// Run the model using IoBinding. Wraps OrtApi::RunWithBinding
  pub fn runWithBinding(self: *@This(), run_options: ?*const RunOptions, binding: *const IoBinding) !void {
    try Error.check(Api.ort.RunWithBinding.?(apiCast(self), apiCast(run_options), apiCast(binding)));
  }

  /// End profiling and return a copy of the profiling file name.
  /// The returned string is allocated using the provided allocator and must be freed by the caller.
  pub fn endProfiling(self: *@This(), allocator: *Allocator) ![*:0]u8 {
    var out: ?[*:0]u8 = null;
    try Error.check(Api.ort.SessionEndProfiling.?(apiCast(self), apiCast(allocator), cStr(&out)));
    return out orelse error.OutOfMemory;
  }

  /// Get profiling start time in nanoseconds
  pub fn getProfilingStartTimeNs(self: *const @This()) !u64 {
    var out: u64 = 0;
    try Error.check(Api.ort.SessionGetProfilingStartTimeNs.?(apiCast(self), &out));
    return out;
  }

  /// Get model metadata
  pub fn getModelMetadata(self: *const @This()) !*Model.Metadata {
    var out: ?*Model.Metadata = null;
    try Error.check(Api.ort.SessionGetModelMetadata.?(apiCast(self), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn getMemoryInfoForInputs(self: *const @This(), out: []*const Allocator.MemoryInfo) !void {
    const out_ptr: [*]?*const Allocator.MemoryInfo = @ptrCast(out.ptr);
    try Error.check(Api.ort.SessionGetMemoryInfoForInputs.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  pub fn getMemoryInfoForOutputs(self: *const @This(), out: []*const Allocator.MemoryInfo) !void {
    const out_ptr: [*]?*const Allocator.MemoryInfo = @ptrCast(out.ptr);
    try Error.check(Api.ort.SessionGetMemoryInfoForOutputs.?(apiCast(self), apiCast(out_ptr), out.len));
  }

  pub fn getEpDeviceForInputs(self: *const @This(), out: []?*const Ep.Device) !void {
    try Error.check(Api.ort.SessionGetEpDeviceForInputs.?(apiCast(self), apiCast(out.ptr), out.len));
  }

  /// Release the Session object.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseSession.?(apiCast(self));
  }
};

/// These are used when creating a Session to dectate it's behavior.
pub const RunOptions = opaque {
  pub const Underlying = Api.c.OrtRunOptions;
  pub fn init() !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateRunOptions.?(apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn setLogVerbosityLevel(self: *@This(), level: c_int) !void {
    try Error.check(Api.ort.RunOptionsSetRunLogVerbosityLevel.?(apiCast(self), level));
  }

  pub fn setLogSeverityLevel(self: *@This(), level: Logging.Level) !void {
    try Error.check(Api.ort.RunOptionsSetRunLogSeverityLevel.?(apiCast(self), @bitCast(@intFromEnum(level))));
  }

  pub fn setRunTag(self: *@This(), tag: [*:0]const u8) !void {
    try Error.check(Api.ort.RunOptionsSetRunTag.?(apiCast(self), cStr(tag)));
  }

  pub fn getLogVerbosityLevel(self: *const @This()) !c_int {
    var level: c_int = 0;
    try Error.check(Api.ort.RunOptionsGetRunLogVerbosityLevel.?(apiCast(self), &level));
    return level;
  }

  pub fn getLogSeverityLevel(self: *const @This()) !Logging.Level {
    var level: c_int = 0;
    try Error.check(Api.ort.RunOptionsGetRunLogSeverityLevel.?(apiCast(self), &level));
    return @enumFromInt(level);
  }

  pub fn getRunTag(self: *const @This()) !?[*:0]const u8 {
    var tag: ?[*:0]const u8 = null;
    try Error.check(Api.ort.RunOptionsGetRunTag.?(apiCast(self), cStr(&tag)));
    return tag;
  }

  pub fn setTerminate(self: *@This()) !void {
    try Error.check(Api.ort.RunOptionsSetTerminate.?(apiCast(self)));
  }

  pub fn unsetTerminate(self: *@This()) !void {
    try Error.check(Api.ort.RunOptionsUnsetTerminate.?(apiCast(self)));
  }

  /// Wrapper around ::OrtLoraAdapter
  /// Holds a set of LoRA Parameters loaded from a single file.
  ///
  /// LoRA (Low-Rank Adaptation) allow you to modify the behavior of a base model without retraining it.
  /// This allows you to hot-swap fine-tuned weights during inference
  pub const LoraAdapter = opaque {
    pub const Underlying = Api.c.OrtLoraAdapter;
    /// Wraps OrtApi::CreateLoraAdapter
    /// adapter_path: Path to the LoRA adapter file.
    /// allocator: Optional allocator. If null, data stays on CPU until inference requires it on device (after which it is copied).
    pub fn init(adapter_path: Utils.Path, allocator: ?*Allocator) !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateLoraAdapter.?(pathCast(adapter_path), apiCast(allocator), apiCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::CreateLoraAdapterFromArray
    /// bytes: In-memory buffer of the LoRA adapter.
    /// allocator: Optional allocator. If null, data stays on CPU until inference requires it on device (after which it is copied).
    pub fn initFromArray(bytes: []const u8, allocator: ?*Allocator) !*@This() {
      var self: ?*@This() = null;
      try Error.check(Api.ort.CreateLoraAdapterFromArray.?(bytes.ptr, bytes.len, apiCast(allocator), apiCast(&self)));
      return self orelse error.OutOfMemory;
    }

    /// Release the LoraAdapter.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseLoraAdapter.?(apiCast(self));
    }
  };

  pub fn addActiveLoraAdapter(self: *@This(), adapter: *const LoraAdapter) !void {
    try Error.check(Api.ort.RunOptionsAddActiveLoraAdapter.?(apiCast(self), apiCast(adapter)));
  }

  pub fn addConfigEntry(self: *@This(), key: [*:0]const u8, value: [*:0]const u8) !void {
    try Error.check(Api.ort.AddRunConfigEntry.?(apiCast(self), key, value));
  }

  pub fn getConfigEntry(self: *const @This(), key: [*:0]const u8) ?[*:0]const u8 {
    return Api.ort.GetRunConfigEntry.?(apiCast(self), key);
  }

  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseRunOptions.?(apiCast(self));
  }
};

/// Useful for zero-copy inference (better performance)
pub const IoBinding = opaque {
  pub const Underlying = Api.c.OrtIoBinding;
  pub fn init(session: *Session) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateIoBinding.?(apiCast(session), apiCast(&out)));
    return out orelse error.OutOfMemory;
  }

  pub fn bindInput(self: *@This(), name: [*:0]const u8, value: *const Value) !void {
    try Error.check(Api.ort.BindInput.?(apiCast(self), cStr(name), apiCast(value)));
  }

  pub fn bindOutput(self: *@This(), name: [*:0]const u8, value: *const Value) !void {
    try Error.check(Api.ort.BindOutput.?(apiCast(self), cStr(name), apiCast(value)));
  }

  pub fn bindOutputToDevice(self: *@This(), name: [*:0]const u8, mem_info: *const Allocator.MemoryInfo) !void {
    try Error.check(Api.ort.BindOutputToDevice.?(apiCast(self), cStr(name), apiCast(mem_info)));
  }

  /// Returns error.Unbounded if `self` has no bound outputs
  pub fn getBoundOutputNames(self: *const @This(), allocator: *Allocator) !struct {
    names: [*]u8,
    lengths: []usize,

    pub fn deinit(self_: *@This(), allocator_: *Allocator) void {
      allocator_.free(self_.names);
      allocator_.free(self_.lengths);
    }
  } {
    var buffer: ?[*]u8 = null;
    var lengths_ptr: ?[*]usize = null;
    var count: usize = 0;
    try Error.check(Api.ort.GetBoundOutputNames.?(
        apiCast(self),
        apiCast(allocator),
        @as([*c][*c]u8, @ptrCast(&buffer)),
        @as([*c][*c]usize, @ptrCast(&lengths_ptr)),
        &count,
    ));

    if (buffer == null or lengths_ptr == null or count == 0) {
      if (buffer) |b| allocator.free(b);
      if (lengths_ptr) |l| allocator.free(l);
      return error.Unbounded;
    }
    return .{ .names = buffer.?, .lengths = lengths_ptr.?[0 .. count] };
  }

  // Returns null if length is 0
  pub fn getBoundOutputValues(self: *const @This(), allocator: *Allocator) !?[]?*Value {
    var output_ptr: ?[*]?*Value = null;
    var count: usize = 0;
    try Error.check(Api.ort.GetBoundOutputValues.?(apiCast(self), apiCast(allocator), apiCast(&output_ptr), &count));
    return (output_ptr orelse return null)[0..count];
  }

  pub fn clearBoundInputs(self: *@This()) void {
    Api.ort.ClearBoundInputs.?(apiCast(self));
  }

  pub fn clearBoundOutputs(self: *@This()) void {
    Api.ort.ClearBoundOutputs.?(apiCast(self));
  }

  pub fn synchronizeInputs(self: *@This()) !void {
    try Error.check(Api.ort.SynchronizeBoundInputs.?(apiCast(self)));
  }

  pub fn synchronizeOutputs(self: *@This()) !void {
    try Error.check(Api.ort.SynchronizeBoundOutputs.?(apiCast(self)));
  }

  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseIoBinding.?(apiCast(self));
  }
};

/// Represents a physical hardware device (CPU, GPU, NPU) available on the system.
/// This is an opaque type managed by ORT.
pub const HardwareDevice = opaque {
  pub const Underlying = Api.c.OrtHardwareDevice;
  /// Maps to OrtHardwareDeviceType in the C API.
  pub const Type = enum(Api.c.OrtHardwareDeviceType) {
    CPU = @bitCast(Api.c.OrtHardwareDeviceType_CPU),
    GPU = @bitCast(Api.c.OrtHardwareDeviceType_GPU),
    NPU = @bitCast(Api.c.OrtHardwareDeviceType_NPU),
  };

  /// Returns the type of hardware (CPU, GPU, or NPU).
  /// Wraps OrtApi::HardwareDevice_Type
  pub fn getType(self: *const @This()) Type {
    return @enumFromInt(Api.ort.HardwareDevice_Type.?(apiCast(self)));
  }

  /// Returns the hardware device's vendor name as a null-terminated string. DO NOT FREE
  /// Wraps OrtApi::HardwareDevice_Vendor
  pub fn getVendor(self: *const @This()) [*:0]const u8 {
    return cStrTo(Api.ort.HardwareDevice_Vendor.?(apiCast(self)), [*:0]const u8);
  }

  /// Returns the hardware device's vendor identifier (e.g., PCI Vendor ID).
  /// Wraps OrtApi::HardwareDevice_VendorId
  pub fn getVendorId(self: *const @This()) u32 {
    return Api.ort.HardwareDevice_VendorId.?(apiCast(self));
  }

  /// Returns the hardware device's unique identifier.
  /// Note: This identifies the specific hardware instance when combined with vendor id.
  /// Wraps OrtApi::HardwareDevice_DeviceId
  pub fn getDeviceId(self: *const @This()) u32 {
    return Api.ort.HardwareDevice_DeviceId.?(apiCast(self));
  }

  /// Returns an OrtKeyValuePairs instance containing additional metadata for the device.
  /// Note: ORT owns this instance; do NOT call deinit/ReleaseKeyValuePairs on it.
  /// Wraps OrtApi::HardwareDevice_Metadata
  pub fn getMetadata(self: *const @This()) *const KeyValuePairs {
    // returns null iff the input device is null
    return apiCastTo(Api.ort.HardwareDevice_Metadata.?(apiCast(self)).?, *const KeyValuePairs);
  }
};

pub const Op = opaque {
  pub const Underlying = Api.c.OrtOp;

  /// computeV2 function receives this context
  pub const KernelContext = opaque {
    pub const Underlying = Api.c.OrtKernelContext;
    pub fn getInputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.KernelContext_GetInputCount.?(apiCast(self), &out));
      return out;
    }

    pub fn getOutputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.KernelContext_GetOutputCount.?(apiCast(self), &out));
      return out;
    }

    pub fn getInput(self: *const @This(), index: usize) !?*const Value {
      var out: ?*const Value = null;
      try Error.check(Api.ort.KernelContext_GetInput.?(apiCast(self), index, apiCast(&out)));
      return out;
    }

    /// DO NOT FREE THE RETURNED VALUE
    pub fn getOutput(self: *@This(), index: usize, dims: []const i64) !?*Value {
      var out: ?*Value = null;
      try Error.check(Api.ort.KernelContext_GetOutput.?(apiCast(self), index, dims.ptr, dims.len, apiCast(&out)));
      return out;
    }

    pub fn getGpuComputeStream(self: *const @This()) !?*anyopaque {
      var out: ?*anyopaque = null;
      try Error.check(Api.ort.KernelContext_GetGPUComputeStream.?(apiCast(self), &out));
      return out;
    }
    
    pub fn getResource(self: *const @This(), version: c_int, id: c_int) !?*anyopaque {
      var out: ?*anyopaque = null;
      try Error.check(Api.ort.KernelContext_GetResource.?(apiCast(self), version, id, &out));
      return out;
    }
    
    pub fn getScratchBuffer(self: *const @This(), mem_info: *const Allocator.MemoryInfo, size: usize) !?*anyopaque {
      var out: ?*anyopaque = null;
      try Error.check(Api.ort.KernelContext_GetScratchBuffer.?(apiCast(self), apiCast(mem_info), size, &out));
      return out;
    }
    
    pub fn getAllocator(self: *const @This(), mem_info: *const Allocator.MemoryInfo) !*Allocator {
      var out: ?*Allocator = null;
      try Error.check(Api.ort.KernelContext_GetAllocator.?(apiCast(self), apiCast(mem_info), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Get the runtime logger.
    pub fn getLogger(self: *const @This()) !*const KernelInfo.Logger {
      var out: ?*const KernelInfo.Logger = null;
      try Error.check(Api.ort.KernelContext_GetLogger.?(apiCast(self), apiCast(&out)));
      return out.?;
    }

    /// Run a function in parallel.
    /// ctx must be a struct with a `run(self: *Self, index: usize) !void` method.
    pub fn parallelFor(
      self: *const @This(), 
      total: usize, 
      num_batch: usize, 
      ctx_ptr: anytype
    ) !void {
      try Error.check(Api.ort.KernelContext_ParallelFor.?(apiCast(self), &struct {
        const Sub = @typeInfo(@TypeOf(ctx_ptr)).pointer.child;
        fn wrapper(ctx: ?*anyopaque, index: usize) callconv(.c) void {
          const self_: @TypeOf(ctx_ptr) = if (@bitSizeOf(Sub) == 0) @constCast(&Sub{}) else @ptrCast(@alignCast(ctx.?));
          self_.run(index);
        }
      }.wrapper, total, num_batch, @ptrCast(ctx_ptr)));
    }
  };

  /// context for Initialization phase of a custom operator
  pub const KernelInfo = opaque {
    pub const Underlying = Api.c.OrtKernelInfo;
    pub fn copy(self: *const @This()) !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CopyKernelInfo.?(apiCast(self), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn _getNodeName(self: *const @This(), out_ptr: [*:0]u8, out_size: *usize) !void {
      try Error.check(Api.ort.KernelInfo_GetNodeName.?(apiCast(self), cStr(out_ptr), out_size));
    }

    /// The returned size does NOT include the null terminator.
    pub fn getNodeNameCount(self: *const @This()) !usize {
      var out: usize = 0;
      self._getNodeName(@ptrCast(@constCast(&empty_string)), &out) catch {};
      return if (out > 0) out - 1 else 0;
    }

    pub fn getNodeName(self: *const @This(), out: [:0]u8) !void {
      var size: usize = out.len + 1;
      try self._getNodeName(out.ptr, &size);
      std.debug.assert(out.len + 1 == size);
    }
    
    pub fn getAttributeFloat(self: *const @This(), name: [*:0]const u8) !f32 {
      var out: f32 = 0;
      try Error.check(Api.ort.KernelInfoGetAttribute_float.?(apiCast(self), cStr(name), &out));
      return out;
    }
    
    pub fn getAttributeInt(self: *const @This(), name: [*:0]const u8) !i64 {
      var out: i64 = 0;
      try Error.check(Api.ort.KernelInfoGetAttribute_int64.?(apiCast(self), cStr(name), &out));
      return out;
    }
    
    pub fn _getAttributeString(self: *const @This(), name: [*:0]const u8, out_ptr: [*:0]u8, out_len: *usize) !void {
      try Error.check(Api.ort.KernelInfoGetAttribute_string.?(apiCast(self), cStr(name), cStr(out_ptr), out_len));
    }

    /// The returned size does NOT include the null terminator.
    pub fn getAttributeStringCount(self: *const @This(), name: [*:0]const u8) usize {
      var out: usize = 0;
      self._getAttributeString(name, @ptrCast(@constCast(&empty_string)), &out) catch {};
      return if (out > 0) out - 1 else 0;
    }

    pub fn getAttributeString(self: *const @This(), name: [*:0]const u8, out: [:0]u8) !void {
      var size: usize = out.len + 1;
      try self._getAttributeString(cStr(name), cStr(out.ptr), &size);
      std.debug.assert(out.len + 1 == size);
    }
    
    pub fn getAttributeTensor(self: *const @This(), name: [*:0]const u8, allocator: *Allocator) !*Value {
      var out: ?*Value = null;
      try Error.check(Api.ort.KernelInfoGetAttribute_tensor.?(apiCast(self), cStr(name), apiCast(allocator), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Get the number of inputs.
    pub fn getInputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.KernelInfo_GetInputCount.?(apiCast(self), &out));
      return out;
    }

    /// Get the number of outputs.
    pub fn getOutputCount(self: *const @This()) !usize {
      var out: usize = 0;
      try Error.check(Api.ort.KernelInfo_GetOutputCount.?(apiCast(self), &out));
      return out;
    }

    pub fn _getInputName(self: *const @This(), index: usize, out_ptr: [*:0]u8, out_len: *usize) !void {
      try Error.check(Api.ort.KernelInfo_GetInputName.?(apiCast(self), index, cStr(out_ptr), out_len));
    }

    /// The returned size does NOT include the null terminator.
    pub fn getInputNameCount(self: *const @This(), index: usize) !usize {
      var out: usize = 0;
      self._getInputName(index, @ptrCast(@constCast(&empty_string)), &out) catch {};
      return if (out > 0) out - 1 else 0;
    }

    /// Get the name of an input.
    /// allocator: The allocator used to allocate the returned string buffer.
    /// Returns a null-terminated string.
    pub fn getInputName(self: *const @This(), index: usize, out: [:0]u8) !void {
      var size: usize = out.len + 1;
      try self._getInputName(index, cStr(out.ptr), &size);
      std.debug.assert(out.len + 1 == size);
    }

    pub fn _getOutputName(self: *const @This(), index: usize, out_ptr: [*:0]u8, out_len: *usize) !void {
      try Error.check(Api.ort.KernelInfo_GetOutputName.?(apiCast(self), index, cStr(out_ptr), out_len));
    }

    /// The returned size does NOT include the null terminator.
    pub fn getOutputNameCount(self: *const @This(), index: usize) !usize {
      var out: usize = 0;
      self._getOutputName(index, @ptrCast(@constCast(&empty_string)), &out) catch {};
      return if (out > 0) out - 1 else 0;
    }

    /// Get the name of an output.
    /// allocator: The allocator used to allocate the returned string buffer.
    /// Returns a null-terminated string.
    pub fn getOutputName(self: *const @This(), index: usize, out: [:0]u8) !void {
      var size: usize = out.len + 1;
      try self._getOutputName(index, cStr(out.ptr), &size);
      std.debug.assert(out.len + 1 == size);
    }

    /// Get the type information for an input.
    /// The returned TypeInfo must be deinitialized by the caller.
    pub fn getInputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.KernelInfo_GetInputTypeInfo.?(apiCast(self), index, apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Get the type information for an output.
    /// The returned TypeInfo must be deinitialized by the caller.
    pub fn getOutputTypeInfo(self: *const @This(), index: usize) !*TypeInfo {
      var out: ?*TypeInfo = null;
      try Error.check(Api.ort.KernelInfo_GetOutputTypeInfo.?(apiCast(self), index, apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub fn _getAttributeArrayFloat(self: *const @This(), name: [*:0]const u8, out_ptr: [*]f32, out_len: *usize) !void {
      try Error.check(Api.ort.KernelInfoGetAttributeArray_float.?(apiCast(self), cStr(name), out_ptr, out_len));
    }

    pub fn getAttributeArrayFloatCount(self: *const @This(), name: [*:0]const u8) !usize {
      var out: usize = 0;
      self._getAttributeArrayFloat(cStr(name), @ptrCast(@constCast(&[_]f32{})), &out) catch {};
      return out;
    }

    /// Fetch a float array stored as an attribute.
    /// allocator: The allocator used to create the slice to hold the result.
    pub fn getAttributeArrayFloat(self: *const @This(), name: [*:0]const u8, out: []f32) !void {
      var size: usize = out.len;
      try self._getAttributeArrayFloat(cStr(name), out.ptr, &size);
      std.debug.assert(out.len == size);
    }

    pub fn _getAttributeArrayInt(self: *const @This(), name: [*:0]const u8, out_ptr: [*]i64, out_len: *usize) !void {
      try Error.check(Api.ort.KernelInfoGetAttributeArray_int64.?(apiCast(self), cStr(name), cCast(out_ptr), out_len));
    }

    pub fn getAttributeArrayIntCount(self: *const @This(), name: [*:0]const u8) !usize {
      var out: usize = 0;
      self._getAttributeArrayInt(cStr(name), @ptrCast(@constCast(&[_]i64{})), &out) catch {};
      return out;
    }

    /// Fetch an int64 array stored as an attribute.
    /// allocator: The allocator used to create the slice to hold the result.
    pub fn getAttributeArrayInt(self: *const @This(), name: [*:0]const u8, out: []i64) !void {
      var size: usize = 0;
      try self._getAttributeArrayInt(cStr(name), out.ptr, &size);
      std.debug.assert(out.len == size);
    }

    /// Get allocator from KernelInfo for a specific memory type. 
    /// Please use `Allocator.deinit` (C API ReleaseAllocator) to release the returned object.
    pub fn getAllocator(self: *const @This(), mem_type: Allocator.MemoryType) !*Allocator {
      var out: ?*Allocator = null;
      try Error.check(Api.ort.KernelInfoGetAllocator.?(apiCast(self), @intFromEnum(mem_type), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    pub const Logger = opaque {
      pub const Underlying = Api.c.OrtLogger;
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
            apiCast(self), 
            @intFromEnum(severity), 
            message, 
            pathCast(file), 
            line, 
            func
        ));
      }

      /// Get the logging severity level.
      pub fn getSeverityLevel(self: *const @This()) !Logging.Level {
        var out: Api.c.OrtLoggingLevel = undefined;
        try Error.check(Api.ort.Logger_GetLoggingSeverityLevel.?(apiCast(self), &out));
        return @enumFromInt(out);
      }
    };

    pub fn getLogger(self: *const @This()) !*const Logger {
      var out: ?*const Logger = null;
      try Error.check(Api.ort.KernelInfo_GetLogger.?(apiCast(self), apiCast(&out)));
      return out.?;
    }

    /// Get a constant input tensor.
    /// is_constant: Output bool indicating if it is constant.
    pub fn getConstantInputTensor(self: *const @This(), index: usize, is_constant: *bool) !?*const Value {
      var out: ?*const Value = null;
      var is_const_c: c_int = 0;
      try Error.check(Api.ort.KernelInfoGetConstantInput_tensor.?(apiCast(self), index, &is_const_c, apiCast(&out)));
      is_constant.* = (is_const_c != 0);
      return out;
    }

    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseKernelInfo.?(apiCast(self));
    }
  };

  /// Create onnxruntime native operator.
  pub fn init(
    info: *const KernelInfo,
    op_name: [*:0]const u8,
    domain: [*:0]const u8,
    version: c_int,
    type_constraint_names: [][*:0]const u8,
    type_constraint_values: []const Value.Sub.Tensor.ElementDataType,
    attrs: []const *const Op.Attr,
    input_count: c_int,
    output_count: c_int
  ) !*@This() {
    std.debug.assert(type_constraint_names.len == type_constraint_values.len);
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateOp.?(
      apiCast(info),
      cStr(op_name),
      cStr(domain),
      version,
      cStr(type_constraint_names.ptr),
      apiCast(type_constraint_values.ptr),
      @intCast(type_constraint_names.len),
      apiCast(attrs.ptr),
      @intCast(attrs.len),
      input_count,
      output_count,
      apiCast(&out)
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
      apiCast(context),
      apiCast(self),
      apiCast(inputs.ptr),
      @intCast(inputs.len),
      apiCast(outputs.ptr),
      @intCast(outputs.len)
    ));
  }

  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseOp.?(apiCast(self));
  }

  pub const Attr = opaque {
    pub const Underlying = Api.c.OrtOpAttr;
    /// Defines the type of data stored in an Operator Attribute.
    pub const Type = enum(Api.c.OrtOpAttrType) {
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
        cStr(name), 
        data, 
        len, 
        @intFromEnum(type_), 
        apiCast(&out)
      ));
      return out orelse error.OutOfMemory;
    }

    // Helpers for common types
    
    pub fn initInt(name: [*:0]const u8, val: i64) !*@This() {
      return @This().init(name, &val, 1, .INT);
    }

    pub fn initFloat(name: [*:0]const u8, val: f32) !*@This() {
      return @This().init(name, &val, 1, .FLOAT);
    }

    pub fn initString(name: [*:0]const u8, val: []const u8) !*@This() {
      return @This().init(name, val.ptr, @intCast(val.len), .STRING);
    }

    pub fn initInts(name: [*:0]const u8, val: []const i64) !*@This() {
      return @This().init(name, val.ptr, @intCast(val.len), .INTS);
    }

    pub fn initFloats(name: [*:0]const u8, val: []const f32) !*@This() {
      return @This().init(name, val.ptr, @intCast(val.len), .FLOATS);
    }

    /// Get the attribute name.
    pub fn getName(self: *const @This()) ![*:0]const u8 {
      var out: ?[*:0]const u8 = null;
      try Error.check(Api.ort.OpAttr_GetName.?(apiCast(self), cStr(&out)));
      return out orelse "";
    }

    /// Get the attribute type.
    pub fn getType(self: *const @This()) !Type {
      var out: Type = undefined;
      try Error.check(Api.ort.OpAttr_GetType.?(apiCast(self), apiCast(&out)));
      return out;
    }

    /// Get the 'TENSOR' attribute as an OrtValue.
    /// Returns a new Value that must be deinitialized.
    pub fn getTensor(self: *const @This()) !*Value {
      var out: ?*Value = null;
      try Error.check(Api.ort.OpAttr_GetTensorAttributeAsOrtValue.?(apiCast(self), apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Read contents of an attribute to data.
    /// out_data: Buffer to write data to.
    /// Returns the number of bytes written or required.
    pub fn read(self: *const @This(), type_: Type, out_data: []u8) !usize {
      var size: usize = 0;
      try Error.check(Api.ort.ReadOpAttr.?(
          apiCast(self), 
          @intFromEnum(type_), 
          cCast(out_data.ptr), 
          out_data.len, 
          &size
      ));
      return size;
    }

    /// Release the attribute.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseOpAttr.?(apiCast(self));
    }
  };

  pub const Custom = struct {
    pub const Underlying = Api.c.OrtCustomOp;
    underlying: @This().Underlying,
    comptime { std.debug.assert(@bitSizeOf(@This()) == @bitSizeOf(@This().Underlying)); }

    pub const InputOutputCharacteristic = enum(Api.c.OrtCustomOpInputOutputCharacteristic) {
      Required = @bitCast(Api.c.INPUT_OUTPUT_REQUIRED),
      Optional = @bitCast(Api.c.INPUT_OUTPUT_OPTIONAL),
      Variadic = @bitCast(Api.c.INPUT_OUTPUT_VARIADIC),
    };

    /// Initialize a CustomOp structure with vtables pointing to the provided Implementer type.
    ///
    /// Requirements for Implementer (T):
    /// - Must have a field `ort_op: CustomOp` for @fieldParentPtr navigation.
    ///
    /// REQUIRED (Instance Methods) - C passes the 'op' handle; T uses @fieldParentPtr.
    /// - fn getName(self: *const T) [*:0]const u8
    /// - fn getExecutionProviderType(self: *const T) ?[*:0]const u8
    /// - fn getInputType(self: *const T, index: usize) Value.Tensor.ElementDataType
    /// - fn getInputTypeCount(self: *const T) usize
    /// - fn getOutputType(self: *const T, index: usize) Value.Tensor.ElementDataType
    /// - fn getOutputTypeCount(self: *const T) usize
    /// - fn createKernelV2(self: *const T, api: *const Api.ort, info: *const KernelInfo) !*anyopaque
    ///
    /// REQUIRED (Static Methods) - C does NOT pass the 'op' handle; T must implement as static.
    /// - fn computeV2(kernel_state: *anyopaque, context: *KernelContext) !void
    /// - fn destroyKernel(kernel_state: *anyopaque) void
    ///
    /// OPTIONAL (Instance Methods) - Configuration and Schema.
    /// - fn getInputCharacteristic(self: *const T, index: usize) InputOutputCharacteristic
    /// - fn getOutputCharacteristic(self: *const T, index: usize) InputOutputCharacteristic
    /// - fn getInputMemoryType(self: *const T, index: usize) Allocator.MemoryType
    /// - fn getVariadicInputMinArity(self: *const T) i32
    /// - fn getVariadicInputHomogeneity(self: *const T) bool
    /// - fn getVariadicOutputMinArity(self: *const T) i32
    /// - fn getVariadicOutputHomogeneity(self: *const T) bool
    /// - fn inferOutputShape(self: *const T, context: *ShapeInferContext) !void
    /// - fn getStartVersion(self: *const T) i32
    /// - fn getEndVersion(self: *const T) i32
    /// - fn createKernelV1(self: *const T, api: *const Api.ort, info: *const KernelInfo) !*anyopaque
    ///
    /// OPTIONAL (Static Methods) - Performance Optimizations and Legacy Compute.
    /// - fn getMayInplace(input_idx: *[*]i32, output_idx: *[*]i32) usize
    /// - fn releaseMayInplace(input_idx: [*]i32, output_idx: [*]i32) void
    /// - fn getAliasMap(input_idx: *[*]i32, output_idx: *[*]i32) usize
    /// - fn releaseAliasMap(input_idx: [*]i32, output_idx: [*]i32) void
    /// - fn computeV1(kernel_state: *anyopaque, context: *KernelContext) void
    pub fn init(comptime T: type) @This() {
      const VTable = struct {
        fn getSelf(ptr: [*c]const Api.c.OrtCustomOp) *const T {
          return @fieldParentPtr("ort_op", apiCastTo(ptr.?, *const Custom));
        }

        fn createKernelV1(op: [*c]const Api.c.OrtCustomOp, api: ?*const Api.c.OrtApi, info: ?*const Api.c.OrtKernelInfo) callconv(.c) ?*anyopaque {
          return getSelf(op).createKernelV1(api.?, apiCast(info.?)) catch null;
        }

        fn getName(op: [*c]const Api.c.OrtCustomOp) callconv(.c) [*c]const u8 {
          return cStr(getSelf(op).getName());
        }

        fn getEPType(op: [*c]const Api.c.OrtCustomOp) callconv(.c) [*c]const u8 {
          return cStr(getSelf(op).getExecutionProviderType());
        }

        fn getInputType(op: [*c]const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.ONNXTensorElementDataType {
          return @intFromEnum(getSelf(op).getInputType(index));
        }

        fn getInputTypeCount(op: [*c]const Api.c.OrtCustomOp) callconv(.c) usize {
          return getSelf(op).getInputTypeCount();
        }

        fn getOutputType(op: [*c]const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.ONNXTensorElementDataType {
          return @intFromEnum(getSelf(op).getOutputType(index));
        }

        fn getOutputTypeCount(op: [*c]const Api.c.OrtCustomOp) callconv(.c) usize {
          return getSelf(op).getOutputTypeCount();
        }

        fn getInputChar(op: [*c]const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.OrtCustomOpInputOutputCharacteristic {
          return @intFromEnum(getSelf(op).getInputCharacteristic(index));
        }

        fn getOutputChar(op: [*c]const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.OrtCustomOpInputOutputCharacteristic {
          return @intFromEnum(getSelf(op).getOutputCharacteristic(index));
        }

        fn getInputMemType(op: [*c]const Api.c.OrtCustomOp, index: usize) callconv(.c) Api.c.OrtMemType {
          return @intFromEnum(getSelf(op).getInputMemoryType(index));
        }

        fn getVarInMin(op: [*c]const Api.c.OrtCustomOp) callconv(.c) c_int {
          return getSelf(op).getVariadicInputMinArity();
        }

        fn getVarInHomog(op: [*c]const Api.c.OrtCustomOp) callconv(.c) c_int {
          return @intFromBool(getSelf(op).getVariadicInputHomogeneity());
        }

        fn getVarOutMin(op: [*c]const Api.c.OrtCustomOp) callconv(.c) c_int {
          return getSelf(op).getVariadicOutputMinArity();
        }

        fn getVarOutHomog(op: [*c]const Api.c.OrtCustomOp) callconv(.c) c_int {
          return @intFromBool(getSelf(op).getVariadicOutputHomogeneity());
        }

        fn createKernelV2(op: [*c]const Api.c.OrtCustomOp, api: ?*const Api.c.OrtApi, info: ?*const Api.c.OrtKernelInfo, kernel_out: ?*?*anyopaque) callconv(.c) ?*Api.c.OrtStatus {
          const res = getSelf(op).createKernelV2(api.?, apiCastTo(info.?, *const Op.KernelInfo)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          kernel_out.?.* = res;
          return null;
        }

        fn inferShape(op: [*c]const Api.c.OrtCustomOp, ctx: ?*Api.c.OrtShapeInferContext) callconv(.c) ?*Api.c.OrtStatus {
          getSelf(op).inferOutputShape(apiCastTo(ctx.?, *const ShapeInferContext)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn getStartVer(op: [*c]const Api.c.OrtCustomOp) callconv(.c) c_int {
          return getSelf(op).getStartVersion();
        }

        fn getEndVer(op: [*c]const Api.c.OrtCustomOp) callconv(.c) c_int {
          return getSelf(op).getEndVersion();
        }

        // Static Methods below this comment (C does NOT pass 'op' pointer)

        fn kernelComputeV1(kernel_state: ?*anyopaque, context: ?*Api.c.OrtKernelContext) callconv(.c) void {
          T.computeV1(kernel_state, apiCast(context.?));
        }

        fn kernelDestroy(kernel_state: ?*anyopaque) callconv(.c) void {
          T.destroyKernel(kernel_state.?);
        }

        fn kernelComputeV2(kernel_state: ?*anyopaque, context: ?*Api.c.OrtKernelContext) callconv(.c) ?*Api.c.OrtStatus {
          T.computeV2(kernel_state.?, apiCastTo(context.?, *KernelContext)) catch |err|
            return apiCast(Error.Status.initInfallible(@intFromEnum(Error.Code.Fail), @errorName(err)));
          return null;
        }

        fn getInplace(in_idx: ?*?*c_int, out_idx: ?*?*c_int) callconv(.c) usize {
          return T.getMayInplace(apiCast(in_idx.?), apiCast(out_idx.?));
        }

        fn releaseInplace(in_idx: ?*c_int, out_idx: ?*c_int) callconv(.c) void {
          T.releaseMayInplace(apiCast(in_idx), apiCast(out_idx));
        }

        fn getAlias(in_idx: ?*?*c_int, out_idx: ?*?*c_int) callconv(.c) usize {
          return T.getAliasMap(apiCast(in_idx.?), apiCast(out_idx.?));
        }

        fn releaseAlias(in_idx: ?*c_int, out_idx: ?*c_int) callconv(.c) void {
          T.releaseAliasMap(apiCast(in_idx), apiCast(out_idx));
        }
      };

      return .{
        .underlying = .{
          .version = Api.c.ORT_API_VERSION,
          .CreateKernel = if (@hasDecl(T, "createKernelV1")) VTable.createKernelV1 else null,
          .GetName = VTable.getName,
          .GetExecutionProviderType = VTable.getEPType,
          .GetInputType = VTable.getInputType,
          .GetInputTypeCount = VTable.getInputTypeCount,
          .GetOutputType = VTable.getOutputType,
          .GetOutputTypeCount = VTable.getOutputTypeCount,
          .KernelCompute = if (@hasDecl(T, "computeV1")) VTable.kernelComputeV1 else null,
          .KernelDestroy = if (@hasDecl(T, "destroyKernel")) VTable.kernelDestroy else null,
          .GetInputCharacteristic = if (@hasDecl(T, "getInputCharacteristic")) VTable.getInputChar else null,
          .GetOutputCharacteristic = if (@hasDecl(T, "getOutputCharacteristic")) VTable.getOutputChar else null,
          .GetInputMemoryType = if (@hasDecl(T, "getInputMemoryType")) VTable.getInputMemType else null,
          .GetVariadicInputMinArity = if (@hasDecl(T, "getVariadicInputMinArity")) VTable.getVarInMin else null,
          .GetVariadicInputHomogeneity = if (@hasDecl(T, "getVariadicInputHomogeneity")) VTable.getVarInHomog else null,
          .GetVariadicOutputMinArity = if (@hasDecl(T, "getVariadicOutputMinArity")) VTable.getVarOutMin else null,
          .GetVariadicOutputHomogeneity = if (@hasDecl(T, "getVariadicOutputHomogeneity")) VTable.getVarOutHomog else null,
          .CreateKernelV2 = if (@hasDecl(T, "createKernelV2")) VTable.createKernelV2 else null,
          .KernelComputeV2 = if (@hasDecl(T, "computeV2")) VTable.kernelComputeV2 else null,
          .InferOutputShapeFn = if (@hasDecl(T, "inferOutputShape")) VTable.inferShape else null,
          .GetStartVersion = if (@hasDecl(T, "getStartVersion")) VTable.getStartVer else null,
          .GetEndVersion = if (@hasDecl(T, "getEndVersion")) VTable.getEndVer else null,
          .GetMayInplace = if (@hasDecl(T, "getMayInplace")) VTable.getInplace else null,
          .ReleaseMayInplace = if (@hasDecl(T, "releaseMayInplace")) VTable.releaseInplace else null,
          .GetAliasMap = if (@hasDecl(T, "getAliasMap")) VTable.getAlias else null,
          .ReleaseAliasMap = if (@hasDecl(T, "releaseAliasMap")) VTable.releaseAlias else null,
        },
      };
    }

    pub const Domain = opaque {
      pub const Underlying = Api.c.OrtCustomOpDomain;
      /// Create a custom op domain.
      pub fn init(domain: [*:0]const u8) !*@This() {
        var out: ?*@This() = null;
        try Error.check(Api.ort.CreateCustomOpDomain.?(domain, apiCast(&out)));
        return out orelse error.OutOfMemory;
      }

      /// Add a custom op to the domain.
      /// The Op.Custom struct must remain valid until the domain is released.
      pub fn add(self: *@This(), op: *Custom) !void {
        try Error.check(Api.ort.CustomOpDomain_Add.?(
          apiCast(self),
          apiCast(op),
        ));
      }

      /// Release the domain.
      pub fn deinit(self: *@This()) void {
        Api.ort.ReleaseCustomOpDomain.?(apiCast(self));
      }
    };
  };
};

/// Information about an initializer stored in an external file (e.g., filepath, offset, size).
/// Wraps OrtExternalInitializerInfo.
pub const ExternalInitializerInfo = opaque {
  pub const Underlying = Api.c.OrtExternalInitializerInfo;
  /// Creates an OrtExternalInitializerInfo instance.
  /// Wraps OrtApi::CreateExternalInitializerInfo
  pub fn init(filepath: Utils.Path, file_offset: i64, byte_size: usize) !*@This() {
    var out: ?*@This() = null;
    try Error.check(Api.ort.CreateExternalInitializerInfo.?(
      pathCast(filepath),
      file_offset,
      byte_size,
      apiCast(&out),
    ));
    return out orelse error.OutOfMemory;
  }

  /// Get the relative path to the file that stores the initializer's data.
  /// The path is relative to the filesystem directory where the ONNX model was stored.
  /// Note: Do NOT free this pointer. It is valid for the lifetime of this object.
  /// Wraps OrtApi::ExternalInitializerInfo_GetFilePath
  pub fn getFilePath(self: *const @This()) Utils.Path {
    return pathCastTo(Api.ort.ExternalInitializerInfo_GetFilePath.?(apiCast(self)), Utils.Path);
  }

  /// Get the byte offset within the file where the initializer's data is stored.
  /// Wraps OrtApi::ExternalInitializerInfo_GetFileOffset
  pub fn getFileOffset(self: *const @This()) i64 {
    return Api.ort.ExternalInitializerInfo_GetFileOffset.?(apiCast(self));
  }

  /// Get the size in bytes of the initializer's data within the file.
  /// Wraps OrtApi::ExternalInitializerInfo_GetByteSize
  pub fn getByteSize(self: *const @This()) usize {
    return Api.ort.ExternalInitializerInfo_GetByteSize.?(apiCast(self));
  }

  /// Release an OrtExternalInitializerInfo instance.
  /// Wraps OrtApi::ReleaseExternalInitializerInfo
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleaseExternalInitializerInfo.?(apiCast(self));
  }
};

/// Wrapper around ::OrtPrepackedWeightsContainer
/// Create only and pass to Session constructor for multiple sessions to share pre-packed weights.
pub const PrepackedWeightsContainer = opaque {
  pub const Underlying = Api.c.OrtPrepackedWeightsContainer;
  /// Wraps OrtApi::CreatePrepackedWeightsContainer
  pub fn init() !*@This() {
    var self: ?*@This() = null;
    try Error.check(Api.ort.CreatePrepackedWeightsContainer.?(apiCast(&self)));
    return self orelse error.OutOfMemory;
  }

  /// Release the container.
  /// Note: Instance must not be released until the sessions using it are released.
  pub fn deinit(self: *@This()) void {
    Api.ort.ReleasePrepackedWeightsContainer.?(apiCast(self));
  }
};

/// ShapeInferContext provides access to input metadata and attributes during the shape inference phase of an ONNX Runtime Custom Operator.
///
/// This context is passed to the custom operator's shape inference function to allow it to compute and set the output shapes based on input shapes and node attributes.
pub const ShapeInferContext = opaque {
  pub const Underlying = Api.c.OrtShapeInferContext;
  /// Returns the number of inputs provided to this operator node.
  pub fn getInputCount(self: *const @This()) !usize {
    var out: usize = 0;
    try Error.check(Api.ort.ShapeInferContext_GetInputCount.?(apiCast(self), &out));
    return out;
  }

  /// Get type and shape info of an input tensor.
  /// 
  /// index: The zero-based index of the input.
  ///
  /// Returns:
  /// - A pointer to the tensor's type and shape.
  /// - An error if the ORT call fails.
  ///
  /// You must NOT free the returned pointer.
  pub fn getInputTypeShape(self: *const @This(), index: usize) !*TensorTypeAndShapeInfo.C {
    var out: ?*TensorTypeAndShapeInfo.C = null;
    try Error.check(Api.ort.ShapeInferContext_GetInputTypeShape.?(apiCast(self), index, apiCast(&out)));
    // Api returns an error "Failed to fetch type shape info for the index." if the info is null => null == oom
    //
    // https://github.com/microsoft/onnxruntime/blob/669128acd5eb804daa5d756e1dcf349abbdfcdac/onnxruntime/core/session/custom_ops.cc#L335
    // if (*info) {
    //   return nullptr;
    // } else {
    //   return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
    //                                "Failed to fetch type shape info for the index.");
    // }

    return out orelse error.OutOfMemory;
  }

  /// Get attribute from OrtShapeInferContext. Note that OrtShapeInferContext is a per-node context, one could only read attribute from current node.
  ///
  /// name: The null-terminated string name of the attribute as defined in the model.
  /// 
  /// Returns:
  /// - A pointer to the attribute if found.
  /// - `null` if the attribute does not exist on this node.
  /// - An error if the ORT call fails.
  pub fn getAttribute(self: *const @This(), name: [*:0]const u8) !?*const Op.Attr {
    var out: ?*const Op.Attr = null;
    try Error.check(Api.ort.ShapeInferContext_GetAttribute.?(apiCast(self), cStr(name), apiCast(&out)));
    return out;
  }

  /// Sets the inferred shape and type for a specific output of the operator.
  /// 
  /// This is the "final step" of shape inference where you provide the calculated output dimensions back to the onnx runtime.
  ///
  /// index: The zero-based index of the output to set.
  /// info: The computed type and shape information.
  pub fn setOutputTypeShape(self: *const @This(), index: usize, info: *const TensorTypeAndShapeInfo.C) !void {
    try Error.check(Api.ort.ShapeInferContext_SetOutputTypeShape.?(apiCast(self), index, apiCast(info)));
  }
};

/// Option structs and opaques for providers
pub const ProviderOptions = struct {
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
    pub const Underlying = Api.c.OrtTensorRTProviderOptionsV2;
    /// Wraps OrtApi::CreateTensorRTProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateTensorRTProviderOptions.?(apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateTensorRTProviderOptions
    /// options: Array of keys and values.
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateTensorRTProviderOptions.?(apiCast(self), cStr(keys.ptr), cStr(values.ptr), keys.len));
    }

    /// Wraps OrtApi::UpdateTensorRTProviderOptionsWithValue
    /// Update an option where its data type is a pointer (e.g., user_compute_stream).
    pub fn updateWithValue(self: *@This(), key: [*:0]const u8, value: *anyopaque) !void {
      try Error.check(Api.ort.UpdateTensorRTProviderOptionsWithValue.?(apiCast(self), cStr(key), value));
    }

    /// Wraps OrtApi::GetTensorRTProviderOptionsByName
    /// Get a provider option where its data type is pointer.
    pub fn getByName(self: *const @This(), key: [*:0]const u8) !?*anyopaque {
      var out: ?*anyopaque = null;
      try Error.check(Api.ort.GetTensorRTProviderOptionsByName.?(apiCast(self), cStr(key), &out));
      return out;
    }

    /// Wraps OrtApi::GetTensorRTProviderOptionsAsString
    /// The returned string must be freed using the allocator.
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetTensorRTProviderOptionsAsString.?(apiCast(self), apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseTensorRTProviderOptions.?(apiCast(self));
    }
  };

  /// ROCM Provider Options
  /// Wraps ::OrtROCMProviderOptions
  pub const ROCM = struct {
    pub const Underlying = Api.c.OrtROCMProviderOptions;
    underlying: Api.c.OrtROCMProviderOptions,

    /// Wraps OrtApi::CreateROCMProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateROCMProviderOptions.?(apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateROCMProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateROCMProviderOptions.?(apiCast(self), cStr(keys.ptr), cStr(values.ptr), keys.len));
    }

    /// Wraps OrtApi::GetROCMProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetROCMProviderOptionsAsString.?(apiCast(self), apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseROCMProviderOptions.?(apiCast(self));
    }
  };

  /// CUDA Provider Options (V2)
  pub const CUDA = opaque {
    pub const Underlying = Api.c.OrtCUDAProviderOptionsV2;
    /// Wraps OrtApi::CreateCUDAProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateCUDAProviderOptions.?(apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateCUDAProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateCUDAProviderOptions.?(apiCast(self), cStr(keys.ptr), cStr(values.ptr), keys.len));
    }

    /// Wraps OrtApi::UpdateCUDAProviderOptionsWithValue
    pub fn updateWithValue(self: *@This(), key: [*:0]const u8, value: *anyopaque) !void {
      try Error.check(Api.ort.UpdateCUDAProviderOptionsWithValue.?(apiCast(self), cStr(key), value));
    }

    /// Wraps OrtApi::GetCUDAProviderOptionsByName
    /// Get a provider option where its data type is pointer.
    pub fn getByName(self: *const @This(), key: [*:0]const u8) !?*anyopaque {
      var out: ?*anyopaque = null;
      try Error.check(Api.ort.GetCUDAProviderOptionsByName.?(apiCast(self), cStr(key), &out));
      return out;
    }

    /// Wraps OrtApi::GetCUDAProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetCUDAProviderOptionsAsString.?(apiCast(self), apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseCUDAProviderOptions.?(apiCast(self));
    }
  };

  /// CANN Provider Options
  pub const CANN = opaque {
    pub const Underlying = Api.c.OrtCANNProviderOptions;
    /// Wraps OrtApi::CreateCANNProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateCANNProviderOptions.?(apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateCANNProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateCANNProviderOptions.?(apiCast(self), cStr(keys.ptr), cStr(values.ptr), keys.len));
    }

    /// Wraps OrtApi::GetCANNProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetCANNProviderOptionsAsString.?(apiCast(self), apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseCANNProviderOptions.?(apiCast(self));
    }
  };

  /// DNNL Provider Options
  pub const Dnnl = opaque {
    pub const Underlying = Api.c.OrtDnnlProviderOptions;
    /// Wraps OrtApi::CreateDnnlProviderOptions
    pub fn init() !*@This() {
      var out: ?*@This() = null;
      try Error.check(Api.ort.CreateDnnlProviderOptions.?(apiCast(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Wraps OrtApi::UpdateDnnlProviderOptions
    pub fn update(self: *@This(), keys: []const [*:0]const u8, values: []const [*:0]const u8) !void {
      if (keys.len != values.len) return error.InvalidArgs;
      try Error.check(Api.ort.UpdateDnnlProviderOptions.?(apiCast(self), cStr(keys.ptr), cStr(values.ptr), keys.len));
    }

    /// Wraps OrtApi::GetDnnlProviderOptionsAsString
    pub fn getOptionsAsString(self: *const @This(), allocator: *Allocator) ![*:0]u8 {
      var out: ?[*:0]u8 = null;
      try Error.check(Api.ort.GetDnnlProviderOptionsAsString.?(apiCast(self), apiCast(allocator), cStr(&out)));
      return out orelse error.OutOfMemory;
    }

    /// Release the options.
    pub fn deinit(self: *@This()) void {
      Api.ort.ReleaseDnnlProviderOptions.?(apiCast(self));
    }
  };
};

