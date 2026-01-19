const std = @import("std");
const testing = std.testing;
const onnx = @import("onnxruntime");
const Api = onnx.Api;

const Utils = onnx.Utils;
pub const Utils_tests = struct {
  // Test createOptionsKVL with .usize variant (Session Options logic)
  test "Utils.createOptionsKVL - usize" {
    const MockOptions = struct {
      intra_threads: usize = 0,
      inter_threads: usize = 0,
      enable_mem_pattern: bool = false,
    };

    const kvl = Utils.createOptionsKVL(MockOptions{
      .intra_threads = 5,
      // inter_threads left at default 0
      .enable_mem_pattern = true,
    }, .usize);

    // Should only have 2 entries because inter_threads is at default
    try testing.expectEqual(@as(usize, 2), kvl.len);

    // Verify Keys
    try testing.expectEqualStrings("intra_threads", std.mem.sliceTo(kvl.keys()[0], 0));
    try testing.expectEqualStrings("enable_mem_pattern", std.mem.sliceTo(kvl.keys()[1], 0));

    // Verify Values (bool true becomes 1, int stays 5)
    try testing.expectEqual(@as(usize, 5), kvl.vals()[0]);
    try testing.expectEqual(@as(usize, 1), kvl.vals()[1]);
  }

  // Test createOptionsKVL with .cstr variant (Execution Provider logic)
  test "Utils.createOptionsKVL - cstr" {
    const MockEpOptions = struct {
      device_id: [*:0]const u8 = "0",
      cache_path: ?[*:0]const u8 = null,
    };

    const inst: MockEpOptions = .{
      .device_id = "1",
      .cache_path = "/tmp/cache",
    };

    const kvl = Utils.createOptionsKVL(inst, .cstr);

    try testing.expectEqual(@as(usize, 2), kvl.len);
    try testing.expectEqualStrings("1", std.mem.sliceTo(kvl.vals()[0], 0));
    try testing.expectEqualStrings("/tmp/cache", std.mem.sliceTo(kvl.vals()[1], 0));
  }
};

const Ep = onnx.Ep;
pub const Ep_tests = struct {
  // Mock for Interface and Factory testing
  const MockProvider = struct {
    ep: Ep.Interface = undefined,
    factory: Ep.Factory = undefined,
    name: [*:0]const u8 = "MockEP",

    pub fn getName(self: *const @This()) [*:0]const u8 { return self.name; }
    pub fn getVendor(_: *const @This()) [*:0]const u8 { return "ZigVendor"; }
    pub fn getVersion(_: *const @This()) [*:0]const u8 { return "1.0.0"; }
    pub fn getCapability(_: *@This(), _: *const Graph, _: *Ep.GraphSupportInfo) !void {}
    pub fn compile(_: *@This(), _: []const *const Graph, _: []const *const Node, _: []*Ep.NodeCompute.Info, _: []*Node) !void {}
    pub fn releaseNodeComputeInfos(_: *@This(), _: []*Ep.NodeCompute.Info) void {}
    pub fn getCompiledModelCompatibilityInfo(_: *@This(), _: *const Graph) [*:0]const u8 { return ""; }
    pub fn getSupportedDevices(_: *@This(), _: []const *const HardwareDevice, ep_out: []*Ep.Device) !usize {
      _ = ep_out;
      return 0;
    }
    pub fn createEp(self: *@This(), _: []const *const HardwareDevice, _: []const *const KeyValuePairs, _: *const Session.Options.C, _: *const Op.KernelInfo.Logger) !*Ep.Interface {
      return &self.ep;
    }
    pub fn releaseEp(_: *@This(), _: *Ep.Interface) void {}
    pub fn getVendorId(_: *const @This()) u32 { return 0; }
    pub fn validateCompiledModelCompatibilityInfo(_: *@This(), _: []const *const HardwareDevice, _: [*:0]const u8) Api.compiler.CompiledModelCompatibility {
      return .NOT_APPLICABLE;
    }
  };

  test "Ep.Interface - VTable Routing" {
    var mock = MockProvider{};
    mock.ep = Ep.Interface.init(MockProvider);

    const c_ep = Utils.apiCast(&mock.ep);
    const name = mock.ep.underlying.GetName.?(c_ep);
    try testing.expectEqualStrings("MockEP", std.mem.sliceTo(name, 0));
  }

  test "Ep.DataTransfer - Mock Implementation" {
    const MockDT = struct {
      data_transfer: Ep.DataTransfer = undefined,
      pub fn canCopy(_: *const @This(), _: *const Allocator.MemoryDevice, _: *const Allocator.MemoryDevice) bool { return true; }
      pub fn copyTensors(_: *@This(), _: []const *const Value, _: []*Value, _: ?[]?*Ep.SyncStream) !void {}
    };

    var mock = MockDT{};
    mock.data_transfer = Ep.DataTransfer.init(MockDT);

    const dev: *Allocator.MemoryDevice = @ptrCast(@constCast(&.{}));
    try testing.expect(mock.data_transfer.canCopy(dev, dev));

    const c_dt = Utils.apiCast(&mock.data_transfer);
    try testing.expect(mock.data_transfer.underlying.CanCopy.?(c_dt, Utils.apiCast(dev), Utils.apiCast(dev)));
  }

  test "Ep.SyncNotification - Implementation Routing" {
    const MockNotif = struct {
      sync_notification: onnx.Ep.SyncNotificationImpl = undefined,
      pub fn activate(_: *@This()) !void {}
      pub fn waitOnDevice(_: *@This(), _: *Ep.SyncStream) !void {}
      pub fn waitOnHost(_: *@This()) !void {}
      pub fn deinit(_: *@This()) void {}
    };

    var mock = MockNotif{};
    mock.sync_notification = onnx.Ep.SyncNotificationImpl.init(MockNotif);
    const c_notif = Utils.apiCast(&mock.sync_notification);

    try Error.check(mock.sync_notification.underlying.Activate.?(c_notif));
    try Error.check(mock.sync_notification.underlying.WaitOnHost.?(c_notif));
    mock.sync_notification.underlying.Release.?(c_notif);
  }

  test "Ep.SyncStreamImpl - Fully Routed" {
    const MockStream = struct {
      sync_stream: onnx.Ep.SyncStream.Impl = undefined,
      pub fn getHandle(_: *@This()) ?*anyopaque { return null; }
      pub fn createNotification(_: *@This(), _: *?*onnx.Ep.SyncNotificationImpl) !void {}
      pub fn flush(_: *@This()) !void {}
      pub fn onSessionRunEnd(_: *@This()) !void {}
    };

    var mock = MockStream{};
    mock.sync_stream = onnx.Ep.SyncStream.Impl.init(MockStream);
    const c_stream = Utils.apiCast(&mock.sync_stream);

    try Error.check(mock.sync_stream.underlying.Flush.?(c_stream));
    try Error.check(mock.sync_stream.underlying.OnSessionRunEnd.?(c_stream));
  }

  test "Ep.SyncStream - Logic and IDs" {
    const devices = try onnx.Api.env.getEpDevices();
    var accel_dev: ?*const onnx.Ep.Device = null;
    for (devices) |d| {
      if (d.getHardwareDevice().getType() != .CPU) {
        accel_dev = d;
        break;
      }
    }

    // Skip stream init if no GPU/NPU is available
    if (accel_dev) |dev| {
      const stream = try Ep.SyncStream.init(dev, null);
      defer stream.deinit();
      _ = stream.getHandle();
      _ = stream.getSyncId();
      _ = stream.getImpl();
      try testing.expectEqual(@as(u64, 0), Ep.SyncStream.getSyncIdForLastWaitOnSyncStream(stream, stream));
    } else return error.SkipZigTest;
  }

  test "Ep.SyncStreamImpl & SyncNotificationImpl - GetHandle" {
    const MockStream = struct {
      sync_stream: Ep.SyncStream.Impl = undefined,
      pub fn getHandle(_: *@This()) ?*anyopaque { return @ptrFromInt(0x456); }
      pub fn createNotification(_: *@This(), out: *?*Ep.SyncNotificationImpl) !void {
        _ = out;
      }
      pub fn flush(_: *@This()) !void {}
      pub fn onSessionRunEnd(_: *@This()) !void {}
    };

    var mock = MockStream{};
    mock.sync_stream = Ep.SyncStream.Impl.init(MockStream);
    try testing.expectEqual(@as(usize, 0x456), @intFromPtr(mock.sync_stream.handle()));
  }

  test "Ep.GraphSupportInfo - Node Aggregation" {
    @setRuntimeSafety(false);
    const info: ?*Ep.GraphSupportInfo = null;
    const node: ?*const Node = null;
    const nodes = [_]?*const onnx.Node{node};

    const addSingleNode: *const fn (self: ?*Ep.GraphSupportInfo, node: ?*const Node) anyerror!void = @ptrCast(&Ep.GraphSupportInfo.addSingleNode);
    try std.testing.expectError(Error.Set.OrtErrorInvalidArgument, addSingleNode(info, node));

    const addNodesToFuse: *const fn (self: ?*Ep.GraphSupportInfo, nodes: []const ?*const Node, options: ?*const Ep.NodeFusionOptions) anyerror!void = @ptrCast(&Ep.GraphSupportInfo.addNodesToFuse);
    try std.testing.expectError(Error.Set.OrtErrorInvalidArgument, addNodesToFuse(info, &nodes, null));
  }

  test "Ep.NodeCompute - VTable Routing" {
    const MockComp = struct {
      compute_info: Ep.NodeCompute.Info = undefined,
      pub fn createState(_: *@This(), _: *Ep.NodeCompute.Context) !*Ep.NodeCompute.State { return @ptrFromInt(0x123); }
      pub fn compute(_: *@This(), _: *Ep.NodeCompute.State, _: *Op.KernelContext) !void {}
      pub fn releaseState(_: *@This(), _: *Ep.NodeCompute.State) void {}
    };

    var mock = MockComp{};
    mock.compute_info = Ep.NodeCompute.Info.init(MockComp);

    const c_info = Utils.apiCast(&mock.compute_info);
    var state_ptr: ?*anyopaque = null;
    try Error.check(mock.compute_info.underlying.CreateState.?(c_info, @ptrCast(@constCast(&.{})), &state_ptr));
    try testing.expectEqual(@as(usize, 0x123), @intFromPtr(state_ptr));
  }

  test "Ep.NodeCompute.Context - Metadata" {
    // We cannot call name() on a fake pointer because the C-API dereferences it 
    // immediately to return a char* (it does not return a Status).
    // We verify the VTable entry is correctly linked to satisfy coverage of the API access.
    try testing.expect(onnx.Ep.api.underlying.NodeComputeContext_NodeName != null);
    
    var ctx: ?*onnx.Ep.NodeCompute.Context = null;
    std.mem.doNotOptimizeAway(&ctx);
    if (ctx) |c| _ = c.name();
  }

  test "Ep.Device - Environment Devices" {
    const devices = try Api.env.getEpDevices();
    for (devices) |dev| {
      _ = dev.getEpName();
      _ = dev.getEpVendor();
      _ = dev.getEpMetadata();
      _ = dev.getEpOptions();
      const hw = dev.getHardwareDevice();
      try testing.expect(hw.getVendorId() >= 0);
    }
  }

  test "Ep.Factory - VTable Routing" {
    var mock = MockProvider{};
    mock.factory = Ep.Factory.init(MockProvider);

    const c_factory = Utils.apiCast(&mock.factory);
    try testing.expectEqualStrings("MockEP", std.mem.sliceTo(mock.factory.underlying.GetName.?(c_factory), 0));
    try testing.expectEqualStrings("ZigVendor", std.mem.sliceTo(mock.factory.underlying.GetVendor.?(c_factory), 0));
  }

  test "Ep.Factory - Compatibility and Metadata" {
    const MockFact = struct {
      factory: onnx.Ep.Factory = undefined,
      pub fn getName(_: *const @This()) [*:0]const u8 { return "F"; }
      pub fn getVendor(_: *const @This()) [*:0]const u8 { return "V"; }
      pub fn getSupportedDevices(_: *@This(), _: []const *const onnx.HardwareDevice, _: []*onnx.Ep.Device) !usize { return 0; }
      pub fn createEp(_: *@This(), _: []const *const onnx.HardwareDevice, _: []const *const onnx.KeyValuePairs, _: *const onnx.Session.Options.C, _: *const onnx.Op.KernelInfo.Logger) !*onnx.Ep.Interface { return undefined; }
      pub fn releaseEp(_: *@This(), _: *onnx.Ep.Interface) void {}
      pub fn getVendorId(_: *const @This()) u32 { return 1; }
      pub fn getVersion(_: *const @This()) [*:0]const u8 { return "1"; }
      pub fn validateCompiledModelCompatibilityInfo(_: *@This(), _: []const *const onnx.HardwareDevice, _: [*:0]const u8) onnx.Api.compiler.CompiledModelCompatibility { return .NOT_APPLICABLE; }
      pub fn isStreamAware(_: *const @This()) bool { return true; }
    };

    var mock = MockFact{};
    mock.factory = onnx.Ep.Factory.init(MockFact);
    const c_fact = Utils.apiCast(&mock.factory);

    try testing.expectEqual(@as(u32, 1), mock.factory.underlying.GetVendorId.?(c_fact));
    try testing.expect(mock.factory.underlying.IsStreamAware.?(c_fact));
    try testing.expectEqualStrings("1", std.mem.sliceTo(mock.factory.underlying.GetVersion.?(c_fact), 0));
  }

  test "Ep.HardwareDevice - Full Getters" {
    const devices = try Api.env.getEpDevices();
    if (devices.len == 0) return;
    
    // Get the hardware device from an existing EP device
    const hw = devices[0].getHardwareDevice();
    
    // Test all getters
    _ = hw.getType();
    _ = hw.getVendorId();
    _ = hw.getDeviceId();
    _ = hw.getVendor();
    
    // Metadata is a KeyValuePairs object owned by ORT
    const meta = hw.getMetadata();
    _ = meta.getKeyValues();
  }

  test "Ep.Device - Manual Init and Specific Getters" {
    const devices = try Api.env.getEpDevices();
    if (devices.len == 0) return;

    const hw = devices[0].getHardwareDevice();

    var run: bool = false;
    _ = &run;
    if (run) { // we only test that compilation succeeds
      // Device.init requires a Factory. 
      // We use a dummy pointer to test the routing to CreateEpDevice
      const dummy_factory: ?*Ep.Factory = null;
      _ = Ep.Device.init(dummy_factory.?, hw, null, null) catch {};

      // Test specific info getters
      const dev = devices[0];
      _ = dev.getEpName();
      _ = dev.getEpVendor();
      _ = dev.getEpMetadata();
      _ = dev.getEpOptions();

      // Test MemoryInfo retrieval
      _ = dev.getMemoryInfo(.DEFAULT);
      _ = dev.getMemoryInfo(.HOST_ACCESSIBLE);
    }
  }

  test "Ep.SyncStream - Detailed IDs" {
    const devices = try Api.env.getEpDevices();
    var accel_dev: ?*const Ep.Device = null;
    for (devices) |d| {
      if (d.getHardwareDevice().getType() != .CPU) {
        accel_dev = d;
        break;
      }
    }

    if (accel_dev) |dev| {
      const stream = try Ep.SyncStream.init(dev, null);
      defer stream.deinit();

      // Ensure ID functions are reached
      _ = stream.getSyncId();
      _ = Ep.SyncStream.getSyncIdForLastWaitOnSyncStream(stream, stream);
    }
  }

  test "Ep.DataTransfer - Release Routing" {
    const MockDT = struct {
      data_transfer: Ep.DataTransfer = undefined,
      pub fn canCopy(_: *const @This(), _: *const Allocator.MemoryDevice, _: *const Allocator.MemoryDevice) bool { return true; }
      pub fn copyTensors(_: *@This(), _: []const *const Value, _: []*Value, _: ?[]?*Ep.SyncStream) !void {}
      pub fn deinit(_: *@This()) void {}
    };

    var mock = MockDT{};
    mock.data_transfer = Ep.DataTransfer.init(MockDT);

    // Tests the Release vtable routing
    mock.data_transfer.deinit();
  }

  test "Ep.NodeFusionOptions - Layout" {
    // NodeFusionOptions is a simple wrapper struct
    const options: Ep.NodeFusionOptions = undefined;
    try testing.expect(@sizeOf(@TypeOf(options)) == @sizeOf(*anyopaque));
  }
};

fn ArgsTuple(comptime Function: type) ?type {
  @setEvalBranchQuota(1000_000);
  const info = @typeInfo(Function);
  if (info != .@"fn") @compileError("ArgsTuple expects a function type");

  const function_info = info.@"fn";
  if (function_info.is_var_args) return null;

  var argument_field_list: [function_info.params.len]type = undefined;
  inline for (function_info.params, 0..) |arg, i| {
    const T = arg.type orelse return null;
    if (T == type or @typeInfo(T) == .@"fn") return null;
    argument_field_list[i] = T;
  }

  var tuple_fields: [argument_field_list.len]std.builtin.Type.StructField = undefined;
  inline for (argument_field_list, 0..) |T, i| {
    @setEvalBranchQuota(10_000);
    var num_buf: [128]u8 = undefined;
    tuple_fields[i] = .{
      .name = std.fmt.bufPrintZ(&num_buf, "{d}", .{i}) catch unreachable,
      .type = T,
      .default_value_ptr = null,
      .is_comptime = false,
      .alignment = @alignOf(T),
    };
  }

  return @Type(.{
    .@"struct" = .{
      .is_tuple = true,
      .layout = .auto,
      .decls = &.{},
      .fields = &tuple_fields,
    },
  });
}

fn initType(comptime T: type) T {
  @setEvalBranchQuota(1000_000);
  comptime var retval: T = undefined;
  switch (@typeInfo(T)) {
    .type => return void,
    .void => return undefined,
    .bool => return false,
    .noreturn => unreachable,
    .int => return 0,
    .float => return 0.0,
    .pointer => return @alignCast(@ptrCast(@constCast(&.{}))),
    .array => |ai| inline for (ai.len) |i| {@field(retval, i) = initType(ai.child);},
    .@"struct" => |si| inline for (si.fields) |field| {@field(retval, field.name) = comptime initType(@FieldType(T, field.name));},
    .comptime_float => return 0.0,
    .comptime_int => return 0,
    .undefined => unreachable,
    .null, .optional => return null,
    .error_union => |eu| return initType(eu.payload),
    .error_set => |es_| if (es_) |es| {
      if (es.len == 0) return undefined;
      return @field(T, es[0].name);
    } else error.AnyError,
    .@"enum" => |ei| if (ei.fields.len != 0) {retval = @field(T, ei.fields[0].name);} else return undefined,
    .@"union" => |ui| if (ui.fields.len != 0) {retval = @unionInit(T, ui.fields[0].name, initType(ui.fields[0].type));},
    .@"fn" => return undefined,
    .@"opaque", .frame, .@"anyframe" => unreachable,
    .vector => |vi| inline for (vi.len) |i| {@field(retval, i) = initType(vi.child);},
    .enum_literal => return undefined,
  }
  return retval;
}

/// If we use std.testing.refAllDeclsRecursive, we get a compile error because c has untranslatable code, hence we use this
/// Even this touches the translated parts of the c code that we touch, but atleast not it doesn't crash
fn refAllDeclsRecursiveExcerptC(comptime T: type) void {
  if (!@import("builtin").is_test) return;

  inline for (comptime std.meta.declarations(T)) |decl| {
    const field = @field(T, decl.name); 
    _ = &field;

    if (@TypeOf(field) == type) {
      if (decl.name.len == 1 and decl.name[0] == 'c') continue;
      switch (@typeInfo(@field(T, decl.name))) {
        .@"struct", .@"enum", .@"union", .@"opaque" => refAllDeclsRecursiveExcerptC(@field(T, decl.name)),
        else => {},
      }
    } else if (@typeInfo(@TypeOf(field)) == .@"fn") { // TODO: this currently causes compiler crash
      var should_run: bool = false;
      _ = &should_run;
      if (should_run) {
        if (ArgsTuple(@TypeOf(field))) |Args| _ = &@call(.auto, field, comptime initType(Args));
      }
    }
  }
}

const Error = onnx.Error;
const Op = onnx.Op;
const Session = onnx.Session;
const HardwareDevice = onnx.HardwareDevice;
const Allocator = onnx.Allocator;
const Value = onnx.Value;
const Node = onnx.Node;
const Graph = onnx.Graph;
const KeyValuePairs = onnx.KeyValuePairs;

test {
  refAllDeclsRecursiveExcerptC(onnx);
  testing.refAllDeclsRecursive(@This());
}

