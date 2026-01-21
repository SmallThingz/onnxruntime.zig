const std = @import("std");
const builtin = @import("builtin");
const onnx = @import("onnxruntime");
const testing = std.testing;
const Api = onnx.Api;

const HardwareDevice = onnx.HardwareDevice;
const Node = onnx.Node;
const Graph = onnx.Graph;

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

const apiCast = Utils.apiCast;
const apiCastTo = Utils.apiCastTo;
const cStr = Utils.cStr;
const cStrTo = Utils.cStrTo;
const pathCast = Utils.pathCast;
const pathCastTo = Utils.pathCastTo;
const cCast = Utils.cCast;

const empty_string = Utils.empty_string;
const empty_path = Utils.empty_path;

const Training = onnx.Training;
pub const Training_tests = struct {
  const CheckpointState = Training.CheckpointState;
  const TrainingSession = Training.TrainingSession;

  test "Training.api - Set Seed" {
    Training.api.setSeed(42) catch |err| {
      if (err == error.TrainingApiNotAvailable) return error.SkipZigTest;
      return err;
    };
  }

  test "CheckpointState - PropertyUnion Deinit Logic" {
    const allocator = try Allocator.getDefault();

    // Verify i64 property deallocation
    const i_ptr = try allocator.alloc(i64, 1);
    const p_i64 = CheckpointState.PropertyUnion{ .i64 = @ptrCast(i_ptr.ptr) };
    p_i64.deinit(allocator);

    // Verify f32 property deallocation
    const f_ptr = try allocator.alloc(f32, 1);
    const p_f32 = CheckpointState.PropertyUnion{ .f32 = @ptrCast(f_ptr.ptr) };
    p_f32.deinit(allocator);

    // Verify string property deallocation
    const s_ptr = try allocator.alloc(u8, 8);
    @memcpy(s_ptr[0..7], "zig_ort");
    s_ptr[7] = 0;
    const p_str = CheckpointState.PropertyUnion{ .str = @ptrCast(s_ptr.ptr) };
    p_str.deinit(allocator);
  }

  test "CheckpointState - API Routing" {
    const path = Utils.asPath("non_existent_checkpoint");

    // Test Load Failure (Validates routing to LoadCheckpoint)
    // This is weird because we get InvalidArgument even on invalid fd
    try testing.expectError(error.OrtErrorInvalidArgument, CheckpointState.load(path));

    // Test LoadFromBuffer Failure
    const dummy_buf = "invalid_data";
    _ = CheckpointState.loadFromBuffer(dummy_buf) catch {};
    
    // Verify VTable availability
    try testing.expect(Training.api.underlying.AddProperty != null);
    try testing.expect(Training.api.underlying.GetParameter != null);
  }

  test "TrainingSession - Initialization Routing" {
    const options = try Session.Options.C.init();
    defer options.deinit();

    const state: *CheckpointState = @ptrFromInt(0x1);
    const path = Utils.asPath("x");

    // Test Create Failure (Validates routing to CreateTrainingSession)
    const res = TrainingSession.init(options, state, path, path, path);
    try testing.expectError(error.OrtErrorNoSuchfile, res);

    // Test CreateFromBuffer routing
    const dummy = "data";
    const res_buf = TrainingSession.initBuffer(options, state, dummy, "", dummy);
    _ = res_buf catch {};
  }

  test "TrainingSession - Step and Update Routing" {
    var session: ?*TrainingSession = null;
    std.mem.doNotOptimizeAway(&session);

    if (session) |s| {
      _ = s.getTrainingModelOutputCount() catch {};
      _ = s.lazyResetGrad() catch {};
      _ = s.getLearningRate() catch {};
      _ = s.setLearningRate(0.01) catch {};
      _ = s.optimizerStep(null) catch {};
      _ = s.schedulerStep() catch {};
      _ = s.getParametersSize(true) catch {};
    }

    // Verify presence of core training functions in the API table
    try testing.expect(Training.api.underlying.TrainStep != null);
    try testing.expect(Training.api.underlying.EvalStep != null);
    try testing.expect(Training.api.underlying.OptimizerStep != null);
  }

  test "TrainingSession - Buffer and Export Routing" {
    var session: ?*TrainingSession = null;
    std.mem.doNotOptimizeAway(&session);

    if (session) |s| {
      const val: *Value = @ptrFromInt(0x1);
      _ = s.copyParametersToBuffer(val, true) catch {};
      _ = s.copyBufferToParameters(val, true) catch {};
      
      const out_names = [_][*:0]const u8{"out"};
      _ = s.exportModelForInferencing(Utils.asPath("out.onnx"), &out_names) catch {};
    }
    
    try testing.expect(Training.api.underlying.CopyParametersToBuffer != null);
    try testing.expect(Training.api.underlying.ExportModelForInferencing != null);
  }

  test "TrainingSession - Input/Output Name Routing" {
    var session: ?*TrainingSession = null;
    std.mem.doNotOptimizeAway(&session);

    if (session) |s| {
      const allocator = try Allocator.getDefault();
      _ = s.getTrainingModelInputName(0, allocator) catch {};
      _ = s.getEvalModelOutputName(0, allocator) catch {};
    }

    try testing.expect(Training.api.underlying.TrainingSessionGetTrainingModelInputName != null);
    try testing.expect(Training.api.underlying.TrainingSessionGetEvalModelOutputName != null);
  }
};

const Op = onnx.Op;
pub const Op_tests = struct {
  const MockOp = struct {
    ort_op: onnx.Op.Custom = undefined,
    called_compute: bool = false,
    
    pub fn getName(_: *const @This()) [*:0]const u8 { return "Mock"; }
    pub fn getExecutionProviderType(_: *const @This()) ?[*:0]const u8 { return null; }
    pub fn getInputType(_: *const @This(), _: usize) onnx.Value.Sub.Tensor.ElementDataType { return .f32; }
    pub fn getInputTypeCount(_: *const @This()) usize { return 1; }
    pub fn getOutputType(_: *const @This(), _: usize) onnx.Value.Sub.Tensor.ElementDataType { return .f32; }
    pub fn getOutputTypeCount(_: *const @This()) usize { return 1; }
    pub fn createKernelV2(_: *const @This(), _: *const Api.c.OrtApi, _: *const onnx.Op.KernelInfo) !*anyopaque { return @ptrFromInt(0x1); }
    
    pub fn computeV2(kernel: *anyopaque, _: *onnx.Op.KernelContext) !void {
      try testing.expectEqual(@as(usize, 0x1), @intFromPtr(kernel));
      // We use a global or pointer to verify the call since compute is static
      routing_check = true;
    }
    pub fn destroyKernel(_: *anyopaque) void {}
  };

  var routing_check: bool = false;

  test "Op.Custom - VTable Routing" {
    var mock = MockOp{};
    mock.ort_op = onnx.Op.Custom.init(MockOp);
    const c_op = Utils.apiCast(&mock.ort_op);
    const vt = mock.ort_op.underlying;

    try testing.expectEqualStrings("Mock", std.mem.sliceTo(vt.GetName.?(c_op), 0));
    try testing.expectEqual(@as(usize, 1), vt.GetInputTypeCount.?(c_op));
    
    var kernel: ?*anyopaque = null;
    try onnx.Error.check(vt.CreateKernelV2.?(c_op, @ptrFromInt(0xdeadbeef0), @ptrFromInt(0xdeadbeef0), &kernel));
    
    routing_check = false;
    try onnx.Error.check(vt.KernelComputeV2.?(kernel, @ptrFromInt(0xdeadbeef0)));
    try testing.expect(routing_check);
  }

  test "Op and Attr - Logic" {
    const attr = try onnx.Op.Attr.initInt("axis", 1);
    defer attr.deinit();
    
    try testing.expectEqualStrings("axis", std.mem.sliceTo(try attr.getName(), 0));
    try testing.expectEqual(onnx.Op.Attr.Type.INT, try attr.getType());

    const tensor_attr = try onnx.Op.Attr.initString("label", "test");
    defer tensor_attr.deinit();

    // Verify Read logic
    var buf: [10]u8 = undefined;
    _ = try tensor_attr.read(.STRING, &buf);
  }
};

const Logging = onnx.Logging;
pub const Logging_tests = struct {
  test "Interface - Routing" {
    const MockLogger = struct {
      called: bool = false,
      pub fn log(self: *@This(), severity: onnx.Logging.Level, category: ?[*:0]const u8, id: ?[*:0]const u8, loc: ?[*:0]const u8, msg: ?[*:0]const u8) void {
        _ = .{ severity, category, id, loc, msg };
        self.called = true;
      }
    };
    var mock = MockLogger{};
    const interface = onnx.Logging.Interface.fromContext(&mock);
    interface.log_fn(interface.ptr, @intFromEnum(onnx.Logging.Level.info), "cat", "id", "loc", "msg");
    try testing.expect(mock.called);
  }
};

const Error = onnx.Error;
pub const Error_tests = struct {
  test "Error - Status and Logic" {
    const status = try Error.Status.init(1, "test error");
    {
      errdefer status.deinit();
      try testing.expectEqual(@as(c_uint, 1), status.getErrorCode());
      try testing.expectEqualStrings("test error", std.mem.sliceTo(status.getErrorMessage(), 0));
    }

    const err_res = Error.check(Utils.apiCast(status));
    try testing.expectError(error.OrtErrorFail, err_res);
  }
};

const KeyValuePairs = onnx.KeyValuePairs;
pub const KeyValuePairs_tests = struct {
  test "KeyValuePairs - CRUD" {
    const kv = try onnx.KeyValuePairs.init();
    defer kv.deinit();
    kv.add("key", "val");
    try testing.expectEqualStrings("val", std.mem.sliceTo(kv.get("key").?, 0));
    const keys, const vals = kv.getKeyValues();
    try testing.expectEqualStrings("key", std.mem.sliceTo(keys[0], 0));
    try testing.expectEqualStrings("val", std.mem.sliceTo(vals[0], 0));
    kv.remove("key");
    try testing.expect(kv.get("key") == null);
  }
};

const Allocator = onnx.Allocator;
pub const Allocator_tests = struct {
  test "Allocator - Config and Stats" {
    const cfg = try onnx.Allocator.ArenaCfg.init(.{});
    defer cfg.deinit();
    const cfg2 = try onnx.Allocator.ArenaCfg.init_DeprecatedV1(.{});
    defer cfg2.deinit();

    const allocator = try onnx.Allocator.getDefault();
    const mem = allocator._alloc(10).?;
    @memset(mem, 0xAA);
    const reserved = try allocator.reserve(10);
    allocator.free(mem.ptr);
    allocator.free(reserved.ptr);

    if (try allocator.stats()) |stats| stats.deinit();
    _ = allocator.info().name() catch "Cpu";
  }
};

const TensorTypeAndShapeInfo = onnx.TensorTypeAndShapeInfo;
pub const TensorTypeAndShapeInfo_tests = struct {
  test "TensorTypeAndShapeInfo - Properties" {
    const info = try onnx.TensorTypeAndShapeInfo.C.init();
    defer info.deinit();
    try info.setElementType(.f32);
    const dims = [_]i64{ 1, 3, 224, 224 };
    try info.setDimensions(&dims);

    try testing.expectEqual(onnx.Value.Sub.Tensor.ElementDataType.f32, try info.elementType());
    try testing.expectEqual(@as(usize, 4), try info.dimensionsCount());
    try testing.expectEqual(@as(usize, 1 * 3 * 224 * 224), try info.shapeElementCount());
  }
};

pub const Value = onnx.Value;
pub const Value_tests = struct {
  test "Value - Tensor Operations" {
    const allocator = try onnx.Allocator.getDefault();
    const dims = [_]i64{ 2, 2 };
    const tensor = try onnx.Value.Sub.Tensor.init(allocator, &dims, .f32);
    defer tensor.deinit();

    try testing.expect(try tensor.toValue().isTensor());
    const data = try tensor.getData(f32);
    data[0] = 42.0;
    try testing.expectEqual(@as(f32, 42.0), (try tensor.at(&[_]i64{ 0, 0 }, f32)).*);
  }

  test "Value - Type Casting" {
    const allocator = try onnx.Allocator.getDefault();
    const tensor = try onnx.Value.Sub.Tensor.init(allocator, &[_]i64{1}, .i64);
    defer tensor.deinit();

    const val = tensor.toValue();
    const u = try val.asUnion();
    try testing.expect(u == .TENSOR);
    _ = val.asType(.TENSOR);
  }

  test "Value - String Tensor" {
    const allocator = try onnx.Allocator.getDefault();
    const tensor = try onnx.Value.Sub.Tensor.init(allocator, &[_]i64{2}, .string);
    defer tensor.deinit();

    const st = @as(*onnx.Value.Sub.Tensor.String, @ptrCast(tensor));
    const strs = [_][*:0]const u8{ "a", "bc" };
    try st.fillString(&strs);
    try testing.expectEqual(@as(usize, 1), try st.getStringElementLength(0));
    try testing.expectEqual(@as(usize, 2), try st.getStringElementLength(1));
  }

  test "Value.Sub - Sparse and Opaque" {
    const allocator = try onnx.Allocator.getDefault();

    const sparse = try onnx.Value.Sub.SparseTensor.init(allocator, &[_]i64{10}, .f32);
    defer sparse.deinit();
    _ = try sparse.getSparseFormat();
    _ = sparse.getValues() catch {};
    _ = sparse.useCooIndices(null, 0) catch {};
    // Test the comptime-heavy getIndices branch
    _ = sparse.getIndices(.COO_INDICES) catch {};

    // Opaque requires a registered type, but calling it exercises the Zig logic
    _ = onnx.Value.Sub.Opaque.init("domain", "type", "data") catch |err| {
      try testing.expect(err == error.OrtErrorFail);
    };
  }

  test "Tensor - at and getDataConst" {
    const allocator = try onnx.Allocator.getDefault();
    const dims = [_]i64{ 1, 1 };
    const tensor = try onnx.Value.Sub.Tensor.init(allocator, &dims, .f32);
    defer tensor.deinit();

    _ = try tensor.at(&[_]i64{ 0, 0 }, f32);
    _ = try tensor.getDataConst(f32);
  }

  test "Opaque - getData" {
    const opq: *onnx.Value.Sub.Opaque = @ptrFromInt(0x1);
    var out: [1]u8 = undefined;
    // Hits the function entry logic in root.zig
    _ = opq.getData(u8, &out, "d", "t") catch {};
  }
};

pub const ModelEditor_tests = struct {
  test "Graph Surgery - Identity Model" {
    const model = try onnx.Model.init(&[_][*:0]const u8{""}, &[_]c_int{21});
    defer model.deinit();
    const graph = try onnx.Graph.init();

    const info = try onnx.TensorTypeAndShapeInfo.C.init();
    defer info.deinit();
    try info.setElementType(.f32);
    try info.setDimensions(&[_]i64{1});

    const v_info = try Api.editor.createValueInfo("X", try onnx.TypeInfo.forTensor(info));
    const v_out = try Api.editor.createValueInfo("Y", try onnx.TypeInfo.forTensor(info));

    const node = try onnx.Node.init("Identity", "", "id1", &[_][*:0]const u8{"X"}, &[_][*:0]const u8{"Y"}, &[_]*onnx.Op.Attr{});
    try graph.addNode(node);
    var inputs = [_]*onnx.Value.Info{v_info};
    try graph.setInputs(&inputs);
    var outputs = [_]*onnx.Value.Info{v_out};
    try graph.setOutputs(&outputs);

    try model.addGraph(graph);
    try testing.expectEqual(@as(usize, 1), try graph.getNodeCount());
  }

  test "Op.Attr - Scalar and Array" {
    const a1 = try onnx.Op.Attr.initInt("test", 10);
    defer a1.deinit();
    try testing.expectEqual(onnx.Op.Attr.Type.INT, try a1.getType());

    const a2 = try onnx.Op.Attr.initFloats("f", &[_]f32{ 1.0, 2.0 });
    defer a2.deinit();
    var buf: [8]u8 = undefined;
    _ = try a2.read(.FLOATS, &buf);
  }
};

const Compiler = onnx.Api.compiler;
pub const Compiler_tests = struct {
  test "Compiler - Routing" {
    const opts = try onnx.Session.Options.C.init();
    defer opts.deinit();
    const c_opts = try onnx.Model.CompilationOptions.init(opts);
    defer c_opts.deinit();

    const MockWriter = struct {
      pub fn write(_: *@This(), data: []const u8) !void { _ = data; }
    };
    var mock = MockWriter{};
    try c_opts.setOutputModelWriteFunc(onnx.Model.CompilationOptions.WriteInterface.fromContext(&mock));
  }
};

const IoBinding = onnx.IoBinding;
pub const IoBinding_tests = struct {
  test "IoBinding - Full Routing" {
    const original_api = Api.ort;
    var mock_api = original_api.*;
    Api.ort = &mock_api;
    defer Api.ort = original_api;

    // Helper to overwrite const fields in the vtable copy
    const patch = struct {
      fn do(target: anytype, func: anytype) void {
        @as(*?*const anyopaque, @ptrCast(@constCast(target))).* = @ptrCast(&func);
      }
    };

    patch.do(&mock_api.CreateIoBinding, struct {
      fn mock(_: ?*Api.c.OrtSession, out: [*c]?*Api.c.OrtIoBinding) callconv(.c) ?*Api.c.OrtStatus {
        out.?.* = @ptrFromInt(0xdead); return null;
      }
    }.mock);

    const session: *onnx.Session = @ptrFromInt(0x1);
    const binding = try onnx.IoBinding.init(session);
    
    patch.do(&mock_api.BindInput, struct {
      fn mock(_: ?*Api.c.OrtIoBinding, _: [*c]const u8, _: ?*const Api.c.OrtValue) callconv(.c) ?*Api.c.OrtStatus { return null; }
    }.mock);
    try binding.bindInput("X", @ptrFromInt(0x1));

    patch.do(&mock_api.SynchronizeBoundInputs, struct {
      fn mock(_: ?*Api.c.OrtIoBinding) callconv(.c) ?*Api.c.OrtStatus { return null; }
    }.mock);
    try binding.synchronizeInputs();

    patch.do(&mock_api.ReleaseIoBinding, struct {
      fn mock(_: ?*Api.c.OrtIoBinding) callconv(.c) void {}
    }.mock);
    binding.deinit();
  }
};

const ThreadingOptions = onnx.ThreadingOptions;
pub const ThreadingOptions_tests = struct {
  test "ThreadingInterface - Routing" {
    const MockThreader = struct {
      pub fn create(_: *@This(), _: ?*const fn (?*anyopaque) callconv(.c) void, _: ?*anyopaque) ?*const anyopaque { return @ptrFromInt(0x1); }
      pub fn join(_: *const anyopaque) void {}
    };
    var mock = MockThreader{};
    const interface = onnx.ThreadingOptions.ThreadingInterface.fromContext(&mock);
    const handle = interface.create_fn(interface.ptr, null, null);
    interface.join_fn(handle);
  }
};

const RunOptions = onnx.RunOptions;
pub const RunOptions_tests = struct {
  test "RunOptions - Tags and LoRA" {
    const opts = try onnx.RunOptions.init();
    defer opts.deinit();
    
    try opts.setRunTag("tag");
    if (try opts.getRunTag()) |tag| try testing.expectEqualStrings("tag", std.mem.sliceTo(tag, 0));

    // LoRA fails on "data", but exercises the wrapper and Error.check
    _ = onnx.RunOptions.LoraAdapter.initFromArray("data", null) catch |err| {
      const allowed = err == error.OrtErrorFail or err == error.OrtErrorNotImplemented;
      try testing.expect(allowed);
    };
  }
};

const ProviderOptions = onnx.ProviderOptions;
pub const Provider_tests = struct {
  test "ProviderOptions - Specific Init" {
    const allocator = try onnx.Allocator.getDefault();

    // TensorRT V2
    const trt = onnx.ProviderOptions.TensorRT.init() catch return;
    defer trt.deinit();
    _ = trt.getOptionsAsString(allocator) catch {};

    // CUDA V2
    const cuda = onnx.ProviderOptions.CUDA.init() catch return;
    defer cuda.deinit();
    try cuda.update(&[_][*:0]const u8{"device_id"}, &[_][*:0]const u8{"0"});
    
    // CANN
    const cann = onnx.ProviderOptions.CANN.init() catch return;
    defer cann.deinit();
  }

  test "ProviderOptions - V2 Structs" {
    // These functions might return error.OrtErrorNotImplemented if ORT was built without these EPs.
    // We catch and ignore to ensure the test runner continues.
    if (onnx.ProviderOptions.TensorRT.init()) |trt| {
      defer trt.deinit();
      _ = trt.getByName("device_id") catch {};
    } else |_| {}

    if (onnx.ProviderOptions.CUDA.init()) |cuda| {
      defer cuda.deinit();
      const allocator = try onnx.Allocator.getDefault();
      _ = cuda.getOptionsAsString(allocator) catch {};
    } else |_| {}

    if (onnx.ProviderOptions.CANN.init()) |cann| {
      defer cann.deinit();
    } else |_| {}
  }
};

const Session = onnx.Session;
pub const Session_tests = struct {
  test "Session.Options - EP Appends" {
    const opts = try onnx.Session.Options.C.init();
    defer opts.deinit();

    const Opts = struct { a: [*:0]const u8 = "b" };
    // Coverage for appendExecutionProvider (Generic)
    _ = opts.appendExecutionProvider("CPU", Opts{}) catch {};

    // Coverage for appendExecutionProviderV2 (Device-based)
    const devices = try Api.env.getEpDevices();
    if (devices.len > 0) {
      const dev_slice = [_]*const onnx.Ep.Device{devices[0]};
      _ = opts.appendExecutionProviderV2(&dev_slice, Opts{}) catch {};
    }
  }
};

const KernelContext = onnx.Op.KernelContext;
pub const KernelContext_tests = struct {
  test "KernelContext - parallelFor" {
    const Task = struct {
      pub fn run(_: *@This(), _: usize) callconv(.c) void {}
    };
    
    var run: bool = false;
    _ = &run;
    if (run) {
      var task: Task = .{};
      KernelContext.parallelFor(@ptrCast(@constCast(&.{})), 1, 1, &task) catch |err| {
        try testing.expect(err == error.OrtErrorInvalidArgument);
      };
    }
  }
};

pub const Async_tests = struct {
  test "Session - runAsync Callback Routing" {
    const MockCtx = struct {
      called: bool = false,
      pub fn callback(self: *@This(), _: []?*onnx.Value, _: ?*onnx.Error.Status) void {
        self.called = true;
      }
    };
    var ctx = MockCtx{};
    const session: *onnx.Session = @ptrFromInt(0x1);
    
    const original_api_ptr = Api.ort;
    var mock_api = original_api_ptr.*;
    Api.ort = &mock_api;
    defer Api.ort = original_api_ptr;

    const patch = struct {
      fn mock(_: ?*Api.c.OrtSession, _: ?*const Api.c.OrtRunOptions, _: [*c]const [*c]const u8, _: [*c]const ?*const Api.c.OrtValue, _: usize, _: [*c]const [*c]const u8, _: usize, _: [*c]?*Api.c.OrtValue, cb: Api.c.RunAsyncCallbackFn, cb_ctx: ?*anyopaque) callconv(.c) ?*Api.c.OrtStatus {
        // Trigger callback with dummy pointer
        cb.?(cb_ctx, @ptrFromInt(0x10), 0, @ptrFromInt(0x1));
        return null;
      }
    };
    
    mock_api.RunAsync = @ptrCast(&patch.mock);
    try session.runAsync(null, &.{}, &.{}, &.{}, &.{}, &ctx);
    try testing.expect(ctx.called);
  }
};

pub const Misc_tests = struct {
  test "Api - Telemetry and Language" {
    try Api.env.setTelemetryEventsState(.disable);
    try Api.env.updateCustomLogLevel(.warning);
    try Api.env.setLanguageProjection(.C);
    
    _ = Api.env.getEpDevices() catch {};
    
    // Test Shared Allocator routing
    const devices = try Api.env.getEpDevices();
    if (devices.len > 0) {
      _ = Api.env.createSharedAllocator(devices[0], .DEFAULT, .device, null) catch {};
    }
  }

  test "CompilationOptions - LocationInterface" {
    const MockLoc = struct {
      pub fn getLocation(
        _: *@This(), 
        _: [*:0]const u8, 
        _: *const onnx.Value, 
        _: ?*const onnx.ExternalInitializerInfo
      ) !?*onnx.ExternalInitializerInfo {
        return null;
      }
    };
    var mock = MockLoc{};
    // Exercise the fromContext factory logic
    const interface = onnx.Model.CompilationOptions.LocationInterface.fromContext(&mock);
    
    // Manually invoke the generated VTable wrapper to hit the inner logic
    var out: ?*onnx.Api.c.OrtExternalInitializerInfo = null;
    _ = interface.loc_fn(interface.ptr, "name", @ptrFromInt(0x1), null, &out);
  }

  test "Api.env - Telemetry and Allocators" {
    try Api.env.setTelemetryEventsState(.disable);
    try Api.env.updateCustomLogLevel(.warning);
    try Api.env.setLanguageProjection(.NODEJS);
    
    // This will likely return empty if no EPs are registered, but hits the line
    const devices = try Api.env.getEpDevices();
    if (devices.len > 0) {
      _ = Api.env.createSharedAllocator(devices[0], .DEFAULT, .device, null) catch {};
    }
  }


  test "PrepackedWeightsContainer - Lifecycle" {
    const container = try onnx.PrepackedWeightsContainer.init();
    container.deinit();
  }

  test "Api - Providers and Devices" {
    const p = try Api.getAvailableProviders();
    defer p.deinit() catch {};
    _ = p.has("CPU");
    
    _ = Api.getCurrentGpuDeviceId() catch {};
    _ = Api.getBuildInfoString();
  }
};

//---
//Referencing
//---

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

  return std.meta.Tuple(&argument_field_list);
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
    .array => |ai| inline for (0..ai.len) |i| {retval[i] = initType(ai.child);},
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
    } else if (@typeInfo(@TypeOf(field)) == .@"fn") {
      var should_run: bool = false;
      _ = &should_run;
      if (should_run) {
        if (ArgsTuple(@TypeOf(field))) |Args| {
          _ = &@call(.auto, field, comptime initType(Args));
        } else comptime {
          const name = std.fmt.comptimePrint("{s}.{s}", .{@typeName(T), decl.name});
          // if (skipFunctions.get(name)) @compileError(std.fmt.comptimePrint("Skipping {s}: {s}\n", .{name, @typeName(@TypeOf(field))}));
          if (!skipFunctions.get(name)) @compileError(std.fmt.comptimePrint("Can't call {s}: {s}\n", .{name, @typeName(@TypeOf(field))}));
        }
      }
    }
  }
}

test {
  refAllDeclsRecursiveExcerptC(onnx);
  testing.refAllDeclsRecursive(@This());
}

/// HashMap implementation used internally while parsing.
/// This is used for key replacement (${...})
/// This is a barebones implementation, it uses 8 bits for the fingerprint
/// unlike the 7 in zig's standard hashmap because we don't require toombstones
///
/// I chose to use write this instead of using the standard hashmap because
/// the standard implementation does not work at comptime, and has toombstones
/// which are not needed for this use case. We would need to use a context variant
/// of the hash map to prevent a new allocation for each value and it would result
/// in same amount of bloat more or less. Besides, this implementation should be
/// slightly faster (hopefully;) and works at comptime as well. Also, converting
/// the standard to ComptimeEnvType / EnvType would need rehashing which this
/// implementation does not need.
fn HashMap(is_const: bool) type {
  return struct {
    const Size = u32;
    pub const String = []const u8;
    pub const KV = struct { key: []const u8 };
    const default_max_load_percentage = 64;

    // This is the start of our allocated block
    keys: if (is_const) []const ?String else []?String = &.{},
    // These will be at the end of our allocated block, 0 means unused.
    meta: if (is_const) []const u8 else []u8 = &.{},
    /// Length for our keys, values, and meta arrays
    cap: Size = 0,
    // How many elements are in use
    size: Size = 0,
    // How many elements are available, this is used to reduce the number of instructions needed for the grow check
    available: Size = 0,

    pub fn initSlice(keys: []const []const u8) HashMap(true) {
      var self = @This().init(keys.len * default_max_load_percentage / 100 + 1);
      for (keys) |key| self.put(key);
      return self.toConst();
    }

    pub fn init(cap: Size) @This() {
      @setEvalBranchQuota(1000_000);
      const c = std.math.ceilPowerOfTwo(Size, cap) catch 16;
      return .{
        .keys = blk: { var keys = [_]?String{null} ** c; break :blk &keys; },
        .meta = blk: { var meta = [_]u8{0} ** c; break :blk &meta; },
        .cap = c,
        .available = c * default_max_load_percentage / 100,
      };
    }

    fn getHFP(key: []const u8) std.meta.Tuple(&.{u64, u8}) {
      const h = std.hash_map.StringContext.hash(undefined, key);
      const fp: u8 = @intCast(h >> 56);
      return .{h, fp};
    }

    fn eqlString(string: String, other: []const u8) bool {
      return std.mem.eql(u8, string.ptr[0..string.len], other);
    }

    fn getIndex(self: *const @This(), fingerprint: u8, hash: u64, key: []const u8) usize {
      var i: usize = @intCast(hash & (self.cap - 1));
      while (self.keys[i] != null) : (i = (i + 1) & (self.cap - 1)) {
        if (self.meta[i] == fingerprint and eqlString(self.keys[i].?, key)) break;
      }
      return i;
    }

    pub fn get(self: *const @This(), key: []const u8) bool {
      @setEvalBranchQuota(1000_000);
      const hash, const fingerprint = getHFP(key);
      const i = self.getIndex(fingerprint, hash, key);
      return self.keys[i] != null;
    }

    pub fn put(self: *@This(), key: []const u8) void {
      @setEvalBranchQuota(1000_000);
      self.grow();

      const hash, const fingerprint = getHFP(key);
      const i = self.getIndex(fingerprint, hash, key);
      if (self.keys[i] == null) {
        self.meta[i] = fingerprint;
        self.keys[i] = key;
        self.size += 1;
        self.available -= 1;
      }
    }

    fn grow(old: *@This()) void {
      @setEvalBranchQuota(1000_000);
      if (old.available > old.size) return;
      var self = init(if (old.size == 0) 16 else old.size * 2);
      self.size = old.size;

      for (old.meta, old.keys) |m, k| {
        if (k == null) continue;
        const hash, _ = getHFP(k.?);
        var i: usize = @intCast(hash & (self.cap - 1));
        while (self.keys[i] != null) : (i = (i + 1) & (self.cap - 1)) {}
        self.meta[i] = m;
        self.keys[i] = k;
      }

      old.* = self;
    }

    pub fn toConst(self: *const @This()) HashMap(true) {
      if (is_const) return self.*;
      const keys: [self.keys.len]?String = self.keys[0..self.keys.len].*;
      const meta: [self.meta.len]u8 = self.meta[0..self.meta.len].*;
      return .{
        .keys = &keys,
        .meta = &meta,
        .cap = self.cap,
        .size = self.size,
        .available = self.available,
      };
    }
  };
}

/// These function are called somewhere in the tests
/// TODO: make this better; could mpve this to test runner to make it cleaner
const skipFunctions = HashMap(false).initSlice(&[_][]const u8{
  "root.Logging.Interface.fromContext",
  "root.Model.CompilationOptions.WriteInterface.fromContext",
  "root.ThreadingOptions.ThreadingInterface.fromContext",
  "root.Op.Custom.init",
  "root.Allocator.alloc",
  "root.Allocator.free",
  "root.Utils.apiCast",
  "root.Utils.apiCastTo",
  "root.Utils.cStr",
  "root.Utils.cStrTo",
  "root.Utils.pathCast",
  "root.Utils.pathCastTo",
  "root.Utils.cCast",
  "root.Utils.createOptionsKVL",
  "root.Value.AsUnion",
  "root.Value.asUnion",
  "root.Value.asType",
  "root.Value.Sub.Sequence.toValue",
  "root.Value.Sub.Map.toValue",
  "root.Value.Sub.Optional.toValue",
  "root.Value.Sub.SparseTensor.toValue",
  "root.Value.Sub.Tensor.toValue",
  "root.Value.Sub.Opaque.toValue",
  "root.Value.Sub.Tensor.getData",
  "root.Value.Sub.Tensor.at",
  "root.Value.Sub.Tensor.getDataConst",
  "root.Value.Sub.Opaque.getData",
  "root.Ep.Interface.init",
  "root.Ep.DataTransfer.init",
  "root.Ep.SyncNotificationImpl.init",
  "root.Ep.Factory.init",
  "root.Ep.SyncStream.Impl.init",
  "root.Ep.NodeCompute.Info.init",
  "root.Session.runAsync",
  "root.Op.KernelContext.parallelFor",
  "root.Model.CompilationOptions.LocationInterface.fromContext",
  "root.Session.Options.C.appendExecutionProvider",
  "root.Session.Options.C.appendExecutionProviderV2",
});

