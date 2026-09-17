const std = @import("std");

/// Complete upstream C API for interoperability; normal use needs no C pointers.
pub const raw = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

pub const Error = error{
    OutOfMemory,
    VersionMismatch,
    Failure,
    InvalidArgument,
    NoSuchFile,
    NoModel,
    EngineError,
    RuntimeException,
    InvalidProtobuf,
    ModelLoaded,
    NotImplemented,
    InvalidGraph,
    ProviderFailure,
    InvalidShape,
    ShapeMismatch,
    TypeMismatch,
    NotTensor,
    InvalidName,
    IndexOutOfBounds,
    InvalidUtf8,
};

fn apiTable() Error!*const raw.OrtApi {
    const base = raw.OrtGetApiBase() orelse return error.VersionMismatch;
    return base.*.GetApi.?(raw.ORT_API_VERSION) orelse error.VersionMismatch;
}

fn check(api: *const raw.OrtApi, status: ?*raw.OrtStatus) Error!void {
    const failure = status orelse return;
    defer api.ReleaseStatus.?(failure);
    return switch (api.GetErrorCode.?(failure)) {
        raw.ORT_INVALID_ARGUMENT => error.InvalidArgument,
        raw.ORT_NO_SUCHFILE => error.NoSuchFile,
        raw.ORT_NO_MODEL => error.NoModel,
        raw.ORT_ENGINE_ERROR => error.EngineError,
        raw.ORT_RUNTIME_EXCEPTION => error.RuntimeException,
        raw.ORT_INVALID_PROTOBUF => error.InvalidProtobuf,
        raw.ORT_MODEL_LOADED => error.ModelLoaded,
        raw.ORT_NOT_IMPLEMENTED => error.NotImplemented,
        raw.ORT_INVALID_GRAPH => error.InvalidGraph,
        raw.ORT_EP_FAIL => error.ProviderFailure,
        else => error.Failure,
    };
}

pub fn version() []const u8 {
    return std.mem.span(raw.OrtGetApiBase().?.*.GetVersionString.?());
}

pub const LogLevel = enum(c_uint) {
    verbose = raw.ORT_LOGGING_LEVEL_VERBOSE,
    info = raw.ORT_LOGGING_LEVEL_INFO,
    warning = raw.ORT_LOGGING_LEVEL_WARNING,
    err = raw.ORT_LOGGING_LEVEL_ERROR,
    fatal = raw.ORT_LOGGING_LEVEL_FATAL,
};

pub const Environment = struct {
    api: *const raw.OrtApi,
    handle: *raw.OrtEnv,

    pub const Options = struct { name: []const u8 = "onnxruntime.zig", log_level: LogLevel = .warning };

    pub fn init(allocator: std.mem.Allocator, options: Options) Error!Environment {
        const api = try apiTable();
        const name = try terminated(allocator, options.name);
        defer allocator.free(name);
        var handle: ?*raw.OrtEnv = null;
        try check(api, api.CreateEnv.?(@intFromEnum(options.log_level), name.ptr, &handle));
        return .{ .api = api, .handle = handle.? };
    }

    /// All models created from this environment must be released first.
    pub fn deinit(self: *Environment) void {
        self.api.ReleaseEnv.?(self.handle);
        self.* = undefined;
    }

    /// The serialized model is borrowed only during this call.
    pub fn load(self: *const Environment, bytes: []const u8, options: Model.Options) Error!Model {
        if (bytes.len == 0) return error.InvalidProtobuf;
        const configured = try options.create(self.api);
        defer self.api.ReleaseSessionOptions.?(configured);
        var session: ?*raw.OrtSession = null;
        try check(self.api, self.api.CreateSessionFromArray.?(self.handle, bytes.ptr, bytes.len, configured, &session));
        return .{ .api = self.api, .handle = session.? };
    }

    /// Native file loading also resolves a model's external weight files.
    pub fn open(self: *const Environment, allocator: std.mem.Allocator, path: []const u8, options: Model.Options) Error!Model {
        const configured = try options.create(self.api);
        defer self.api.ReleaseSessionOptions.?(configured);
        if (std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidName;
        var session: ?*raw.OrtSession = null;
        if (@import("builtin").os.tag == .windows) {
            const wide = try std.unicode.utf8ToUtf16LeAllocZ(allocator, path);
            defer allocator.free(wide);
            try check(self.api, self.api.CreateSession.?(self.handle, wide.ptr, configured, &session));
        } else {
            const terminated_path = try allocator.dupeZ(u8, path);
            defer allocator.free(terminated_path);
            try check(self.api, self.api.CreateSession.?(self.handle, terminated_path.ptr, configured, &session));
        }
        return .{ .api = self.api, .handle = session.? };
    }
};

pub const Optimization = enum(c_uint) {
    disabled = raw.ORT_DISABLE_ALL,
    basic = raw.ORT_ENABLE_BASIC,
    extended = raw.ORT_ENABLE_EXTENDED,
    all = raw.ORT_ENABLE_ALL,
};

pub const Model = struct {
    api: *const raw.OrtApi,
    handle: *raw.OrtSession,

    pub const Options = struct {
        intra_op_threads: u16 = 1,
        inter_op_threads: u16 = 1,
        optimization: Optimization = .all,

        fn create(self: Options, api: *const raw.OrtApi) Error!*raw.OrtSessionOptions {
            if (self.intra_op_threads == 0 or self.inter_op_threads == 0) return error.InvalidArgument;
            var options: ?*raw.OrtSessionOptions = null;
            try check(api, api.CreateSessionOptions.?(&options));
            errdefer api.ReleaseSessionOptions.?(options.?);
            try check(api, api.SetIntraOpNumThreads.?(options.?, self.intra_op_threads));
            try check(api, api.SetInterOpNumThreads.?(options.?, self.inter_op_threads));
            try check(api, api.SetSessionGraphOptimizationLevel.?(options.?, @intFromEnum(self.optimization)));
            return options.?;
        }
    };

    pub fn deinit(self: *Model) void {
        self.api.ReleaseSession.?(self.handle);
        self.* = undefined;
    }

    pub fn inputNames(self: *const Model, allocator: std.mem.Allocator) Error!Names {
        return self.names(allocator, true);
    }

    pub fn outputNames(self: *const Model, allocator: std.mem.Allocator) Error!Names {
        return self.names(allocator, false);
    }

    fn names(self: *const Model, allocator: std.mem.Allocator, comptime input: bool) Error!Names {
        var count: usize = 0;
        const count_fn = if (input) self.api.SessionGetInputCount.? else self.api.SessionGetOutputCount.?;
        const name_fn = if (input) self.api.SessionGetInputName.? else self.api.SessionGetOutputName.?;
        try check(self.api, count_fn(self.handle, &count));
        const native_allocator = try defaultAllocator(self.api);
        const items = try allocator.alloc([]const u8, count);
        errdefer allocator.free(items);
        var initialized: usize = 0;
        errdefer for (items[0..initialized]) |name| allocator.free(name);
        for (items) |*name| {
            var native_name: [*c]u8 = null;
            try check(self.api, name_fn(self.handle, initialized, native_allocator, &native_name));
            defer native_allocator.Free.?(native_allocator, native_name);
            name.* = try allocator.dupe(u8, std.mem.span(native_name));
            initialized += 1;
        }
        return .{ .allocator = allocator, .items = items };
    }

    /// Runs synchronously and returns all model outputs in model order.
    pub fn run(self: *Model, allocator: std.mem.Allocator, inputs: []const NamedTensor) Error!Outputs {
        var names_ = try self.outputNames(allocator);
        defer names_.deinit();
        return self.runSelected(allocator, inputs, names_.items);
    }

    pub fn runSelected(self: *Model, allocator: std.mem.Allocator, inputs: []const NamedTensor, output_names: []const []const u8) Error!Outputs {
        if (output_names.len == 0) return error.InvalidArgument;
        const input_names = try allocator.alloc([:0]u8, inputs.len);
        defer allocator.free(input_names);
        var initialized: usize = 0;
        defer for (input_names[0..initialized]) |name| allocator.free(name);
        const input_ptrs = try allocator.alloc([*c]const u8, inputs.len);
        defer allocator.free(input_ptrs);
        const input_values = try allocator.alloc(?*const raw.OrtValue, inputs.len);
        defer allocator.free(input_values);
        for (inputs, 0..) |input, index| {
            for (inputs[0..index]) |previous| {
                if (std.mem.eql(u8, previous.name, input.name)) return error.InvalidName;
            }
            input_names[index] = try terminated(allocator, input.name);
            initialized += 1;
            input_ptrs[index] = input_names[index].ptr;
            input_values[index] = input.tensor.handle;
        }
        const out_names = try allocator.alloc([:0]u8, output_names.len);
        defer allocator.free(out_names);
        var out_initialized: usize = 0;
        defer for (out_names[0..out_initialized]) |name| allocator.free(name);
        const out_ptrs = try allocator.alloc([*c]const u8, output_names.len);
        defer allocator.free(out_ptrs);
        for (output_names, 0..) |name, index| {
            out_names[index] = try terminated(allocator, name);
            out_initialized += 1;
            out_ptrs[index] = out_names[index].ptr;
        }
        const native_outputs = try allocator.alloc(?*raw.OrtValue, output_names.len);
        defer allocator.free(native_outputs);
        @memset(native_outputs, null);
        var transferred = false;
        defer if (!transferred) {
            for (native_outputs) |value| if (value) |handle| self.api.ReleaseValue.?(handle);
        };
        const values = try allocator.alloc(Tensor, output_names.len);
        errdefer allocator.free(values);
        try check(self.api, self.api.Run.?(self.handle, null, input_ptrs.ptr, input_values.ptr, inputs.len, out_ptrs.ptr, output_names.len, native_outputs.ptr));
        for (native_outputs, values) |native, *value| {
            const handle = native orelse return error.Failure;
            var tensor: c_int = 0;
            try check(self.api, self.api.IsTensor.?(handle, &tensor));
            if (tensor == 0) return error.NotTensor;
            value.* = .{ .api = self.api, .handle = handle };
        }
        transferred = true;
        return .{ .allocator = allocator, .values = values };
    }
};

pub const NamedTensor = struct { name: []const u8, tensor: *const Tensor };

pub const Names = struct {
    allocator: std.mem.Allocator,
    items: [][]const u8,
    pub fn deinit(self: *Names) void {
        for (self.items) |name| self.allocator.free(name);
        self.allocator.free(self.items);
        self.* = undefined;
    }
};

pub const Outputs = struct {
    allocator: std.mem.Allocator,
    values: []Tensor,
    pub fn deinit(self: *Outputs) void {
        for (self.values) |*value| value.deinit();
        self.allocator.free(self.values);
        self.* = undefined;
    }
};

pub const Tensor = struct {
    api: *const raw.OrtApi,
    handle: *raw.OrtValue,

    /// Copies input data into a native owned tensor; input slices may then die.
    pub fn fromSlice(comptime T: type, shape_: []const i64, source: []const T) Error!Tensor {
        const api = try apiTable();
        const expected = try elementCount(shape_);
        if (expected != source.len) return error.ShapeMismatch;
        var handle: ?*raw.OrtValue = null;
        try check(api, api.CreateTensorAsOrtValue.?(try defaultAllocator(api), shape_.ptr, shape_.len, elementType(T), &handle));
        var result: Tensor = .{ .api = api, .handle = handle.? };
        errdefer result.deinit();
        @memcpy(try result.data(T), source);
        return result;
    }

    /// Explicit zero-copy input. The backing slice must outlive this tensor and
    /// every run using it. Deinitialization releases the handle, never the slice.
    pub fn borrowSlice(comptime T: type, shape_: []const i64, backing: []T) Error!Tensor {
        if (try elementCount(shape_) != backing.len) return error.ShapeMismatch;
        const byte_len = std.math.mul(usize, backing.len, @sizeOf(T)) catch return error.InvalidShape;
        const api = try apiTable();
        var memory: ?*raw.OrtMemoryInfo = null;
        try check(api, api.CreateCpuMemoryInfo.?(raw.OrtArenaAllocator, raw.OrtMemTypeDefault, &memory));
        defer api.ReleaseMemoryInfo.?(memory.?);
        var handle: ?*raw.OrtValue = null;
        try check(api, api.CreateTensorWithDataAsOrtValue.?(memory.?, backing.ptr, byte_len, shape_.ptr, shape_.len, elementType(T), &handle));
        return .{ .api = api, .handle = handle.? };
    }

    pub fn deinit(self: *Tensor) void {
        self.api.ReleaseValue.?(self.handle);
        self.* = undefined;
    }

    /// Borrowed mutable data, valid until tensor/output deinitialization.
    pub fn data(self: *const Tensor, comptime T: type) Error![]T {
        const info = try self.typeInfo();
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(info);
        var element: raw.ONNXTensorElementDataType = 0;
        try check(self.api, self.api.GetTensorElementType.?(info, &element));
        if (element != elementType(T)) return error.TypeMismatch;
        var count: usize = 0;
        try check(self.api, self.api.GetTensorShapeElementCount.?(info, &count));
        if (count == 0) return &.{};
        var pointer: ?*anyopaque = null;
        try check(self.api, self.api.GetTensorMutableData.?(self.handle, &pointer));
        const data_ptr: [*]T = @ptrCast(@alignCast(pointer orelse return error.Failure));
        return data_ptr[0..count];
    }

    pub fn shape(self: *const Tensor, allocator: std.mem.Allocator) Error![]i64 {
        const info = try self.typeInfo();
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(info);
        var rank: usize = 0;
        try check(self.api, self.api.GetDimensionsCount.?(info, &rank));
        const dimensions = try allocator.alloc(i64, rank);
        errdefer allocator.free(dimensions);
        try check(self.api, self.api.GetDimensions.?(info, dimensions.ptr, rank));
        return dimensions;
    }

    fn typeInfo(self: *const Tensor) Error!*raw.OrtTensorTypeAndShapeInfo {
        var info: ?*raw.OrtTensorTypeAndShapeInfo = null;
        try check(self.api, self.api.GetTensorTypeAndShape.?(self.handle, &info));
        return info.?;
    }
};

fn elementType(comptime T: type) raw.ONNXTensorElementDataType {
    return switch (T) {
        f16 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        f32 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        f64 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        i8 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        i16 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        i32 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        i64 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        u8 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        u16 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        u32 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        u64 => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        bool => raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        else => @compileError("unsupported tensor element type: " ++ @typeName(T)),
    };
}

fn elementCount(shape: []const i64) Error!usize {
    var count: usize = 1;
    for (shape) |dimension| {
        if (dimension < 0) return error.InvalidShape;
        const size = std.math.cast(usize, dimension) orelse return error.InvalidShape;
        count = std.math.mul(usize, count, size) catch return error.InvalidShape;
    }
    return count;
}

fn defaultAllocator(api: *const raw.OrtApi) Error!*raw.OrtAllocator {
    var allocator: ?*raw.OrtAllocator = null;
    try check(api, api.GetAllocatorWithDefaultOptions.?(&allocator));
    return allocator.?;
}

fn terminated(allocator: std.mem.Allocator, name: []const u8) Error![:0]u8 {
    if (std.mem.indexOfScalar(u8, name, 0) != null) return error.InvalidName;
    return allocator.dupeZ(u8, name);
}
