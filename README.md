# onnxruntime-zig

A type-safe, idiomatic, and effectively zero-overhead Zig wrapper for [ONNX Runtime](https://onnxruntime.ai/). 

This library provides a high-level Zig interface to the ONNX Runtime C API, including support for inference, training, the Model Editor, and custom Execution Provider (EP) development.

> [!IMPORTANT]  
> **Development Status:** Some advanced features of ONNX Runtime may be missing or partially implemented.

## Key Features

- **Full API Support**: Support for Inference, On-Device Training, Model Compilation, and the Model Editor API.
- **Idiomatic V-Tables**: Seamlessly implement custom Execution Providers (EPs) or Operators (CustomOps) using standard Zig structs.
- **Zig-Native Error Handling**: `OrtStatus` pointers are automatically converted to Zig error sets like `error.OrtErrorInvalidArgument`.
- **Zero-Copy Performance**: Direct access to Tensor data and support for `IoBinding` for GPU-accelerated, zero-copy inference.

## Setup

### Requirements
 - **Zig Version:** `0.15.0` or higher.  
 - **ORT Version:** Supports features up to ORT `1.23` (including SyncStreams and new EP interfaces).

### Installation
Add `onnxruntime-zig` to your `build.zig.zon` and then add the module to your `build.zig`:

```zig
const onnx_mod = b.dependency("onnxruntime", .{
  .target = target,
  .optimize = optimize,
}).module("onnxruntime");

exe.root_module.addImport("onnxruntime", onnx_mod);
exe.linkSystemLibrary("onnxruntime");
```

### Initialization
You must initialize the global API once before use.

```zig
const onnx = @import("onnxruntime");

pub fn main() !void {
  // Initialize the global environment and API structures
  try onnx.Api.init(.{
    .log_level = .warning,
    .log_id = "my_app",
    .editor = true, // Enable Model Editor API
    .compiler = false, // disabled by default, you can omit this
  }, .{
    .compile_behavior = .panicking, // Panic if calling uninitialized compile functions
  });

  defer onnx.Api.deinit(); // cleanup after you are done using the api
}
```

### 2. Inference
```zig
const allocator = try onnx.Allocator.getDefault();

// Load a session
const c_opts = try onnx.Session.Options{ .optimization_level = .ALL }.c();
defer c_opts.deinit();
var session = try onnx.Session.initZ("model.onnx", c_opts);
defer session.deinit();

// Prepare input tensor [1, 3]
const dims = [_]i64{ 1, 3 };
const input_val = try onnx.Value.Sub.Tensor.init(allocator, &dims, .f32);
defer input_val.deinit();

const data = try input_val.getData(f32);
@memcpy(data, &[_]f32{ 1.0, 2.0, 3.0 });

// Run
var output_val: ?*onnx.Value = null;
try session.run(null, &.{"input"}, &.{input_val.toValue()}, &.{"output"}, &.{output_val});
```

## Advanced Usage

### Custom Operators
You can implement native ONNX operators directly in Zig:

```zig
const MyOp = struct {
  ort_op: onnx.Op.Custom,
  
  pub fn getName(_: *const @This()) [*:0]const u8 { return "MyCustomOp"; }
  pub fn getInputTypeCount(_: *const @This()) usize { return 1; }
  pub fn getOutputTypeCount(_: *const @This()) usize { return 1; }
  
  pub fn createKernelV2(_: *const @This(), _: *const onnx.Api.ort, _: *const onnx.Op.KernelInfo) !*anyopaque {
    return @ptrFromInt(0x1); // Return your kernel state
  }

  pub fn computeV2(kernel_state: *anyopaque, ctx: *onnx.Op.KernelContext) !void {
    const input = (try ctx.getInput(0)).?;
    // Your logic here...
  }
  
  pub fn destroyKernel(kernel_state: *anyopaque) void { _ = kernel_state; }
};
```

### Graph Surgery (Model Editor)
Modify existing models or build new ones at runtime:

```zig
const graph = try onnx.Graph.init();
const node = try onnx.Node.init(
  "Relu", "", "my_node", 
  &.{"X"}, &.{"Y"}, &.{}
);
try graph.addNode(node);
try graph.setInputs(&.{val_info_x});
try graph.setOutputs(&.{val_info_y});
```

## License

This project is licensed under the MIT License. Reference the ONNX Runtime license for the underlying C library.
