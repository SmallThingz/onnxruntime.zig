# onnxruntime-zig

A type-safe, idiomatic, and zero-overhead Zig wrapper for [ONNX Runtime](https://onnxruntime.ai/). 

This library provides a high-level Zig interface to the ONNX Runtime C API, including support for inference, training, the Model Editor, and custom Execution Provider (EP) development.

> [!WARNING]  
> **Development Status:** This project is currently in active development. APIs are subject to change, and some advanced features of ONNX Runtime may still be missing or partially implemented.

## Key Features

- **Idiomatic V-Tables**: Uses `@fieldParentPtr` and `@hasDecl` to allow Zig structs to implement complex C vtables easily.
- **Error Handling**: Automatically converts `OrtStatus` pointers into Zig error sets.
- **Comprehensive Coverage**: Includes bindings for:
    - Inference Session (`Session`)
    - On-Device Training (`Training`)
    - Graph/Model Construction (`Api.editor`)
    - Compile Api (`Api.compiler`)
    - Custom Operators (`Op.Custom`)
    - Execution Provider Plugins (`Ep.Interface` / `Ep.Factory`)

## Getting Started

### Requirements
- Zig `0.15.0` or higher.
- ONNX Runtime shared library (`.so`, `.dll`, or `.dylib`) and headers installed on your system.

### Initialization
You must initialize the global API once before use.

```zig
const onnx = @import("onnxruntime");

pub fn main() !void {
    // Initialize the global environment and API structures
    try onnx.Api.init(.{
        .log_level = .warning,
        .log_id = "my_app",
        .editor = true,   // Enable Model Editor API
        .compiler = false,
    }, .{
        .compile_behavior = .panicking, // Panic if calling uninitialized compile functions
    });
    defer onnx.Api.deinit();
}
```

### Inference Example
```zig
const session_opts = onnx.Session.Options{
    .optimization_level = .ALL,
};
const c_opts = try session_opts.c();
defer c_opts.deinit();

// Load model
var session = try ort.Session.initZ("model.onnx", c_opts);
defer session.deinit();

// Run inference
try session.run(
    null,           // RunOptions
    &.{"input1"},   // Input names
    &.{input_val},  // Input OrtValues
    &.{"output1"},  // Output names
    &.{output_val}, // Output OrtValues
);
```

## Project Structure

- `Api`: Global entry point and library management.
- `Session`: Core inference functionality.
- `Training`: On-device training extension.
- `Value`: Handle Tensors, Sequences, and Maps.
- `Ep`: Interfaces for building Execution Provider plugins.

## License

This project is licensed under the MIT License. Reference the ONNX Runtime license for the underlying C library.
