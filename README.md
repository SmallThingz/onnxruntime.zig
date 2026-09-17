# onnxruntime.zig

Owned models and typed tensors for Zig 0.16.0, backed by ONNX Runtime 1.23.2. The package builds pinned upstream C++ sources and dependencies directly with `std.Build`. It does not invoke upstream CMake, Make, Python build scripts, or an installed ONNX Runtime library.

```zig
const std = @import("std");
const ort = @import("onnxruntime");

fn infer(allocator: std.mem.Allocator, model_bytes: []const u8) !void {
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    var model = try environment.load(model_bytes, .{});
    defer model.deinit();
    var x = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var y = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 4, 5, 6 });
    defer y.deinit();
    var outputs = try model.run(allocator, &.{
        .{ .name = "x", .tensor = &x },
        .{ .name = "y", .tensor = &y },
    });
    defer outputs.deinit();
    const result = try outputs.values[0].data(f32);
    std.debug.print("{any}\n", .{result});
}
```

## Build

The source build currently targets x86_64 Linux CPU inference, including ONNX Runtime's CPU/contrib kernels. GPU execution providers, training, and other target platforms are not enabled. First builds compile a substantial C++ dependency graph; subsequent builds reuse Zig's cache. See `zig build --help` for build options.

```sh
zig build test -j2 -Doptimize=ReleaseFast
zig build example -j2 -Doptimize=ReleaseFast
zig build test -j2 -Doptimize=Debug -Dnative-optimize=ReleaseFast
zig build test -j2 -Doptimize=ReleaseSafe -Dnative-optimize=ReleaseFast
```

`native-optimize` defaults to `optimize`. Set it separately to keep the expensive native library cached while checking the Zig wrapper in Debug or ReleaseSafe; this does not enable safety instrumentation in the ReleaseFast C++ library.

The native graph builds the host protobuf compiler as a Zig build artifact and generates model schemas through tracked build steps. No prebuilt inference binary is substituted. Keep target instruction workarounds separate from validation on CPUs that lack those instructions.

## API and ownership

There is no mutable global API initialization. `Environment.init` obtains and checks the linked runtime's API table. Release all models before their environment. Native handles are owned: do not copy their owning structs or deinitialize them twice.

- `environment.load(bytes, options)` borrows serialized ONNX bytes during loading.
- `environment.open(allocator, path, options)` accepts a normal path slice and lets native file loading resolve external weight files.
- `Model.Options` provides typed graph optimization, sequential/parallel execution mode, and intra/inter-op thread counts (both default to one). Inter-op parallelism is active only in parallel execution mode.
- `model.inputNames(allocator)` / `outputNames(allocator)` return a `Names` owner containing ordinary string slices; call `deinit`.
- `Tensor.fromSlice(T, dimensions, values)` validates nonnegative dimensions, checked element counts, and matching data length, then **copies** values into native owned storage. An empty shape is a scalar; zero dimensions describe empty tensors.
- `Tensor.borrowSlice(T, dimensions, backing)` explicitly borrows mutable caller storage. Keep it alive through all runs and tensor deinitialization. Releasing this tensor never frees the backing storage.
- `tensor.data(T)` checks the actual native element type and returns a borrowed mutable slice; `tensor.shape(allocator)` returns caller-owned dimensions.
- `model.run(allocator, inputs)` returns all outputs in model order. `runSelected(allocator, inputs, names)` selects named outputs. Input/output names are ordinary slices and embedded NULs/duplicate input names are rejected.
- `Outputs.deinit` releases every output tensor and the containing slice. Do not separately deinitialize tensors still owned by `Outputs`.

Inference is synchronous. Retain tensors during runs and synchronize shared mutable tensor data. The typed API supports numeric and boolean tensors; non-tensor outputs return `NotTensor`. Full upstream interoperability remains available through `raw`, including operations outside this focused surface.

Zig allocators own name lists, path conversions, run arrays, and metadata snapshots. ONNX Runtime owns native environment/session/tensor allocations through its native allocator. Allocation-failure tests cover the Zig-owned allocations; they do not claim to intercept every C++ allocation.

Native status objects are always released and mapped to typed Zig errors. `TypeMismatch`, `ShapeMismatch`, `InvalidShape`, and `InvalidName` distinguish local validation failures from runtime errors such as `InvalidGraph`, `InvalidProtobuf`, and `InvalidArgument`.

## Migration

The old global C-shaped API has been replaced. Import `onnxruntime` and use `Environment`, `Model`, `Tensor`, and `Outputs` instead of manually managing C out-parameters or calling global initialization. For advanced C-level features use `raw.OrtGetApiBase`; its handles follow upstream ownership rules.

Add the package at a pinned commit, then wire its module:

```zig
const dep = b.dependency("onnxruntime", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("onnxruntime", dep.module("onnxruntime"));
```

Tests run real embedded Add and MatMul→Relu models, including parallel execution and optimized/unoptimized matrix inference. They assert exact values, tensor type/shape validation, copied and borrowed storage lifetimes, invalid model/name/input failures, and cleanup at every Zig allocation failure. [Fixture description](tests/README.md).

## Licenses

The Zig wrapper uses the repository MIT license. Native ONNX Runtime and its dependencies retain their upstream licenses. `zig build` installs ONNX Runtime's `LICENSE` and `ThirdPartyNotices.txt` under `share/licenses/onnxruntime`. Applications importing the module must carry the applicable upstream notices when distributing native binaries; dependency install steps are not automatically part of the application's install step.
