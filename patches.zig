const std = @import("std");

pub const Onnx = struct {
    include: std.Build.LazyPath,
    nn: std.Build.LazyPath,
    old_nn: std.Build.LazyPath,
    rnn: std.Build.LazyPath,
};

/// Applies the source changes in ORT 1.23.2's cmake/patches/onnx/onnx.patch.
/// Inputs belong to a content-hashed dependency; WriteFiles keys generated
/// outputs by their full contents. Never modifies the fetched package.
pub fn onnx(b: *std.Build, dependency: *std.Build.Dependency) Onnx {
    const output = b.addWriteFiles();
    _ = transform(b, output, dependency, "onnx/defs/schema.h", &.{
        .{ "explicit OpSchemaRegisterOnce(", "OpSchemaRegisterOnce(" },
    });
    return .{
        .include = output.getDirectory(),
        .nn = transform(b, output, dependency, "onnx/defs/nn/defs.cc", &.{
            .{ "static void convPoolShapeInference(", "void convPoolShapeInference(" },
            .{ "static void convTransposeShapeInference(", "void convTransposeShapeInference(" },
            .{ "static void globalPoolTypeShapeInference(", "void globalPoolTypeShapeInference(" },
        }),
        .old_nn = transform(b, output, dependency, "onnx/defs/nn/old.cc", &.{
            .{ "    GroupNormalization,\n    18,\n    OpSchema()\n        .Deprecate()\n", "    GroupNormalization,\n    18,\n    OpSchema()\n" },
        }),
        .rnn = transform(b, output, dependency, "onnx/defs/rnn/defs.cc", &.{
            .{ "static void RNNShapeInference(", "void RNNShapeInference(" },
        }),
    };
}

/// Avoids a compiler-runtime CPU table dependency for one feature already
/// available in the current CPUID leaf, matching the upstream non-Linux path.
pub fn cpuid(b: *std.Build, dependency: *std.Build.Dependency) std.Build.LazyPath {
    return transform(b, b.addWriteFiles(), dependency, "onnxruntime/core/common/cpuid_info.cc", &.{
        .{ "has_tpause_ = __builtin_cpu_supports(\"waitpkg\") != 0;", "has_tpause_ = (data[2] & (1 << 5)) != 0;" },
    });
}

/// Recognizes native x86 vendor identity when optional libcpuinfo is absent.
/// Unknown vendors retain upstream's explicit unknown result and warning.
pub fn cpuidVendor(b: *std.Build, dependency: *std.Build.Dependency) std.Build.LazyPath {
    return transform(b, b.addWriteFiles(), dependency, "onnxruntime/core/common/cpuid_info_vendor.cc", &.{
        .{ "#include <string_view>", "#include <string_view>\n#if defined(CPUIDINFO_ARCH_X86) && defined(__GNUC__)\n#include <cpuid.h>\n#endif" },
        .{
            "#endif  // defined(CPUINFO_SUPPORTED)\n    return result;",
            \\#elif defined(CPUIDINFO_ARCH_X86) && defined(__GNUC__)
            \\    unsigned int eax, ebx, ecx, edx;
            \\    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
            \\      if (ebx == signature_INTEL_ebx && edx == signature_INTEL_edx && ecx == signature_INTEL_ecx) {
            \\        result = cpuinfo_vendor_intel;
            \\      } else if (ebx == signature_AMD_ebx && edx == signature_AMD_edx && ecx == signature_AMD_ecx) {
            \\        result = cpuinfo_vendor_amd;
            \\      }
            \\    }
            \\#endif  // defined(CPUINFO_SUPPORTED)
            \\    return result;
            ,
        },
    });
}

fn transform(b: *std.Build, output: *std.Build.Step.WriteFile, dependency: *std.Build.Dependency, path: []const u8, replacements: []const [2][]const u8) std.Build.LazyPath {
    b.dependOnFileContents(dependency.path(path));
    const directory = dependency.builder.root.openDir(b.graph.io, ".", .{}) catch @panic("cannot open pinned native source directory");
    defer directory.close(b.graph.io);
    var text = directory.readFileAlloc(b.graph.io, path, b.allocator, .limited(4 * 1024 * 1024)) catch @panic("cannot read pinned native source");
    for (replacements) |replacement| {
        const start = std.mem.indexOf(u8, text, replacement[0]) orelse @panic("native patch context missing");
        const end = start + replacement[0].len;
        if (std.mem.indexOf(u8, text[end..], replacement[0]) != null) @panic("native patch context is not unique");
        text = std.mem.concat(b.allocator, u8, &.{ text[0..start], replacement[1], text[end..] }) catch @panic("out of memory");
    }
    return output.add(path, text);
}
