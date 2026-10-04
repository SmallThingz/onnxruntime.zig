const std = @import("std");
const sources = @import("sources.zig");
const patches = @import("patches.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const native_optimize = b.option(std.builtin.Optimize, "native-optimize", "C++ runtime optimization (defaults to optimize)") orelse optimize;
    if (target.result.os.tag != .linux or target.result.cpu.arch != .x86_64)
        @panic("The native CPU build currently supports x86_64 Linux; other targets are not yet qualified");
    const ort = b.dependency("ort", .{});
    const pb = b.dependency("protobuf", .{});
    const onnx = b.dependency("onnx", .{});
    const absl = b.dependency("abseil_cpp", .{});
    const re2 = b.dependency("re2", .{});
    const release_flags: []const []const u8 = &.{ "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations" };
    const flags = joinFlags(b, if (native_optimize == .debug) &.{ "-std=c++17", "-Wno-deprecated-declarations" } else release_flags, &.{"-Wno-deprecated-literal-operator"});
    const proto_host = cppLibrary(b, "protobuf-host", b.graph.host, .fast);
    proto_host.root_module.addIncludePath(pb.path("src"));
    proto_host.root_module.addCSourceFiles(.{ .root = pb.path(""), .files = sources.libprotobuf, .flags = release_flags });
    const protoc = b.addExecutable(.{ .name = "protoc", .root_module = b.createModule(.{ .target = b.graph.host, .optimize = .fast, .link_libcpp = true }), .use_llvm = true, .use_lld = true });
    protoc.root_module.addIncludePath(pb.path("src"));
    protoc.root_module.addCSourceFiles(.{ .root = pb.path(""), .files = sources.libprotoc, .flags = release_flags });
    protoc.root_module.addCSourceFile(.{ .file = pb.path("src/google/protobuf/compiler/main.cc"), .flags = release_flags });
    protoc.root_module.linkLibrary(proto_host);
    const generate = b.addRunArtifact(protoc);
    generate.addPrefixedDirectoryArg("-I", onnx.path(""));
    const generated = generate.addPrefixedOutputDirectoryArg("--cpp_out=", "onnx-protos");
    for ([_][]const u8{ "onnx/onnx-ml.proto", "onnx/onnx-operators-ml.proto", "onnx/onnx-data.proto" }) |path| generate.addFileArg(onnx.path(path));

    const lib = cppLibrary(b, "onnxruntime", target, native_optimize);
    const m = lib.root_module;
    const patched_onnx = patches.onnx(b, onnx);
    m.addIncludePath(patched_onnx.include);
    m.linkSystemLibrary("dl", .{});
    m.linkSystemLibrary("pthread", .{});
    const config = b.addWriteFiles();
    m.addIncludePath(config.add("onnxruntime_config.h", "#pragma once\n#define ORT_VERSION \"1.23.2\"\n#define ORT_BUILD_INFO \"Zig native CPU build\"\n").dirname());
    _ = config.add("onnxruntime/core/session/onnxruntime_config.h", "#pragma once\n");
    _ = config.add("onnx/version.h", "#pragma once\n#define ONNX_VERSION \"1.18.0\"\n");
    _ = config.add("onnxruntime/core/session/onnxruntime_version.h", "#pragma once\n#define ORT_VERSION \"1.23.2\"\n");
    m.addIncludePath(generated);
    for ([_][]const u8{ "", "onnxruntime", "include/onnxruntime", "include/onnxruntime/core/session", "onnxruntime/core/mlas/inc", "onnxruntime/core/mlas/lib" }) |path| m.addIncludePath(ort.path(path));
    m.addIncludePath(pb.path("src"));
    m.addIncludePath(onnx.path(""));
    m.addIncludePath(absl.path(""));
    m.addIncludePath(re2.path(""));
    const includes = .{ .{ "eigen", "" }, .{ "flatbuffers", "include" }, .{ "json", "include" }, .{ "microsoft_gsl", "include" }, .{ "mp11", "include" }, .{ "safeint", "" }, .{ "date", "include" } };
    inline for (includes) |pair| m.addIncludePath(b.dependency(pair[0], .{}).path(pair[1]));
    m.addCMacro("ONNX_NAMESPACE", "onnx");
    m.addCMacro("ONNX_ML", "1");
    m.addCMacro("ORT_STATIC_LIB", "1");
    m.addCMacro("EIGEN_MPL2_ONLY", "1");
    m.addCMacro("EIGEN_USE_THREADS", "1");
    m.addCMacro("PLATFORM_POSIX", "1");
    m.addCMacro("USE_RE2", "1");
    m.addCSourceFiles(.{ .root = pb.path(""), .files = sources.libprotobuf, .flags = flags });
    const onnx_flags = joinFlags(b, flags, &.{"-D__ONNX_DISABLE_STATIC_REGISTRATION"});
    m.addCSourceFiles(.{ .root = onnx.path(""), .files = sources.onnx, .flags = onnx_flags });
    for ([_]std.Build.LazyPath{ patched_onnx.nn, patched_onnx.old_nn, patched_onnx.rnn }) |source| m.addCSourceFile(.{ .file = source, .flags = onnx_flags });
    for ([_][]const u8{ "onnx/onnx-ml.pb.cc", "onnx/onnx-operators-ml.pb.cc", "onnx/onnx-data.pb.cc" }) |path| m.addCSourceFile(.{ .file = generated.path(b, path), .flags = flags });
    m.addCSourceFiles(.{ .root = absl.path(""), .files = sources.abseil, .flags = flags });
    m.addCSourceFiles(.{ .root = re2.path(""), .files = sources.re2, .flags = flags });
    m.addCSourceFiles(.{ .root = ort.path(""), .files = sources.ort, .flags = flags });
    m.addCSourceFile(.{ .file = patches.cpuid(b, ort), .flags = flags });
    m.addCSourceFile(.{ .file = patches.cpuidVendor(b, ort), .flags = flags });
    m.addCSourceFile(.{ .file = b.path("native/spin_pause.cc"), .flags = isaFlags(b, flags, &.{"-mwaitpkg"}) });
    m.addCSourceFiles(.{ .root = ort.path(""), .files = sources.mlas, .flags = flags });
    // Preserve MLAS runtime dispatch: ISA flags apply only to selected kernels.
    // AVX-VNNI assembly is separate so AVX2 C++ kernels stay AVX2-compatible.
    const groups = .{
        .{ sources.mlas_amx, &[_][]const u8{ "-mevex512", "-mavx2", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mavx512f" } },
        .{ sources.mlas_sse2, &[_][]const u8{"-msse2"} },
        .{ sources.mlas_avx, &[_][]const u8{"-mavx"} },
        .{ sources.mlas_avx2, &[_][]const u8{ "-mavx2", "-mfma", "-mf16c", "-mno-avxvnni" } },
        .{ sources.mlas_avxvnni, &[_][]const u8{ "-mavx2", "-mavxvnni" } },
        .{ sources.mlas_avx512f, &[_][]const u8{ "-mevex512", "-mavx512f" } },
        .{ sources.mlas_avx512core_cpp, &[_][]const u8{ "-mevex512", "-mfma", "-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mno-avx512vnni", "-mno-avxvnni" } },
        .{ sources.mlas_avx512core, &[_][]const u8{ "-mevex512", "-mfma", "-mavx512vnni", "-mavx512bw", "-mavx512dq", "-mavx512vl" } },
        .{ sources.mlas_avx512vnni, &[_][]const u8{ "-mevex512", "-mfma", "-mavx512vnni", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mavx512f" } },
    };
    inline for (groups) |group| m.addCSourceFiles(.{ .root = ort.path(""), .files = group[0], .flags = isaFlags(b, flags, group[1]) });
    b.installArtifact(lib);
    b.getInstallStep().dependOn(&b.addInstallFile(ort.path("LICENSE"), "share/licenses/onnxruntime/LICENSE").step);
    b.getInstallStep().dependOn(&b.addInstallFile(ort.path("ThirdPartyNotices.txt"), "share/licenses/onnxruntime/ThirdPartyNotices.txt").step);
    const mod = b.addModule("onnxruntime", .{ .root_source_file = b.path("root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    mod.addIncludePath(ort.path("include/onnxruntime/core/session"));
    mod.linkLibrary(lib);
    const bindings = b.addTranslateC(.{
        .root_source_file = ort.path("include/onnxruntime/core/session/onnxruntime_c_api.h"),
        .target = target,
        .optimize = optimize,
    });
    bindings.addIncludePath(ort.path("include/onnxruntime/core/session"));
    mod.addImport("onnxruntime_c", bindings.createModule());
    b.step("bindings", "Generate Zig declarations from the pinned C headers").dependOn(&bindings.step);
    const check_module = b.createModule(.{
        .root_source_file = b.path("root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "onnxruntime_c", .module = bindings.createModule() }},
    });
    const wrapper_check = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "onnxruntime", .module = check_module }},
    }), .use_llvm = true });
    b.step("check", "Type-check the wrapper and tests without building native libraries").dependOn(&wrapper_check.step);
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("tests/test.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "onnxruntime", .module = mod }} }), .use_llvm = true, .use_lld = true });
    const run_tests = b.addRunArtifact(tests);
    run_tests.setCwd(b.path(""));
    b.step("test", "Run real CPU inference and ownership tests").dependOn(&run_tests.step);
    const example = b.addExecutable(.{ .name = "inference", .root_module = b.createModule(.{ .root_source_file = b.path("examples/inference.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "onnxruntime", .module = mod }} }), .use_llvm = true, .use_lld = true });
    const run_example = b.addRunArtifact(example);
    run_example.setCwd(b.path(""));
    b.step("example", "Run the owned-tensor inference example").dependOn(&run_example.step);
}

fn cppLibrary(b: *std.Build, name: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize) *std.Build.Step.Compile {
    return b.addLibrary(.{ .name = name, .linkage = .static, .root_module = b.createModule(.{ .target = target, .optimize = optimize, .link_libcpp = true }), .use_llvm = true, .use_lld = true });
}

fn joinFlags(b: *std.Build, common: []const []const u8, specific: []const []const u8) []const []const u8 {
    return std.mem.concat(b.allocator, []const u8, &.{ common, specific }) catch @panic("out of memory");
}

// Zig 0.16 explicitly forwards disabled host features to cc1 after driver -m
// options. Override only these dispatched translation units at the cc1 level;
// promoting the CPU target of the whole library would be unsafe.
fn isaFlags(b: *std.Build, common: []const []const u8, specific: []const []const u8) []const []const u8 {
    var result: std.ArrayList([]const u8) = .empty;
    result.appendSlice(b.allocator, common) catch @panic("out of memory");
    for (specific) |flag| {
        std.debug.assert(std.mem.startsWith(u8, flag, "-m"));
        // Clang's driver deprecates -mevex512; its cc1 feature remains required.
        if (!std.mem.eql(u8, flag, "-mevex512")) result.append(b.allocator, flag) catch @panic("out of memory");
    }
    for (specific) |flag| {
        const disable = std.mem.startsWith(u8, flag, "-mno-");
        const feature = std.fmt.allocPrint(b.allocator, "{s}{s}", .{ if (disable) "-" else "+", flag[if (disable) 5 else 2..] }) catch @panic("out of memory");
        result.appendSlice(b.allocator, &.{ "-Xclang", "-target-feature", "-Xclang", feature }) catch @panic("out of memory");
    }
    return result.toOwnedSlice(b.allocator) catch @panic("out of memory");
}
