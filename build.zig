const std = @import("std");

pub fn build(b: *std.Build) void {
  const target = b.standardTargetOptions(.{});
  const optimize = b.standardOptimizeOption(.{});

  const mod = b.addModule("onnxruntime", .{
    .root_source_file = b.path("root.zig"),
    .target = target,
    .optimize = optimize,
  });

  const tests = b.addTest(.{
    .root_module = mod,
  });
  tests.linkLibC();
  tests.addIncludePath(.{ .cwd_relative = "/usr/include/onnxruntime" });
  tests.linkSystemLibrary("onnxruntime");
  tests.addRPath(.{ .cwd_relative = "/usr/lib" });

  const run_mod_tests = b.addRunArtifact(tests);
  const test_step = b.step("test", "Run tests");
  test_step.dependOn(&run_mod_tests.step);
}

