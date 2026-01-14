const std = @import("std");

pub fn build(b: *std.Build) !void {
  const target = b.standardTargetOptions(.{});
  const optimize = b.standardOptimizeOption(.{});

  const mod = b.addModule("onnxruntime", .{
    .root_source_file = b.path("root.zig"),
    .target = target,
    .optimize = optimize,
  });

  addTestStep(b, mod, target, optimize) catch |err| {
    std.log.warn("Failed to add test step, error: {}", .{err});
  };
}

fn addTestStep(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !void {
  const test_step = b.step("test", "Run tests");
  const test_dir_name = "tests";
  var test_dir = try std.fs.cwd().openDir(test_dir_name, .{ .access_sub_paths = true, .iterate = true });
  defer test_dir.close();
  var it = test_dir.iterate();
  while (try it.next()) |entry| {
    const tests = b.addTest(.{
      .root_module = b.createModule(.{
        .root_source_file = b.path(b.pathJoin(&.{test_dir_name, entry.name})),
        .target = target,
        .optimize = optimize,
      }),
    });

    tests.root_module.addImport("onnxruntime", mod);
    tests.linkLibC();
    tests.addIncludePath(.{ .cwd_relative = "/usr/include/onnxruntime" });
    tests.linkSystemLibrary("onnxruntime");
    tests.addRPath(.{ .cwd_relative = "/usr/lib" });
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);
  }
}

