const std = @import("std");

pub fn build(b: *std.Build) !void {
  const target = b.standardTargetOptions(.{});
  const optimize = b.standardOptimizeOption(.{});

  const mod = b.addModule("onnxruntime", .{
    .root_source_file = b.path("root.zig"),
    .link_libc = true,
    .target = target,
    .optimize = optimize,
  });

  const include_paths = b.option([]const []const u8, "include_paths", "the paths to include for the onnxruntime module")
    orelse &[_][]const u8{"/usr/include/", "/usr/include/onnxruntime/"};
  for (include_paths) |path| mod.addIncludePath(.{ .cwd_relative = path });

  const r_paths = b.option([]const []const u8, "r_paths", "the paths to add to the rpath for the onnxruntime module")
    orelse &[_][]const u8{"/usr/lib/"};
  for (r_paths) |path| mod.addRPath(.{ .cwd_relative = path });

  mod.linkSystemLibrary("onnxruntime", .{});

  addTestStep(b, mod, target, optimize) catch {}; // |err| std.log.warn("Failed to add test step, error: {}", .{err});
}

fn exists(path: []const u8) bool {
  std.fs.cwd().access(path, .{}) catch return false;
  return true;
}

fn addTestStep(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !void {
  const test_step = b.step("test", "Run tests");
  const test_file_name = "test.zig";
  const test_runner_name = "test_runner.zig";

  // we don't care about time-of-check time-of-use race conditions as this is a simple test runner
  if (!exists(test_file_name)) return error.MissingTestFile;
  if (!exists(test_runner_name)) return error.MissingTestRunner;

  const tests = b.addTest(.{
    .root_module = b.createModule(.{
      .root_source_file = b.path(test_file_name),
      .target = target,
      .optimize = optimize,
    }),
    .test_runner = .{
      .path = b.path(test_runner_name),
      .mode = .simple,
    }
  });

  tests.root_module.addImport("onnxruntime", mod);
  const run_tests = b.addRunArtifact(tests);
  test_step.dependOn(&run_tests.step);
}

