const std = @import("std");
const builtin = @import("builtin");
const onnx = @import("onnxruntime");

pub fn main() !void {
  const test_fns: []const std.builtin.TestFn = builtin.test_functions;
  
  std.debug.print("--- Initializing ONNX Runtime ---\n\n", .{});
  try onnx.Api.init(.{
    .log_level = .warning,
    .log_id = "zig-ort-tests",
    .editor = true,
    .compiler = true,
    .ep = true,
    .training = true,
  }, .{});
  
  var passed: usize = 0;
  var failed: usize = 0;
  var skipped: usize = 0;

  for (test_fns) |test_fn| {
    std.debug.print("{s} => ", .{test_fn.name});
    
    if (test_fn.func()) |_| {
      std.debug.print("OK\n", .{});
      passed += 1;
    } else |err| {
      if (err == error.SkipZigTest) {
        std.debug.print("SKIPPED\n", .{});
        skipped += 1;
      } else {
        std.debug.print("FAILED ({s})\n", .{@errorName(err)});
        failed += 1;
      }
    }
  }

  std.debug.print("\n--- Deinitializing ONNX Runtime ---\n", .{});
  onnx.Api.deinit();

  std.debug.print("\nTest Summary: {} passed, {} failed, {} skipped\n", .{ passed, failed, skipped });
  if (failed > 0) std.process.exit(1);
}
