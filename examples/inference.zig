const std = @import("std");
const ort = @import("onnxruntime");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, "tests/add.onnx", allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    var model = try environment.load(bytes, .{});
    defer model.deinit();
    var x = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var y = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 4, 5, 6 });
    defer y.deinit();
    var outputs = try model.run(allocator, &.{ .{ .name = "x", .tensor = &x }, .{ .name = "y", .tensor = &y } });
    defer outputs.deinit();
    const result = try outputs.values[0].data(f32);
    if (!std.mem.eql(f32, result, &.{ 5, 7, 9 })) return error.UnexpectedInference;
    std.debug.print("ONNX Runtime {s}: {any}\n", .{ ort.version(), result });
}
