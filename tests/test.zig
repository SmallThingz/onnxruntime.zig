const std = @import("std");
const ort = @import("onnxruntime");
const allocator = std.testing.allocator;
const model_bytes = @embedFile("add.onnx");

test "owned typed tensor copies data and validates dimensions and element types" {
    var source = [_]f32{ 1, 2, 3 };
    var tensor = try ort.Tensor.fromSlice(f32, &.{3}, &source);
    defer tensor.deinit();
    source[0] = 99;
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, try tensor.data(f32));
    (try tensor.data(f32))[1] = 7;
    try std.testing.expectEqual(@as(f32, 7), (try tensor.data(f32))[1]);
    try std.testing.expectError(error.TypeMismatch, tensor.data(i32));
    const shape = try tensor.shape(allocator);
    defer allocator.free(shape);
    try std.testing.expectEqualSlices(i64, &.{3}, shape);
    try std.testing.expectError(error.InvalidShape, ort.Tensor.fromSlice(f32, &.{-1}, &source));
    try std.testing.expectError(error.InvalidShape, ort.Tensor.fromSlice(f32, &.{ std.math.maxInt(i64), 8 }, &source));
    try std.testing.expectError(error.ShapeMismatch, ort.Tensor.fromSlice(f32, &.{4}, &source));
    var scalar = try ort.Tensor.fromSlice(i64, &.{}, &.{42});
    defer scalar.deinit();
    try std.testing.expectEqualSlices(i64, &.{42}, try scalar.data(i64));
    var empty = try ort.Tensor.fromSlice(u8, &.{0}, &.{});
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try empty.data(u8)).len);
    var half = try ort.Tensor.fromSlice(f16, &.{2}, &.{ 1.5, -2.25 });
    defer half.deinit();
    try std.testing.expectEqualSlices(f16, &.{ 1.5, -2.25 }, try half.data(f16));
    var flags = try ort.Tensor.fromSlice(bool, &.{2}, &.{ true, false });
    defer flags.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false }, try flags.data(bool));
}

test "native ONNX Add inference exposes owned names and exact output values" {
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    var model = try environment.load(model_bytes, .{});
    defer model.deinit();
    var names = try model.inputNames(allocator);
    defer names.deinit();
    try std.testing.expectEqual(@as(usize, 2), names.items.len);
    try std.testing.expectEqualStrings("x", names.items[0]);
    try std.testing.expectEqualStrings("y", names.items[1]);
    var out_names = try model.outputNames(allocator);
    defer out_names.deinit();
    try std.testing.expectEqualStrings("sum", out_names.items[0]);
    var x = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 1, -2, 3.5 });
    defer x.deinit();
    var y = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 4, 2, 0.5 });
    defer y.deinit();
    const inputs = [_]ort.NamedTensor{ .{ .name = "x", .tensor = &x }, .{ .name = "y", .tensor = &y } };
    var outputs = try model.run(allocator, &inputs);
    defer outputs.deinit();
    try std.testing.expectEqual(@as(usize, 1), outputs.values.len);
    try std.testing.expectEqualSlices(f32, &.{ 5, 0, 4 }, try outputs.values[0].data(f32));
    var selected = try model.runSelected(allocator, &inputs, &.{"sum"});
    defer selected.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 0, 4 }, try selected.values[0].data(f32));
    try std.testing.expectError(error.InvalidArgument, model.runSelected(allocator, &inputs, &.{"missing"}));
    try std.testing.expectError(error.InvalidName, model.run(allocator, &.{ inputs[0], inputs[0] }));
    try std.testing.expectError(error.InvalidName, model.runSelected(allocator, &inputs, &.{"sum\x00suffix"}));
    try std.testing.expectError(error.InvalidArgument, model.run(allocator, inputs[0..1]));
}

test "model loading rejects invalid bytes and options without leaking native handles" {
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    try std.testing.expectError(error.InvalidProtobuf, environment.load("", .{}));
    try std.testing.expectError(error.InvalidProtobuf, environment.load("broken protobuf", .{}));
    try std.testing.expectError(error.InvalidArgument, environment.load(model_bytes, .{ .intra_op_threads = 0 }));
    try std.testing.expectError(error.InvalidName, ort.Environment.init(allocator, .{ .name = "name\x00tail" }));
    var from_file = try environment.open(allocator, "tests/add.onnx", .{});
    defer from_file.deinit();
    var names = try from_file.outputNames(allocator);
    defer names.deinit();
    try std.testing.expectEqualStrings("sum", names.items[0]);
}

test "explicit borrowed tensors retain caller ownership and observe backing updates" {
    const backing = try allocator.alloc(i32, 2);
    defer allocator.free(backing);
    @memcpy(backing, &[_]i32{ 12, 34 });
    var tensor = try ort.Tensor.borrowSlice(i32, &.{2}, backing);
    backing[0] = 99;
    try std.testing.expectEqualSlices(i32, &.{ 99, 34 }, try tensor.data(i32));
    tensor.deinit();
    backing[1] = 77;
    try std.testing.expectEqual(@as(i32, 77), backing[1]);
}

fn allocationWorkflow(test_allocator: std.mem.Allocator, model: *ort.Model, x: *const ort.Tensor, y: *const ort.Tensor) !void {
    var names = try model.inputNames(test_allocator);
    defer names.deinit();
    const dims = try x.shape(test_allocator);
    defer test_allocator.free(dims);
    var outputs = try model.run(test_allocator, &.{ .{ .name = "x", .tensor = x }, .{ .name = "y", .tensor = y } });
    defer outputs.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, try outputs.values[0].data(f32));
}

test "every Zig allocation failure releases partial names and run arrays" {
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    var model = try environment.load(model_bytes, .{});
    defer model.deinit();
    var x = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var y = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 4, 5, 6 });
    defer y.deinit();
    try std.testing.checkAllAllocationFailures(allocator, allocationWorkflow, .{ &model, &x, &y });
}
