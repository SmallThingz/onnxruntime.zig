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

test "native inference rejects wrong input element types dimensions and rank" {
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    var model = try environment.load(model_bytes, .{});
    defer model.deinit();
    var y = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 4, 5, 6 });
    defer y.deinit();
    var wrong_type = try ort.Tensor.fromSlice(i32, &.{3}, &.{ 1, 2, 3 });
    defer wrong_type.deinit();
    var wrong_length = try ort.Tensor.fromSlice(f32, &.{2}, &.{ 1, 2 });
    defer wrong_length.deinit();
    var wrong_rank = try ort.Tensor.fromSlice(f32, &.{ 1, 3 }, &.{ 1, 2, 3 });
    defer wrong_rank.deinit();
    var empty = try ort.Tensor.fromSlice(f32, &.{0}, &.{});
    defer empty.deinit();
    for ([_]*const ort.Tensor{ &wrong_type, &wrong_length, &wrong_rank, &empty }) |invalid| {
        try std.testing.expectError(error.InvalidArgument, model.run(allocator, &.{
            .{ .name = "x", .tensor = invalid },
            .{ .name = "y", .tensor = &y },
        }));
    }
    // A failed run must leave the session usable.
    var x = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var outputs = try model.run(allocator, &.{
        .{ .name = "x", .tensor = &x },
        .{ .name = "y", .tensor = &y },
    });
    defer outputs.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, try outputs.values[0].data(f32));
    const shape = try outputs.values[0].shape(allocator);
    defer allocator.free(shape);
    try std.testing.expectEqualSlices(i64, &.{3}, shape);
}

test "run outputs and copied names outlive model and input tensors" {
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    const result = blk: {
        var model = try environment.load(model_bytes, .{});
        defer model.deinit();
        var x = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 8, -1, 0.5 });
        defer x.deinit();
        var y = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 2, 1, 1.5 });
        defer y.deinit();
        var names = try model.outputNames(allocator);
        errdefer names.deinit();
        const outputs = try model.run(allocator, &.{
            .{ .name = "x", .tensor = &x },
            .{ .name = "y", .tensor = &y },
        });
        break :blk .{ .names = names, .outputs = outputs };
    };
    var names = result.names;
    defer names.deinit();
    var outputs = result.outputs;
    defer outputs.deinit();
    try std.testing.expectEqualStrings("sum", names.items[0]);
    try std.testing.expectEqualSlices(f32, &.{ 10, 0, 2 }, try outputs.values[0].data(f32));
    const shape = try outputs.values[0].shape(allocator);
    defer allocator.free(shape);
    try std.testing.expectEqualSlices(i64, &.{3}, shape);
    (try outputs.values[0].data(f32))[0] = 12;
    try std.testing.expectEqual(@as(f32, 12), (try outputs.values[0].data(f32))[0]);
}

test "zero-sized and borrowed scalar tensors preserve exact shape contracts" {
    var empty = try ort.Tensor.fromSlice(f32, &.{ 0, std.math.maxInt(i64), 8 }, &.{});
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try empty.data(f32)).len);
    const dimensions = try empty.shape(allocator);
    defer allocator.free(dimensions);
    try std.testing.expectEqualSlices(i64, &.{ 0, std.math.maxInt(i64), 8 }, dimensions);
    var scalar_data = [_]f64{4.25};
    var scalar = try ort.Tensor.borrowSlice(f64, &.{}, &scalar_data);
    defer scalar.deinit();
    const scalar_shape = try scalar.shape(allocator);
    defer allocator.free(scalar_shape);
    try std.testing.expectEqual(@as(usize, 0), scalar_shape.len);
    (try scalar.data(f64))[0] = -3.5;
    try std.testing.expectEqual(@as(f64, -3.5), scalar_data[0]);
    try std.testing.expectError(error.ShapeMismatch, ort.Tensor.borrowSlice(f64, &.{0}, &scalar_data));
    try std.testing.expectError(error.InvalidShape, ort.Tensor.borrowSlice(f64, &.{-1}, &scalar_data));
    try std.testing.expectError(error.InvalidShape, ort.Tensor.borrowSlice(f64, &.{ std.math.maxInt(i64), 8 }, &scalar_data));
    try std.testing.expectError(error.InvalidShape, ort.Tensor.fromSlice(u8, &.{ 0, -1 }, &.{}));
}

test "parallel execution options run actual inference" {
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    var model = try environment.load(model_bytes, .{
        .execution_mode = .parallel,
        .inter_op_threads = 2,
        .intra_op_threads = 1,
    });
    defer model.deinit();
    var x = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 2, -4, 0.25 });
    defer x.deinit();
    var y = try ort.Tensor.fromSlice(f32, &.{3}, &.{ 3, 4, 0.75 });
    defer y.deinit();
    var outputs = try model.run(allocator, &.{
        .{ .name = "x", .tensor = &x },
        .{ .name = "y", .tensor = &y },
    });
    defer outputs.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 0, 1 }, try outputs.values[0].data(f32));
}

test "native MatMul and Relu match full matrix reference through MLAS kernels" {
    const rows = 32;
    const inner = 64;
    const columns = 48;
    var a: [rows * inner]f32 = undefined;
    var b: [inner * columns]f32 = undefined;
    for (&a, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast((i * 7 + i / inner) % 13)) - 6);
    for (&b, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast((i * 3 + i / columns) % 11)) - 5);
    var expected: [rows * columns]f32 = undefined;
    var zeros: usize = 0;
    for (0..rows) |row| {
        for (0..columns) |column| {
            var sum: f32 = 0;
            for (0..inner) |k| sum += a[row * inner + k] * b[k * columns + column];
            expected[row * columns + column] = @max(sum, 0);
            if (sum <= 0) zeros += 1;
        }
    }
    try std.testing.expect(zeros > 0 and zeros < expected.len);
    var environment = try ort.Environment.init(allocator, .{});
    defer environment.deinit();
    var input_a = try ort.Tensor.fromSlice(f32, &.{ rows, inner }, &a);
    defer input_a.deinit();
    var input_b = try ort.Tensor.fromSlice(f32, &.{ inner, columns }, &b);
    defer input_b.deinit();
    for ([_]ort.Optimization{ .disabled, .all }) |optimization| {
        var model = try environment.load(@embedFile("matmul_relu.onnx"), .{ .optimization = optimization });
        defer model.deinit();
        var output = try model.run(allocator, &.{
            .{ .name = "a", .tensor = &input_a },
            .{ .name = "b", .tensor = &input_b },
        });
        defer output.deinit();
        try std.testing.expectEqual(@as(usize, 1), output.values.len);
        const shape = try output.values[0].shape(allocator);
        defer allocator.free(shape);
        try std.testing.expectEqualSlices(i64, &.{ rows, columns }, shape);
        // Integer-valued inputs and sums are exactly representable in f32;
        // every output is checked, including positive and clamped values.
        try std.testing.expectEqualSlices(f32, &expected, try output.values[0].data(f32));
    }
}
