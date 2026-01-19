const std = @import("std");
const onnx = @import("onnxruntime");
const Api = onnx.Api;

/// If we use std.testing.refAllDeclsRecursive, we get a compile error because c has untranslatable code, hence we use this
/// Even this touches the translated parts of the c code that we touch, but atleast not it doesn't crash
fn refAllDeclsRecursiveExcerptC(comptime T: type) void {
  if (!@import("builtin").is_test) return;
  inline for (comptime std.meta.declarations(T)) |decl| {
    _ = &@field(T, decl.name);
    if (@TypeOf(@field(T, decl.name)) == type) {
      if (decl.name.len == 1 and decl.name[0] == 'c') continue;
      switch (@typeInfo(@field(T, decl.name))) {
        .@"struct", .@"enum", .@"union", .@"opaque" => refAllDeclsRecursiveExcerptC(@field(T, decl.name)),
        else => {},
      }
    }
  }
}

test {
  refAllDeclsRecursiveExcerptC(onnx);
}

