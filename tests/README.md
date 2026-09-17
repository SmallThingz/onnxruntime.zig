Fixture `add.onnx` is a minimal ONNX IR 10/opset 13 model generated directly from the protobuf schema: two float32 inputs `x` and `y`, each shape `[3]`; one Add node producing `sum`, shape `[3]`. No weights or external files. Tests compare actual native inference against exact expected arithmetic.

`matmul_relu.onnx` is an ONNX IR 10/opset 13 graph generated once from the protobuf
wire schema: float32 `a[32,64]` and `b[64,48]`, MatMul, then Relu, yielding
`result[32,48]`. It contains no weights or external files. The regression computes
all 1,536 outputs with an independent Zig loop using small integral float inputs,
then checks exact equality with optimizations disabled and enabled. The nontrivial
matrix size exercises native matrix-multiplication dispatch; the test does not
claim a specific ISA branch without instrumentation. No generator runs at build
time and no Python package is required.
