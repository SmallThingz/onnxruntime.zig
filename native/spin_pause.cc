// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Adapted from ONNX Runtime 1.23.2 core/common/spin_pause.cc: use Clang's
// public two-argument intrinsic rather than its version-dependent builtin.
#include "core/common/spin_pause.h"
#include "core/common/cpuid_info.h"
#include <immintrin.h>

namespace onnxruntime::concurrency {
void SpinPause() {
  static const bool has_tpause = CPUIDInfo::GetCPUIDInfo().HasTPAUSE();
  if (has_tpause) {
    _tpause(0, __rdtsc() + 1000);
  } else {
    _mm_pause();
  }
}
}  // namespace onnxruntime::concurrency
