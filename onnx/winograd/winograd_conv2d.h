#pragma once

#include <hip/hip_runtime.h>

bool try_run_winograd_conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int K_h,
    int K_w,
    int out_H,
    int out_W,
    hipStream_t stream);
