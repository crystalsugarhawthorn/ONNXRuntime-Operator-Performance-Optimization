#include "winograd_conv2d.h"

#include <cstdio>

bool try_run_winograd_f2x3(
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

bool try_run_winograd_f4x3(
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
    hipStream_t stream) {
    if (try_run_winograd_f4x3(input, weight, bias, output,
                              N, C_in, H, W, C_out, K_h, K_w,
                              out_H, out_W, stream)) {
        return true;
    }
    if (try_run_winograd_f2x3(input, weight, bias, output,
                              N, C_in, H, W, C_out, K_h, K_w,
                              out_H, out_W, stream)) {
        return true;
    }
    return false;
}
