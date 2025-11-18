#include "winograd_conv2d.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int FLT_H = 3;
constexpr int FLT_W = 3;
constexpr int TILE_IN_H = 6;
constexpr int TILE_IN_W = 6;
constexpr int TILE_OUT_H = 4;
constexpr int TILE_OUT_W = 4;
constexpr int TRANSFORM_SIZE = TILE_IN_H * TILE_IN_W;

constexpr int BLK_M = 32;
constexpr int BLK_N = 14;
constexpr int BLK_K = 9;
constexpr int WARP_SIZE = 64;
constexpr int TCU_SIZE = 16;

using fp32x4 = float __attribute__((ext_vector_type(4)));

struct TileIndex {
    int b;
    int th;
    int tw;
};

// 计算给定 tileNo 对应的批次和 tile 在高/宽方向的索引
__device__ __forceinline__ TileIndex getTileIndex(int tileNo, int tiles_per_img, int tiles_w) {
    TileIndex ti;
    ti.b = tileNo / tiles_per_img;
    tileNo = tileNo % tiles_per_img;
    ti.th = tileNo / tiles_w;
    ti.tw = tileNo % tiles_w;
    return ti;
}

// 检查参数是否满足使用 Winograd F(4x4,3x3) 的条件
bool supports_winograd_f4x3(int N,
                            int C_in,
                            int H,
                            int W,
                            int C_out,
                            int K_h,
                            int K_w,
                            int out_H,
                            int out_W) {
    if (N <= 0 || C_in <= 0 || C_out <= 0) {
        return false;
    }
    if (K_h != FLT_H || K_w != FLT_W) {
        return false;
    }
    if (out_H != (H - 2) || out_W != (W - 2)) {
        return false;
    }
    if (out_H <= 0 || out_W <= 0) {
        return false;
    }
    if ((out_H % TILE_OUT_H) != 0 || (out_W % TILE_OUT_W) != 0) {
        return false;
    }
    return true;
}

}  // namespace

template <int BLK_M_, int BLK_N_, int BLK_K_>
__global__ void __launch_bounds__(BLK_M_ * BLK_K_ * 2, 1)
winograd_4x3_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int out_H, int out_W,
    int tiles_h, int tiles_w, int tiles_per_img, int total_tiles)
{
    static_assert(BLK_M_ == BLK_M && BLK_N_ == BLK_N && BLK_K_ == BLK_K, "Kernel tuning assumes 32x32x16 tiles");

    __shared__ union {
        struct {
            float V[TILE_IN_H][TILE_IN_W][BLK_K_][BLK_M_];
            float Ut[TILE_IN_H][TILE_IN_W][BLK_N_][BLK_K_];
        } s;
        struct {
            float Y[TILE_IN_H][TILE_IN_W][BLK_N_][BLK_M_];
        } y;
    } lds;

    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tid = thx + thy * blockDim.x;
    const int tile_blk = BLK_M_ * blockIdx.x;
    const int oc_blk = BLK_N_ * blockIdx.y;

    fp32x4 C_reg_acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            C_reg_acc[i][j] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    const int num_k_tiles = (C_in + BLK_K_ - 1) / BLK_K_;
    if (num_k_tiles == 0) return;

    auto load_tiles = [&](int ic_base) {
        if (thy == 1) {
            const int local_ic_idx = thx % BLK_K_;
            const int local_oc_idx = thx / BLK_K_;
            const int oc = oc_blk + local_oc_idx;
            const int ic = ic_base + local_ic_idx;

            float filter_patch[FLT_H][FLT_W] = {};
            if (oc < C_out && ic < C_in) {
                const size_t base = (static_cast<size_t>(oc) * C_in + ic) * (FLT_H * FLT_W);
                const float* base_ptr = weight + base;
                #pragma unroll
                for (int h = 0; h < FLT_H; ++h) {
                    #pragma unroll
                    for (int w = 0; w < FLT_W; ++w) {
                        filter_patch[h][w] = base_ptr[h * FLT_W + w];
                    }
                }
            }

            float tmp[TILE_IN_H][FLT_W];
            #pragma unroll
            for (int w = 0; w < FLT_W; ++w) {
                float z0 = 1.0f/4.0f * filter_patch[0][w];
                float z1 = -1.0f/6.0f * filter_patch[0][w];
                float z2 = -1.0f/6.0f * filter_patch[0][w];
                float z3 = 1.0f/24.0f * filter_patch[0][w];
                float z4 = 1.0f/24.0f * filter_patch[0][w];
                float z5 = 0.0f;

                z1 += -1.0f/6.0f * filter_patch[1][w];
                z2 += 1.0f/6.0f * filter_patch[1][w];
                z3 += 1.0f/12.0f * filter_patch[1][w];
                z4 += -1.0f/12.0f * filter_patch[1][w];

                z1 += -1.0f/6.0f * filter_patch[2][w];
                z2 += -1.0f/6.0f * filter_patch[2][w];
                z3 += 1.0f/6.0f * filter_patch[2][w];
                z4 += 1.0f/6.0f * filter_patch[2][w];
                z5 = filter_patch[2][w];

                tmp[0][w] = z0;
                tmp[1][w] = z1;
                tmp[2][w] = z2;
                tmp[3][w] = z3;
                tmp[4][w] = z4;
                tmp[5][w] = z5;
            }

            #pragma unroll
            for (int h = 0; h < TILE_IN_H; ++h) {
                float z0 = 1.0f/4.0f * tmp[h][0];
                float z1 = -1.0f/6.0f * tmp[h][0];
                float z2 = -1.0f/6.0f * tmp[h][0];
                float z3 = 1.0f/24.0f * tmp[h][0];
                float z4 = 1.0f/24.0f * tmp[h][0];
                float z5 = 0.0f;

                z1 += -1.0f/6.0f * tmp[h][1];
                z2 += 1.0f/6.0f * tmp[h][1];
                z3 += 1.0f/12.0f * tmp[h][1];
                z4 += -1.0f/12.0f * tmp[h][1];

                z1 += -1.0f/6.0f * tmp[h][2];
                z2 += -1.0f/6.0f * tmp[h][2];
                z3 += 1.0f/6.0f * tmp[h][2];
                z4 += 1.0f/6.0f * tmp[h][2];
                z5 = tmp[h][2];

                lds.s.Ut[h][0][local_oc_idx][local_ic_idx] = z0;
                lds.s.Ut[h][1][local_oc_idx][local_ic_idx] = z1;
                lds.s.Ut[h][2][local_oc_idx][local_ic_idx] = z2;
                lds.s.Ut[h][3][local_oc_idx][local_ic_idx] = z3;
                lds.s.Ut[h][4][local_oc_idx][local_ic_idx] = z4;
                lds.s.Ut[h][5][local_oc_idx][local_ic_idx] = z5;
            }
        }

        if (thy == 0) {
            const int local_tile_idx = thx % BLK_M_;
            const int local_ic_idx = thx / BLK_M_;
            const int tile_idx = tile_blk + local_tile_idx;
            const int ic = ic_base + local_ic_idx;

            float img_tile[TILE_IN_H][TILE_IN_W] = {};
            if (tile_idx < total_tiles && ic < C_in) {
                TileIndex ti = getTileIndex(tile_idx, tiles_per_img, tiles_w);
                const int b = ti.b;
                const int th = ti.th;
                const int tw = ti.tw;
                const int in_w_base = tw * TILE_OUT_W;
                const size_t image_plane_base = (static_cast<size_t>(b) * C_in + ic) * H * W;

                #pragma unroll
                for (int h = 0; h < TILE_IN_H; ++h) {
                    const int in_h = th * TILE_OUT_H + h;
                    if (in_h < H) {
                        const size_t row_offset = image_plane_base + static_cast<size_t>(in_h) * W;
                        #pragma unroll
                        for (int w = 0; w < TILE_IN_W; ++w) {
                            const int in_w = in_w_base + w;
                            if (in_w < W) {
                                const size_t offset = row_offset + in_w;
                                img_tile[h][w] = input[offset];
                            }
                        }
                    }
                }
            }

            float tmp[TILE_IN_H][TILE_IN_W];
            #pragma unroll
            for (int w = 0; w < TILE_IN_W; ++w) {
                float z0 = 4.0f * img_tile[0][w];
                float z1 = -4.0f * img_tile[1][w];
                float z2 =  4.0f * img_tile[1][w];
                float z3 = -2.0f * img_tile[1][w];
                float z4 =  2.0f * img_tile[1][w];
                float z5 =  4.0f * img_tile[1][w];

                z0 += -5.0f * img_tile[2][w];
                z1 += -4.0f * img_tile[2][w];
                z2 += -4.0f * img_tile[2][w];
                z3 += -img_tile[2][w];
                z4 += -img_tile[2][w];

                z1 +=  img_tile[3][w];
                z2 -=  img_tile[3][w];
                z3 +=  2.0f * img_tile[3][w];
                z4 -=  2.0f * img_tile[3][w];
                z5 += -5.0f * img_tile[3][w];

                z0 += img_tile[4][w];
                z1 += img_tile[4][w];
                z2 += img_tile[4][w];
                z3 += img_tile[4][w];
                z4 += img_tile[4][w];

                z5 += img_tile[5][w];

                tmp[0][w] = z0;
                tmp[1][w] = z1;
                tmp[2][w] = z2;
                tmp[3][w] = z3;
                tmp[4][w] = z4;
                tmp[5][w] = z5;
            }

            #pragma unroll
            for (int h = 0; h < TILE_IN_H; ++h) {
                float z0 = 4.0f * tmp[h][0];
                float z1 = -4.0f * tmp[h][1];
                float z2 =  4.0f * tmp[h][1];
                float z3 = -2.0f * tmp[h][1];
                float z4 =  2.0f * tmp[h][1];
                float z5 =  4.0f * tmp[h][1];

                z0 += -5.0f * tmp[h][2];
                z1 += -4.0f * tmp[h][2];
                z2 += -4.0f * tmp[h][2];
                z3 += -tmp[h][2];
                z4 += -tmp[h][2];

                z1 +=  tmp[h][3];
                z2 -=  tmp[h][3];
                z3 +=  2.0f * tmp[h][3];
                z4 -=  2.0f * tmp[h][3];
                z5 += -5.0f * tmp[h][3];

                z0 += tmp[h][4];
                z1 += tmp[h][4];
                z2 += tmp[h][4];
                z3 += tmp[h][4];
                z4 += tmp[h][4];

                z5 += tmp[h][5];

                lds.s.V[h][0][local_ic_idx][local_tile_idx] = z0;
                lds.s.V[h][1][local_ic_idx][local_tile_idx] = z1;
                lds.s.V[h][2][local_ic_idx][local_tile_idx] = z2;
                lds.s.V[h][3][local_ic_idx][local_tile_idx] = z3;
                lds.s.V[h][4][local_ic_idx][local_tile_idx] = z4;
                lds.s.V[h][5][local_ic_idx][local_tile_idx] = z5;
            }
        }
    };

    load_tiles(0);
    __syncthreads();

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int elem_idx = tid / WARP_SIZE;
        if (elem_idx < TILE_IN_H * TILE_IN_W) {
            const int h = elem_idx / TILE_IN_W;
            const int w = elem_idx % TILE_IN_W;
            const int write_dim_m = (tid % WARP_SIZE) % BLK_K_;
            const int write_dim_n = (tid % WARP_SIZE) / BLK_K_;

            #pragma unroll
            for (int k = 0; k < BLK_K_; ++k) {
                const float v0 = lds.s.V[h][w][k][write_dim_m +  0];
                const float v1 = lds.s.V[h][w][k][write_dim_m + 16];

                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int col = write_dim_n + t * 4;
                    C_reg_acc[0][0][t] += v0 * lds.s.Ut[h][w][col +  0][k];
                    C_reg_acc[0][1][t] += v0 * lds.s.Ut[h][w][col + 16][k];
                    C_reg_acc[1][0][t] += v1 * lds.s.Ut[h][w][col +  0][k];
                    C_reg_acc[1][1][t] += v1 * lds.s.Ut[h][w][col + 16][k];
                }
            }
        }

        __syncthreads();
        if (k_tile + 1 < num_k_tiles) {
            const int next_ic_base = (k_tile + 1) * BLK_K_;
            load_tiles(next_ic_base);
            __syncthreads();
        }
    }

    const int elem_idx_out = tid / WARP_SIZE;
    if (elem_idx_out < TILE_IN_H * TILE_IN_W) {
        const int h = elem_idx_out / TILE_IN_W;
        const int w = elem_idx_out % TILE_IN_W;
        const size_t write_local_tile = (tid % WARP_SIZE) % TCU_SIZE;
        const size_t write_local_oc = (tid % WARP_SIZE) / TCU_SIZE;

        lds.y.Y[h][w][write_local_oc +  0 +  0][write_local_tile +  0] = C_reg_acc[0][0].x;
        lds.y.Y[h][w][write_local_oc +  4 +  0][write_local_tile +  0] = C_reg_acc[0][0].y;
        lds.y.Y[h][w][write_local_oc +  8 +  0][write_local_tile +  0] = C_reg_acc[0][0].z;
        lds.y.Y[h][w][write_local_oc + 12 +  0][write_local_tile +  0] = C_reg_acc[0][0].w;
    }

    __syncthreads();

    const int oc_group = thx / BLK_M_;
    const int local_tile_idx = thx % BLK_M_;
    const int oc_stride = BLK_N_ / blockDim.y;
    const int local_oc_idx = oc_group + thy * oc_stride;

    if (local_oc_idx < BLK_N_) {
        const int oc = oc_blk + local_oc_idx;
        const int tile = tile_blk + local_tile_idx;
        const bool oc_valid = (oc < C_out);
        const bool tile_valid = (tile < total_tiles);

        TileIndex ti{};
        if (tile_valid) ti = getTileIndex(tile, tiles_per_img, tiles_w);

        const int out_h_base = tile_valid ? ti.th * TILE_OUT_H : 0;
        const int out_w_base = tile_valid ? ti.tw * TILE_OUT_W : 0;
        const size_t out_plane_base = (oc_valid && tile_valid)
            ? (static_cast<size_t>(ti.b) * C_out + oc) * out_H * out_W
            : 0;
        const float bias_val = (bias != nullptr && oc_valid) ? bias[oc] : 0.0f;

        for (int h = 0; h < TILE_OUT_H; ++h) {
            const int oh = out_h_base + h;
            if (oh >= out_H) continue;
            for (int w = 0; w < TILE_OUT_W; ++w) {
                const int ow = out_w_base + w;
                if (ow >= out_W) continue;
                float accum_out = 0.0f;
                accum_out = lds.y.Y[0][0][local_oc_idx][local_tile_idx];
                if (oc_valid && tile_valid) {
                    output[out_plane_base + (size_t)oh * out_W + ow] = accum_out + bias_val;
                }
            }
        }
    }
}

// 尝试用 Winograd F4x3 路径运行卷积，成功返回 true，否则返回 false
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
    hipStream_t stream) {
    if (!supports_winograd_f4x3(N, C_in, H, W, C_out, K_h, K_w, out_H, out_W)) {
        return false;
    }

    const int tiles_h = out_H / TILE_OUT_H;
    const int tiles_w = out_W / TILE_OUT_W;
    const int tiles_per_img = tiles_h * tiles_w;
    const int total_tiles = N * tiles_per_img;

    dim3 grid((total_tiles + BLK_M - 1) / BLK_M, (C_out + BLK_N - 1) / BLK_N);
    dim3 block(BLK_M * BLK_K, 2);

    hipLaunchKernelGGL(
        (winograd_4x3_fused_kernel<BLK_M, BLK_N, BLK_K>),
        grid,
        block,
        0,
        stream,
        input, weight, bias, output,
        N, C_in, H, W, C_out, out_H, out_W,
        tiles_h, tiles_w, tiles_per_img, total_tiles);

    hipError_t err = hipGetLastError();
    return err == hipSuccess;
}
