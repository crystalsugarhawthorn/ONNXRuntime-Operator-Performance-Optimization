#include "winograd_conv2d.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int FLT_H = 3;
constexpr int FLT_W = 3;
constexpr int TILE_IN_H = 4;
constexpr int TILE_IN_W = 4;
constexpr int TILE_OUT_H = 2;
constexpr int TILE_OUT_W = 2;

constexpr int BLK_M = 32;  
constexpr int BLK_N = 32;
constexpr int BLK_K = 16;
constexpr int WARP_SIZE = 64;
constexpr int NUM_LDS_BUFFERS = 1;
constexpr int TCU_SIZE = 16;

using fp32x4 = float __attribute__((ext_vector_type(4)));

struct TileIndex {
    int b;   
    int th; 
    int tw; 
};

// 从线性 tile 索引计算所属的 batch 以及 tile 在高/宽方向的坐标
__device__ __forceinline__ TileIndex getTileIndex(int tileNo, int tiles_per_img, int tiles_w) {
    TileIndex ti;
    ti.b = tileNo / tiles_per_img;
    tileNo = tileNo % tiles_per_img;
    ti.th = tileNo / tiles_w;
    ti.tw = tileNo % tiles_w;
    return ti;
}


// Winograd F(2x2,3x3) 融合内核：负责输入变换、分块 GEMM 与输出反变换
template <int BLK_M_, int BLK_N_, int BLK_K_>
__global__ void __launch_bounds__(BLK_M_ * BLK_K_ * 2, 1)
winograd_2x3_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int out_H, int out_W,
    int tiles_h, int tiles_w, int tiles_per_img, int total_tiles)
{
    static_assert(BLK_M_ == 32 && BLK_N_ == 32 && BLK_K_ == 16, "Kernel tuning assumes 32x32x16 tiles");

    __shared__ union {
        struct {
            float V[NUM_LDS_BUFFERS][TILE_IN_H][TILE_IN_W][BLK_K_][BLK_M_];
            float Ut[NUM_LDS_BUFFERS][TILE_IN_H][TILE_IN_W][BLK_N_][BLK_K_];
        };
        struct {
            float Y[TILE_IN_H][TILE_IN_W][BLK_N_][BLK_M_];
        };
    } lds;

    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tid = thx + thy * blockDim.x;
    const int tile_blk = BLK_M_ * blockIdx.x;
    const int oc_blk = BLK_N_ * blockIdx.y;

    fp32x4 C_reg_acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            C_reg_acc[i][j] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    const int num_k_tiles = (C_in + BLK_K_ - 1) / BLK_K_;
    if (num_k_tiles == 0) {
        return;
    }

    auto load_tiles = [&](int ic_base, int buffer_idx) {
        float z0, z1, z2, z3, z6;

        if (thy == 1) {
            const int local_ic_idx = thx % BLK_K_;
            const int local_oc_idx = thx / BLK_K_;
            const int oc = oc_blk + local_oc_idx;
            const int ic = ic_base + local_ic_idx;

            float filter_patch[FLT_H][FLT_W] = {};
            if (oc < C_out && ic < C_in) {
                const size_t base = (static_cast<size_t>(oc) * C_in + ic) * (FLT_H * FLT_W);
                const float* base_ptr = weight + base;
                const uintptr_t addr0 = reinterpret_cast<uintptr_t>(base_ptr);
                const uintptr_t addr1 = reinterpret_cast<uintptr_t>(base_ptr + 4);

                if (((addr0 | addr1) & 0xF) == 0) {
                    const float4 vec0 = reinterpret_cast<const float4*>(base_ptr)[0];
                    const float4 vec1 = reinterpret_cast<const float4*>(base_ptr + 4)[0];
                    const float tail = base_ptr[8];

                    filter_patch[0][0] = vec0.x;
                    filter_patch[0][1] = vec0.y;
                    filter_patch[0][2] = vec0.z;

                    filter_patch[1][0] = vec0.w;
                    filter_patch[1][1] = vec1.x;
                    filter_patch[1][2] = vec1.y;

                    filter_patch[2][0] = vec1.z;
                    filter_patch[2][1] = vec1.w;
                    filter_patch[2][2] = tail;
                } else {
                    #pragma unroll
                    for (int h = 0; h < FLT_H; ++h) {
                        #pragma unroll
                        for (int w = 0; w < FLT_W; ++w) {
                            filter_patch[h][w] = base_ptr[h * FLT_W + w];
                        }
                    }
                }
            }

            float tmp[TILE_IN_H][FLT_W];

            #pragma unroll
            for (int w = 0; w < FLT_W; ++w) {
                z6 = filter_patch[0][w];
                z0 = z6;
                z1 = 0.5f * z6;
                z2 = 0.5f * z6;

                z6 = filter_patch[1][w];
                z1 += 0.5f * z6;
                z2 -= 0.5f * z6;

                z6 = filter_patch[2][w];
                z1 += 0.5f * z6;
                z2 += 0.5f * z6;
                z3 = z6;

                tmp[0][w] = z0;
                tmp[1][w] = z1;
                tmp[2][w] = z2;
                tmp[3][w] = z3;
            }

            #pragma unroll
            for (int h = 0; h < TILE_IN_H; ++h) {
                z6 = tmp[h][0];
                z0 = z6;
                z1 = 0.5f * z6;
                z2 = 0.5f * z6;

                z6 = tmp[h][1];
                z1 += 0.5f * z6;
                z2 -= 0.5f * z6;

                z6 = tmp[h][2];
                z1 += 0.5f * z6;
                z2 += 0.5f * z6;
                z3 = z6;

                lds.Ut[buffer_idx][h][0][local_oc_idx][local_ic_idx] = z0;
                lds.Ut[buffer_idx][h][1][local_oc_idx][local_ic_idx] = z1;
                lds.Ut[buffer_idx][h][2][local_oc_idx][local_ic_idx] = z2;
                lds.Ut[buffer_idx][h][3][local_oc_idx][local_ic_idx] = z3;
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
                const size_t image_plane_base =
                    (static_cast<size_t>(b) * C_in + ic) * H * W;

                #pragma unroll
                for (int h = 0; h < TILE_IN_H; ++h) {
                    const int in_h = th * TILE_OUT_H + h;
                    if (in_h < H) {
                        const size_t row_offset = image_plane_base + static_cast<size_t>(in_h) * W;
                        if (in_w_base + TILE_IN_W <= W) {
                            const float* row_ptr = input + row_offset + in_w_base;
                            if ((reinterpret_cast<uintptr_t>(row_ptr) & 0xF) == 0) {
                                const float4 vec = reinterpret_cast<const float4*>(row_ptr)[0];
                                img_tile[h][0] = vec.x;
                                img_tile[h][1] = vec.y;
                                img_tile[h][2] = vec.z;
                                img_tile[h][3] = vec.w;
                                continue;
                            }
                        }

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

            #pragma unroll
            for (int w = 0; w < TILE_IN_W; ++w) {
                z6 = img_tile[0][w];
                z0 = z6;

                z6 = img_tile[1][w];
                z1 = z6;
                z2 = -z6;
                z3 = z6;

                z6 = img_tile[2][w];
                z0 -= z6;
                z1 += z6;
                z2 += z6;

                z6 = img_tile[3][w];
                z3 -= z6;

                lds.V[buffer_idx][0][w][local_ic_idx][local_tile_idx] = z0;
                lds.V[buffer_idx][1][w][local_ic_idx][local_tile_idx] = z1;
                lds.V[buffer_idx][2][w][local_ic_idx][local_tile_idx] = z2;
                lds.V[buffer_idx][3][w][local_ic_idx][local_tile_idx] = z3;
            }

            #pragma unroll
            for (int h = 0; h < TILE_IN_H; ++h) {
                z6 = lds.V[buffer_idx][h][0][local_ic_idx][local_tile_idx];
                z0 = z6;

                z6 = lds.V[buffer_idx][h][1][local_ic_idx][local_tile_idx];
                z1 = z6;
                z2 = -z6;
                z3 = z6;

                z6 = lds.V[buffer_idx][h][2][local_ic_idx][local_tile_idx];
                z0 -= z6;
                z1 += z6;
                z2 += z6;

                z6 = lds.V[buffer_idx][h][3][local_ic_idx][local_tile_idx];
                z3 -= z6;

                lds.V[buffer_idx][h][0][local_ic_idx][local_tile_idx] = z0;
                lds.V[buffer_idx][h][1][local_ic_idx][local_tile_idx] = z1;
                lds.V[buffer_idx][h][2][local_ic_idx][local_tile_idx] = z2;
                lds.V[buffer_idx][h][3][local_ic_idx][local_tile_idx] = z3;
            }
        }
    };

    load_tiles(0, 0);
    __syncthreads();

    for (int tile_idx = 0; tile_idx < num_k_tiles; ++tile_idx) {

        const int elem_idx = tid / WARP_SIZE;
        if (elem_idx < TILE_IN_H * TILE_IN_W) {
            const int h = elem_idx / TILE_IN_W;
            const int w = elem_idx % TILE_IN_W;
            const int write_dim_m = (tid % WARP_SIZE) % BLK_K_;
            const int write_dim_n = (tid % WARP_SIZE) / BLK_K_; 

            #pragma unroll
            for (int k = 0; k < BLK_K_; ++k) {
                const float v0 = lds.V[0][h][w][k][write_dim_m +  0];
                const float v1 = lds.V[0][h][w][k][write_dim_m + 16];

                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int col = write_dim_n + t * 4;
                    C_reg_acc[0][0][t] += v0 * lds.Ut[0][h][w][col +  0][k];
                    C_reg_acc[0][1][t] += v0 * lds.Ut[0][h][w][col + 16][k];
                    C_reg_acc[1][0][t] += v1 * lds.Ut[0][h][w][col +  0][k];
                    C_reg_acc[1][1][t] += v1 * lds.Ut[0][h][w][col + 16][k];
                }
            }
        }

        __syncthreads();
        if (tile_idx + 1 < num_k_tiles) {
            const int next_ic_base = (tile_idx + 1) * BLK_K_;
            load_tiles(next_ic_base, 0);
            __syncthreads();
        }
    }

#ifdef NOP_48_CYCLES
#undef NOP_48_CYCLES
#endif

    const int elem_idx = tid / WARP_SIZE;
    if (elem_idx < TILE_IN_H * TILE_IN_W) {
        const int h = elem_idx / TILE_IN_W;
        const int w = elem_idx % TILE_IN_W;
        const size_t write_local_tile = (tid % WARP_SIZE) % TCU_SIZE;
        const size_t write_local_oc = (tid % WARP_SIZE) / TCU_SIZE;

        lds.Y[h][w][write_local_oc +  0 +  0][write_local_tile +  0] = C_reg_acc[0][0].x;
        lds.Y[h][w][write_local_oc +  4 +  0][write_local_tile +  0] = C_reg_acc[0][0].y;
        lds.Y[h][w][write_local_oc +  8 +  0][write_local_tile +  0] = C_reg_acc[0][0].z;
        lds.Y[h][w][write_local_oc + 12 +  0][write_local_tile +  0] = C_reg_acc[0][0].w;

        lds.Y[h][w][write_local_oc +  0 + 16][write_local_tile +  0] = C_reg_acc[0][1].x;
        lds.Y[h][w][write_local_oc +  4 + 16][write_local_tile +  0] = C_reg_acc[0][1].y;
        lds.Y[h][w][write_local_oc +  8 + 16][write_local_tile +  0] = C_reg_acc[0][1].z;
        lds.Y[h][w][write_local_oc + 12 + 16][write_local_tile +  0] = C_reg_acc[0][1].w;

        lds.Y[h][w][write_local_oc +  0 +  0][write_local_tile + 16] = C_reg_acc[1][0].x;
        lds.Y[h][w][write_local_oc +  4 +  0][write_local_tile + 16] = C_reg_acc[1][0].y;
        lds.Y[h][w][write_local_oc +  8 +  0][write_local_tile + 16] = C_reg_acc[1][0].z;
        lds.Y[h][w][write_local_oc + 12 +  0][write_local_tile + 16] = C_reg_acc[1][0].w;

        lds.Y[h][w][write_local_oc +  0 + 16][write_local_tile + 16] = C_reg_acc[1][1].x;
        lds.Y[h][w][write_local_oc +  4 + 16][write_local_tile + 16] = C_reg_acc[1][1].y;
        lds.Y[h][w][write_local_oc +  8 + 16][write_local_tile + 16] = C_reg_acc[1][1].z;
        lds.Y[h][w][write_local_oc + 12 + 16][write_local_tile + 16] = C_reg_acc[1][1].w;
    }

    __syncthreads();

    const int oc_group = thx / BLK_M_;
    const int local_tile_idx = thx % BLK_M_;
    const int oc_stride = BLK_N_ / blockDim.y;
    const int local_oc_idx = oc_group + thy * oc_stride;  // Split output channels across the two threadIdx.y lanes

    if (local_oc_idx < BLK_N_) {
        const int oc = oc_blk + local_oc_idx;
        const int tile = tile_blk + local_tile_idx;
        const bool oc_valid = (oc < C_out);
        const bool tile_valid = (tile < total_tiles);

        TileIndex ti{};
        if (tile_valid) {
            ti = getTileIndex(tile, tiles_per_img, tiles_w);
        }

        const int out_h_base = tile_valid ? ti.th * TILE_OUT_H : 0;
        const int out_w_base = tile_valid ? ti.tw * TILE_OUT_W : 0;
        const size_t out_plane_base = (oc_valid && tile_valid)
            ? (static_cast<size_t>(ti.b) * C_out + oc) * out_H * out_W
            : 0;
        const float bias_val = (bias != nullptr && oc_valid) ? bias[oc] : 0.0f;

        float z0_out, z1_out, z4_out;

        #pragma unroll
        for (int w = 0; w < TILE_IN_W; ++w) {
            z4_out = lds.Y[0][w][local_oc_idx][local_tile_idx];
            z0_out = z4_out;

            z4_out = lds.Y[1][w][local_oc_idx][local_tile_idx];
            z0_out += z4_out;
            z1_out = z4_out;

            z4_out = lds.Y[2][w][local_oc_idx][local_tile_idx];
            z0_out += z4_out;
            z1_out -= z4_out;

            z4_out = lds.Y[3][w][local_oc_idx][local_tile_idx];
            z1_out -= z4_out;

            lds.Y[0][w][local_oc_idx][local_tile_idx] = z0_out;
            lds.Y[1][w][local_oc_idx][local_tile_idx] = z1_out;
        }

        #pragma unroll
        for (int h = 0; h < TILE_OUT_H; ++h) {
            z4_out = lds.Y[h][0][local_oc_idx][local_tile_idx];
            z0_out = z4_out;

            z4_out = lds.Y[h][1][local_oc_idx][local_tile_idx];
            z0_out += z4_out;
            z1_out = z4_out;

            z4_out = lds.Y[h][2][local_oc_idx][local_tile_idx];
            z0_out += z4_out;
            z1_out -= z4_out;

            z4_out = lds.Y[h][3][local_oc_idx][local_tile_idx];
            z1_out -= z4_out;

            if (oc_valid && tile_valid) {
                const int out_h = out_h_base + h;
                const int out_w0 = out_w_base;
                const int out_w1 = out_w_base + 1;

                if (out_h < out_H && out_w0 < out_W) {
                    const size_t out_offset = out_plane_base + static_cast<size_t>(out_h) * out_W + out_w0;
                    const float val0 = z0_out + bias_val;
                    const float val1 = z1_out + bias_val;
                    if (out_w1 < out_W && (out_offset % 2 == 0)) {
                        const float2 vec = {val0, val1};
                        reinterpret_cast<float2*>(output + out_offset)[0] = vec;
                    } else {
                        output[out_offset] = val0;
                        if (out_w1 < out_W) {
                            output[out_offset + 1] = val1;
                        }
                    }
                } else if (out_h < out_H && out_w1 < out_W) {
                    const size_t out_offset = out_plane_base + static_cast<size_t>(out_h) * out_W + out_w1;
                    output[out_offset] = z1_out + bias_val;
                }
            }
        }
    }
}

// 检查输入参数是否满足用 Winograd F(2x2,3x3) 算法的前提条件
bool supports_winograd_f2x3(int N,
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

// 尝试使用 Winograd F2x3 实现执行卷积；成功则返回 true
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
    hipStream_t stream) {
    if (!supports_winograd_f2x3(N, C_in, H, W, C_out, K_h, K_w, out_H, out_W)) {
        return false;
    }

    const int tiles_h = out_H / TILE_OUT_H;
    const int tiles_w = out_W / TILE_OUT_W;
    const int tiles_per_img = tiles_h * tiles_w;
    const int total_tiles = N * tiles_per_img;

    dim3 grid((total_tiles + BLK_M - 1) / BLK_M, (C_out + BLK_N - 1) / BLK_N);
    dim3 block(BLK_M * BLK_K, 2);

    hipLaunchKernelGGL(
        (winograd_2x3_fused_kernel<BLK_M, BLK_N, BLK_K>),
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
