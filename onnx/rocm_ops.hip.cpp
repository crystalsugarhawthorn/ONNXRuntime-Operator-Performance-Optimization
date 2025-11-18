#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <miopen/miopen.h>
#include <stdio.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "winograd/winograd_conv2d.h"

// ==========================================================================================
// Attention 算子实现
// ==========================================================================================

#define WARP_SIZE 64
#define WARPS_PER_BLOCK 4

// 在同一 warp 内执行最大值归约并广播结果
__device__ float warpAllReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset, WARP_SIZE));
    }
    return __shfl(val, 0, WARP_SIZE);
}

// 在同一 warp 内执行求和归约并广播结果
__device__ float warpAllReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WARP_SIZE);
    }
    return __shfl(val, 0, WARP_SIZE);
}


// 对每个 batch 的 scores 行执行数值稳定的 softmax（使用共享内存）
__global__ void _Softmax(int B, int S, float* scores) {
    extern __shared__ float shared_scores[];
    
    const int b = blockIdx.z;
    const int warp_id_in_block = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int global_warp_id = blockIdx.y * WARPS_PER_BLOCK + warp_id_in_block;
    const int i = global_warp_id;
    
    if (b >= B || i >= S) return;
    
    float* my_scores = &shared_scores[warp_id_in_block * S];
    const int score_base = (b * S + i) * S;
    
    for (int k = lane_id; k < S; k += WARP_SIZE) {
        my_scores[k] = scores[score_base + k];
    }
    __syncthreads();
    
    float max_val = -1e20f;
    for (int k = lane_id; k < S; k += WARP_SIZE) {
        max_val = fmaxf(max_val, my_scores[k]);
    }
    max_val = warpAllReduceMax(max_val);
    
    float sum = 0.0f;
    for (int k = lane_id; k < S; k += WARP_SIZE) {
        float val = expf(my_scores[k] - max_val);
        my_scores[k] = val;
        sum += val;
    }
    sum = warpAllReduceSum(sum);
    
    for (int k = lane_id; k < S; k += WARP_SIZE) {
        scores[score_base + k] = my_scores[k] / (sum + 1e-6f);
    }
}


// Attention 入口：计算 Q@K^T -> softmax -> 与 V 相乘，结果写入 Out
extern "C" void rocm_attention(int B, int S, int H,
                                const float* Q, const float* K, const float* V,
                                float* Out, hipStream_t stream) {
    
    static hipblasHandle_t handle = nullptr;
    if (handle == nullptr) {
        hipblasCreate(&handle);
        hipblasSetStream(handle, stream);
    } else {
        hipblasSetStream(handle, stream);
    }
    
    float* d_scores;
    size_t scores_size = B * S * S * sizeof(float);
    hipMalloc(&d_scores, scores_size);

    float scale = 1.0f / sqrtf((float)H);
    float beta = 0.0f;
    

    hipblasSgemmStridedBatched(
        handle,
        HIPBLAS_OP_T,     // K需要转置
        HIPBLAS_OP_N,     // Q不转置
        S,                // K^T的行数 = S
        S,                // Q的列数（实际意义上的行数）= S  
        H,                // K^T的列数 = Q的行数（实际意义上的列数）= H
        &scale,           // alpha（缩放因子）
        K,                // K矩阵的指针
        H,                // K的leading dimension（行主序下是H）
        S * H,            // K的batch stride
        Q,                // Q矩阵的指针
        H,                // Q的leading dimension
        S * H,            // Q的batch stride
        &beta,            // beta = 0
        d_scores,         // 输出scores
        S,                // scores的leading dimension
        S * S,            // scores的batch stride
        B                 // batch数量
    );
    
    dim3 blockDim(WARP_SIZE, WARPS_PER_BLOCK);
    dim3 gridDim(1, (S + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
    size_t shared_mem_size = WARPS_PER_BLOCK * S * sizeof(float);
    
    _Softmax<<<gridDim, blockDim, shared_mem_size, stream>>>(B, S, d_scores);
    

    float alpha = 1.0f;
    beta = 0.0f;
    
    hipblasSgemmStridedBatched(
        handle,
        HIPBLAS_OP_N,     // V不转置
        HIPBLAS_OP_N,     // scores不转置
        H,                // V的列数（实际意义上）= H
        S,                // scores的行数（实际意义上）= S
        S,                // scores的列数 = V的行数 = S
        &alpha,
        V,                // V矩阵
        H,                // V的leading dimension
        S * H,            // V的batch stride
        d_scores,         // scores矩阵
        S,                // scores的leading dimension
        S * S,            // scores的batch stride
        &beta,
        Out,              // 输出矩阵
        H,                // Out的leading dimension
        S * H,            // Out的batch stride
        B                 // batch数量
    );
    
    hipStreamSynchronize(stream);
    hipFree(d_scores);
}


// ==========================================================================================
// BatchNormalization 算子实现
// ==========================================================================================

// 计算并写回融合后的 BN scale 和 shift（用于前向应用）
__global__ void _CalculateFusedBNParamsKernel(
    int64_t C,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float epsilon,
    float* __restrict__ d_scale,
    float* __restrict__ d_shift) {

    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float inv_std = rsqrtf(var[c] + epsilon);
    float scale_val = gamma[c] * inv_std;
    float shift_val = beta[c] - mean[c] * scale_val;

    d_scale[c] = scale_val;
    d_shift[c] = shift_val;
}

// 向量化的 BatchNormalization 主内核，处理以 float4 对齐的大部分数据
__global__ void _BatchNormalization(
    const int64_t total_float4,
    const int C,
    const int H,
    const int W,
    const float4* __restrict__ X,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    float4* __restrict__ Y)
{
    const int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx4 < total_float4;
         idx4 += grid_stride)
    {
        float4 data = X[idx4];

        const int64_t start_idx_float = idx4 * 4; 
        const int64_t hw_plane_size = (int64_t)H * (int64_t)W; 

        int c0, c1, c2, c3;

        if (hw_plane_size > 3) {
            int64_t q = start_idx_float / hw_plane_size;       
            int64_t pos = start_idx_float - q * hw_plane_size; 

            int base_c = (int)(q % (int64_t)C);

            int inc1 = (pos + 1 >= hw_plane_size) ? 1 : 0;
            int inc2 = (pos + 2 >= hw_plane_size) ? 1 : 0;
            int inc3 = (pos + 3 >= hw_plane_size) ? 1 : 0;

            c0 = base_c;
            c1 = base_c + inc1; if (c1 >= C) c1 -= C;
            c2 = base_c + inc2; if (c2 >= C) c2 -= C;
            c3 = base_c + inc3; if (c3 >= C) c3 -= C;
        } else {
            const int64_t i0 = start_idx_float + 0;
            const int64_t i1 = start_idx_float + 1;
            const int64_t i2 = start_idx_float + 2;
            const int64_t i3 = start_idx_float + 3;
            c0 = (int)((i0 / hw_plane_size) % C);
            c1 = (int)((i1 / hw_plane_size) % C);
            c2 = (int)((i2 / hw_plane_size) % C);
            c3 = (int)((i3 / hw_plane_size) % C);
        }

        float s0 = scale[c0], sh0 = shift[c0];
        float s1 = scale[c1], sh1 = shift[c1];
        float s2 = scale[c2], sh2 = shift[c2];
        float s3 = scale[c3], sh3 = shift[c3];

        data.x = fmaf(data.x, s0, sh0);
        data.y = fmaf(data.y, s1, sh1);
        data.z = fmaf(data.z, s2, sh2);
        data.w = fmaf(data.w, s3, sh3);

        Y[idx4] = data;
    }
}

// 处理 BatchNormalization 中剩余不足 4 个元素的尾部内核
__global__ void _BatchNormalizationRemainder(
    const int64_t total_elements,
    const int C,
    const int H,
    const int W,
    const float* __restrict__ X,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    float* __restrict__ Y)
{
    const int64_t remainder_start_idx = (total_elements / 4) * 4;

    int64_t idx = remainder_start_idx + threadIdx.x;
    if (idx < total_elements) {
        const int hw_plane_size = H * W;
        const int c = (idx / hw_plane_size) % C;
        Y[idx] = fmaf(X[idx], scale[c], shift[c]);
    }
}


// 批归一化主机接口：准备参数并启动内核执行
extern "C" void rocm_batch_norm(
    int64_t N, int64_t C, int64_t H, int64_t W,
    const float* X,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    float epsilon,
    float* Y,
    hipStream_t stream) {

    const size_t total = (size_t)N * C * H * W;
    if (total == 0 || C == 0) return;

    static float* d_scale = nullptr;
    static float* d_shift = nullptr;
    static int64_t cached_C = 0;

    if (C != cached_C) {
        if (d_scale) hipFree(d_scale);
        if (d_shift) hipFree(d_shift);

        hipMallocAsync(&d_scale, C * sizeof(float), stream);
        hipMallocAsync(&d_shift, C * sizeof(float), stream);
        cached_C = C;
    }

    {
        const int threads = 256;
        const int blocks = (C + threads - 1) / threads;
        _CalculateFusedBNParamsKernel<<<blocks, threads, 0, stream>>>(
            C, gamma, beta, mean, var, epsilon, d_scale, d_shift);
    }

    const int threads_per_block = 256; 

    const int64_t total_float4 = total / 4;
    if (total_float4 > 0) {
        const int blocks = std::min((int64_t)65535, (total_float4 + threads_per_block - 1) / threads_per_block);
        _BatchNormalization<<<blocks, threads_per_block, 0, stream>>>(
            total_float4, C, H, W,
            reinterpret_cast<const float4*>(X),
            d_scale,
            d_shift,
            reinterpret_cast<float4*>(Y)
        );
    }

    const int64_t remainder = total % 4;
    if (remainder > 0) {
        _BatchNormalizationRemainder<<<1, remainder, 0, stream>>>(
            total, C, H, W, X, d_scale, d_shift, Y
        );
    }
}

// ==========================================================================================
// Conv2D 算子实现（新增winogard算法，调用文件位置为./winogard）
// ==========================================================================================

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 16

#define THREAD_M 8
#define THREAD_N 8

#define THREADS_PER_BLOCK_X (BLOCK_N / THREAD_N)
#define THREADS_PER_BLOCK_Y (BLOCK_M / THREAD_M)

// 将权重矩阵的一块 Tile 以向量化方式加载到共享内存
__device__ void load_A_tile_vectorized(
    const float* weight,
    float* s_A_dest,
    int g_m_start, int g_k_start,
    int M, int K)
{
    const int num_vec4_elements = (BLOCK_M * BLOCK_K) / 4;
    const int threads_per_block = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y;
    
    for (int i = threadIdx.y * THREADS_PER_BLOCK_X + threadIdx.x; i < num_vec4_elements; i += threads_per_block) {
        int m = (i * 4) / BLOCK_K;
        int k_start = (i * 4) % BLOCK_K;
        int g_m = g_m_start + m;
        int g_k = g_k_start + k_start;

        const float* g_ptr = weight + (size_t)g_m * K + g_k;
        float4* s_ptr = (float4*)s_A_dest;

        if (g_m < M && (g_k + 3) < K) {
            s_ptr[i] = *((const float4*)g_ptr);
        } else {
            for (int k_offset = 0; k_offset < 4; ++k_offset) {
                if (g_m < M && (g_k + k_offset) < K) {
                    s_A_dest[m * BLOCK_K + k_start + k_offset] = weight[(size_t)g_m * K + g_k + k_offset];
                } else {
                    s_A_dest[m * BLOCK_K + k_start + k_offset] = 0.0f;
                }
            }
        }
    }
}

// 将输入（隐式 im2col）的一块 Tile 加载到共享内存并进行边界填充
__device__ void load_B_tile(
    const float* input,
    float* s_B_dest,
    int g_n_start, int g_k_start,
    int N_batch_idx, int C_in, int H, int W,
    int K_h, int K_w, int out_H, int out_W,
    int N_gemm, int K)
{
    for (int i = threadIdx.y * THREADS_PER_BLOCK_X + threadIdx.x; i < BLOCK_K * BLOCK_N; i += THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y) {
        int k = i / BLOCK_N;
        int n = i % BLOCK_N;
        int g_k = g_k_start + k;
        int g_n = g_n_start + n;

        if (g_k < K && g_n < N_gemm) {
            const int ic = g_k / (K_h * K_w);            // 输入通道
            const int kh = (g_k % (K_h * K_w)) / K_w;    // 滤波器行
            const int kw = g_k % K_w;                    // 滤波器列
            const int oh = g_n / out_W;                  // 输出特征图行
            const int ow = g_n % out_W;                  // 输出特征图列
            const int ih = oh + kh;                      // 输入特征图行
            const int iw = ow + kw;                      // 输入特征图列

            const size_t input_offset = (size_t)N_batch_idx * C_in * H * W + (size_t)ic * H * W + (size_t)ih * W + iw;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                s_B_dest[k * BLOCK_N + n] = input[input_offset];
            } else {
                s_B_dest[k * BLOCK_N + n] = 0.0f;
            }
        } else {
            s_B_dest[k * BLOCK_N + n] = 0.0f;
        }
    }
}


// 基于分块 GEMM 的 Conv2D 内核，实现寄存器分块与双缓冲软件流水线
__global__ __launch_bounds__(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
void _Conv2dKernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N_batch, int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int out_H, int out_W) {

    const int M = C_out;
    const int N = out_H * out_W;
    const int K = C_in * K_h * K_w;

    const int C_tile_m_start = blockIdx.y * BLOCK_M;
    const int C_tile_n_start = blockIdx.x * BLOCK_N;
    const int batch_idx = blockIdx.z;

    extern __shared__ float smem[];
    const size_t s_A_tile_size = BLOCK_M * BLOCK_K;
    const size_t s_B_tile_size = BLOCK_K * BLOCK_N;
    
    float* s_A_buffers[2];
    s_A_buffers[0] = smem;
    s_A_buffers[1] = smem + s_A_tile_size;

    float* s_B_buffers[2];
    s_B_buffers[0] = smem + 2 * s_A_tile_size;
    s_B_buffers[1] = smem + 2 * s_A_tile_size + s_B_tile_size;

    float accum[THREAD_M][THREAD_N] = {{0.0f}};

    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    if (num_k_tiles > 0) {
        load_A_tile_vectorized(weight, s_A_buffers[0], C_tile_m_start, 0, M, K);
        load_B_tile(input, s_B_buffers[0], C_tile_n_start, 0, batch_idx, C_in, H, W, K_h, K_w, out_H, out_W, N, K);
    }
    __syncthreads();

    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles - 1; ++k_tile_idx) {
        const int compute_buf_idx = k_tile_idx % 2;
        const int load_buf_idx = (k_tile_idx + 1) % 2;

        load_A_tile_vectorized(weight, s_A_buffers[load_buf_idx], C_tile_m_start, (k_tile_idx + 1) * BLOCK_K, M, K);
        load_B_tile(input, s_B_buffers[load_buf_idx], C_tile_n_start, (k_tile_idx + 1) * BLOCK_K, batch_idx, C_in, H, W, K_h, K_w, out_H, out_W, N, K);

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                int s_A_m = threadIdx.y * THREAD_M + m;
                float s_A_val = s_A_buffers[compute_buf_idx][s_A_m * BLOCK_K + k];
                #pragma unroll
                for (int n = 0; n < THREAD_N; ++n) {
                    int s_B_n = threadIdx.x * THREAD_N + n;
                    accum[m][n] += s_A_val * s_B_buffers[compute_buf_idx][k * BLOCK_N + s_B_n];
                }
            }
        }
        __syncthreads();
    }

    if (num_k_tiles > 0) {
        const int last_buf_idx = (num_k_tiles - 1) % 2;
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                int s_A_m = threadIdx.y * THREAD_M + m;
                float s_A_val = s_A_buffers[last_buf_idx][s_A_m * BLOCK_K + k];
                #pragma unroll
                for (int n = 0; n < THREAD_N; ++n) {
                    int s_B_n = threadIdx.x * THREAD_N + n;
                    accum[m][n] += s_A_val * s_B_buffers[last_buf_idx][k * BLOCK_N + s_B_n];
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        int g_m = C_tile_m_start + threadIdx.y * THREAD_M + m;
        int g_n = C_tile_n_start + threadIdx.x * THREAD_N;

        if (g_m < M && (g_n + THREAD_N - 1) < N) {
            float final_val[THREAD_N];
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                final_val[n] = accum[m][n];
                if (bias != nullptr) {
                    final_val[n] += bias[g_m];
                }
            }
            
            *(reinterpret_cast<float4*>(output + (size_t)batch_idx * M * N + (size_t)g_m * N + g_n)) =
                *reinterpret_cast<float4*>(final_val);
        } else {
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                if (g_m < M && (g_n + n) < N) {
                    float final_val = accum[m][n];
                    if (bias != nullptr) final_val += bias[g_m];
                    output[(size_t)batch_idx * M * N + (size_t)g_m * N + g_n + n] = final_val;
                }
            }
        }
    }
}

// Conv2D 主机接口：优先尝试 Winograd，否则调用分块 GEMM 内核
extern "C" void rocm_conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int out_H, int out_W,
    hipStream_t stream) {
    if (try_run_winograd_conv2d(
            input, weight, bias, output,
            N, C_in, H, W, C_out,
            K_h, K_w, out_H, out_W, stream)) {
        return;
    }
    
    dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 blocks(
        (out_H * out_W + BLOCK_N - 1) / BLOCK_N,
        (C_out + BLOCK_M - 1) / BLOCK_M,
        N
    );

    size_t total_smem_bytes = 2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float);
    
    hipLaunchKernelGGL(
        _Conv2dKernel,
        blocks,
        threads,
        total_smem_bytes,
        stream,
        input, weight, bias, output,
        N, C_in, H, W, C_out, K_h, K_w, out_H, out_W);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Conv2D Kernel (Reg Tiling) launch failed: %s\n", hipGetErrorString(err));
    }
}

// ==========================================================================================
// LeakyReLU 算子实现
// ==========================================================================================

// 向量化的 LeakyReLU 内核，使用 float4 批量读写并避免分支
__global__ void _LeakyReLU_(const float* __restrict__ X, float* __restrict__ Y,
                                           int64_t size, float alpha) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    int64_t n_vec4 = size / 4;
    for (int64_t i = tid; i < n_vec4; i += stride) {
        const int64_t data_idx = i * 4;

        float4 x4 = *(reinterpret_cast<const float4*>(&X[data_idx]));
        float4 y4;

        y4.x = fmaxf(x4.x, alpha * x4.x);
        y4.y = fmaxf(x4.y, alpha * x4.y);
        y4.z = fmaxf(x4.z, alpha * x4.z);
        y4.w = fmaxf(x4.w, alpha * x4.w);

        *(reinterpret_cast<float4*>(&Y[data_idx])) = y4;
    }

    for (int64_t i = n_vec4 * 4 + tid; i < size; i += stride) {
        float x = X[i];
        Y[i] = fmaxf(x, alpha * x);
    }
}

// LeakyReLU 主机接口：配置网格并启动 LeakyReLU 内核
extern "C" void rocm_leaky_relu(
    int64_t size,
    const float* d_X,
    float* d_Y,
    float alpha,
    hipStream_t stream) {
    if (size <= 0) return; 

    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);

    int grid_size = (size + block.x - 1) / block.x;
    const int max_grid_size = 4096;
    grid_size = std::min(grid_size, max_grid_size);
    dim3 grid(grid_size);
    
    _LeakyReLU_<<<grid, block, 0, stream>>>(d_X, d_Y, size, alpha);
}

// ==========================================================================================
// GroupNormalization 算子实现
// ==========================================================================================

#define WARP_SIZE 64

// 在同一 warp 内执行浮点求和归约并返回结果
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WARP_SIZE);
    }
    return val;
}

// 优化的 GroupNormalization 内核：计算组内均值/方差并应用 gamma/beta
__global__ void _GroupNormalization(
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t G,
    float eps,
    const float* __restrict__ X,
    float* __restrict__ Y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta
) {
    const int64_t group_idx = blockIdx.x;
    const int64_t n = blockIdx.y;
    
    const int64_t channels_per_group = C / G;
    const int64_t c_start = group_idx * channels_per_group;
    const int64_t group_size = channels_per_group * H * W;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    
    __shared__ float warp_sum[8];
    __shared__ float warp_sum_sq[8];
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    const int64_t base_idx = n * C * H * W + c_start * H * W;
    const int64_t vec4_size = group_size / 4;

    for (int64_t i = tid; i < vec4_size; i += blockDim.x) {
        float4 data = reinterpret_cast<const float4*>(&X[base_idx + i * 4])[0];
        
        local_sum += data.x + data.y + data.z + data.w;
        local_sum_sq += data.x * data.x + data.y * data.y + 
                        data.z * data.z + data.w * data.w;
    }

    for (int64_t i = vec4_size * 4 + tid; i < group_size; i += blockDim.x) {
        float val = X[base_idx + i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    local_sum = warpReduceSum(local_sum);
    local_sum_sq = warpReduceSum(local_sum_sq);

    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < num_warps) ? warp_sum[lane_id] : 0.0f;
        float sum_sq = (lane_id < num_warps) ? warp_sum_sq[lane_id] : 0.0f;
        
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
        
        if (lane_id == 0) {
            warp_sum[0] = sum / group_size;
            warp_sum_sq[0] = sum_sq / group_size;
        }
    }
    __syncthreads();
    
    const float mean = warp_sum[0];
    const float variance = warp_sum_sq[0] - mean * mean;
    const float inv_std = rsqrtf(variance + eps);

    for (int64_t i = tid; i < vec4_size; i += blockDim.x) {
        const int64_t vec_offset = i * 4;
        const int64_t c_offset = vec_offset / (H * W);
        const int64_t c = c_start + c_offset;
        
        float4 data = reinterpret_cast<const float4*>(&X[base_idx + vec_offset])[0];

        data.x = (data.x - mean) * inv_std;
        data.y = (data.y - mean) * inv_std;
        data.z = (data.z - mean) * inv_std;
        data.w = (data.w - mean) * inv_std;

        const int64_t idx0 = vec_offset;
        const int64_t idx1 = vec_offset + 1;
        const int64_t idx2 = vec_offset + 2;
        const int64_t idx3 = vec_offset + 3;
        
        const int64_t c0 = c_start + idx0 / (H * W);
        const int64_t c1 = c_start + idx1 / (H * W);
        const int64_t c2 = c_start + idx2 / (H * W);
        const int64_t c3 = c_start + idx3 / (H * W);
        
        data.x = gamma[c0] * data.x + beta[c0];
        data.y = gamma[c1] * data.y + beta[c1];
        data.z = gamma[c2] * data.z + beta[c2];
        data.w = gamma[c3] * data.w + beta[c3];
        
        reinterpret_cast<float4*>(&Y[base_idx + vec_offset])[0] = data;
    }

    for (int64_t i = vec4_size * 4 + tid; i < group_size; i += blockDim.x) {
        const int64_t c = c_start + i / (H * W);
        const float val = X[base_idx + i];
        const float normalized = (val - mean) * inv_std;
        Y[base_idx + i] = gamma[c] * normalized + beta[c];
    }
}

// GroupNormalization 主机接口：检查参数并启动归一化内核
extern "C" void rocm_group_norm(
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t G,
    float eps, 
    const float* X,      
    float* Y,            
    const float* gamma,  
    const float* beta,   
    hipStream_t compute_stream
) {
    if (C % G != 0) {
        fprintf(stderr, "Error: Channels must be divisible by groups.\n");
        return;
    }

    dim3 block_dim(256);
    dim3 grid_dim(G, N);
    
    _GroupNormalization<<<grid_dim, block_dim, 0, compute_stream>>>(
        N, C, H, W, G, eps, X, Y, gamma, beta
    );
}