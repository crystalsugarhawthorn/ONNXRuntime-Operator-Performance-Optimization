// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_ROCM

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "core/providers/rocm/rocm_context.h"
#include "onnxruntime_lite_custom_op.h"

extern "C"{ 
	
   // === LeakyReLU ===
    void rocm_leaky_relu(
    int64_t size,
    const float* d_X,
    float* d_Y,
    float alpha,
    hipStream_t stream);
    
  // === Attention ===
	void rocm_attention(int B, int S, int H,
                            const float* Q, const float* K, const float* V,
                            float* Out, hipStream_t stream);
   
  // === BatchNormalization ===
   void rocm_batch_norm(int64_t N, int64_t C, int64_t H, int64_t W,
                            const float* X,
                            const float* gamma,
                            const float* beta,
                            const float* mean,
                            const float* var,
                            float epsilon,
                            float* Y,
                            hipStream_t stream);

  // === GroupNormalization ===
   void rocm_group_norm(
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t G,
    float eps, const float* X, float* Y,
    const float* gamma, const float* beta,
    hipStream_t compute_stream
  );

  // === Conv ===
  void rocm_conv2d(const float* input, const float* weight, const float* bias, float* output,
                   int N, int C_in, int H, int W, int C_out, int K_h, int K_w,
                   int out_H, int out_W, hipStream_t stream);

}



using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

namespace Rocm {
// === LeakyReLU ===
void rocm_leaky_relu_forward(const RocmContext& ctx, const Tensor<float>& X, Tensor<float>& Y) {
  CUSTOM_ENFORCE(ctx.hip_stream, "No HIP stream available");
  int64_t size = X.NumberOfElement();
  const float alpha = 0.01f;
  auto* y_ptr = Y.Allocate(X.Shape());
  rocm_leaky_relu(size, X.Data(), y_ptr, alpha, ctx.hip_stream);
}

// === Attention ===
void rocm_attention_forward(const RocmContext& ctx, const Tensor<float>& Q,
                            const Tensor<float>& K, const Tensor<float>& V, Tensor<float>& Out) {
  CUSTOM_ENFORCE(ctx.hip_stream, "No HIP stream available");
  auto shape = Q.Shape();
  CUSTOM_ENFORCE(shape.size() == 3, "Expected shape [B, S, H]");
  int B = shape[0], S = shape[1], H = shape[2];
  auto* out_ptr = Out.Allocate({B, S, H});
  rocm_attention(B, S, H, Q.Data(), K.Data(), V.Data(), out_ptr, ctx.hip_stream);
}

// === BatchNormalization ===
void rocm_batchnorm_forward(const RocmContext& ctx, const Tensor<float>& X,
                             const Tensor<float>& scale, const Tensor<float>& B,
                             const Tensor<float>& mean, const Tensor<float>& var,
                             Tensor<float>& Y) {
  CUSTOM_ENFORCE(ctx.hip_stream, "No HIP stream available");
  auto shape = X.Shape();
  CUSTOM_ENFORCE(shape.size() == 4, "Expected shape [N, C, H, W]");
  int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
  float epsilon = 1e-5f;
  auto* y_ptr = Y.Allocate({N, C, H, W});
  rocm_batch_norm(N, C, H, W, X.Data(), scale.Data(), B.Data(), mean.Data(), var.Data(), epsilon, y_ptr, ctx.hip_stream);
}

// === GroupNormalization ===
void rocm_groupnorm_forward(const RocmContext& ctx, const Tensor<int64_t>& G,
                             const Tensor<float>& X, const Tensor<float>& gamma,
                             const Tensor<float>& beta, Tensor<float>& Y) {
  CUSTOM_ENFORCE(ctx.hip_stream, "No HIP stream available");
  auto shape = X.Shape();
  CUSTOM_ENFORCE(shape.size() == 4, "Expected shape [N, C, H, W]");
  int64_t num_groups = *G.Data();
  int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
  float epsilon = 1e-5f;
  auto* y_ptr = Y.Allocate({N, C, H, W});
  rocm_group_norm(N, C, H, W, num_groups, epsilon, X.Data(), y_ptr, gamma.Data(), beta.Data(), ctx.hip_stream);
}

// === Conv ===
void rocm_conv_forward(const RocmContext& ctx, const Tensor<float>& input,
                       const Tensor<float>& weight, const Tensor<float>& bias,
                       Tensor<float>& output) {
  CUSTOM_ENFORCE(ctx.hip_stream, "No HIP stream available");
  auto input_shape = input.Shape();
  auto weight_shape = weight.Shape();
  int64_t N = input_shape[0], C_in = input_shape[1], H = input_shape[2], W = input_shape[3];
  int64_t C_out = weight_shape[0], K_h = weight_shape[2], K_w = weight_shape[3];
  int64_t out_H = (H - K_h) + 1;
  int64_t out_W = (W - K_w) + 1;
  auto* y_ptr = output.Allocate({N, C_out, out_H, out_W});
  rocm_conv2d(input.Data(), weight.Data(), bias.Data(), y_ptr, N, C_in, H, W, C_out, K_h, K_w, out_H, out_W, ctx.hip_stream);
}

void RegisterOps(Ort::CustomOpDomain& domain) {

  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpAttention{Ort::Custom::CreateLiteCustomOp("Attention", "ROCMExecutionProvider", rocm_attention_forward)};
 domain.Add(c_CustomOpAttention.get());
 
 static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpBatchNorm{Ort::Custom::CreateLiteCustomOp("BatchNormalization", "ROCMExecutionProvider", rocm_batchnorm_forward)};
 domain.Add(c_CustomOpBatchNorm.get());
 
 static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpGroupNorm{Ort::Custom::CreateLiteCustomOp("GroupNormalization", "ROCMExecutionProvider", rocm_groupnorm_forward)};
 domain.Add(c_CustomOpGroupNorm.get());
 
 static const std::unique_ptr<OrtLiteCustomOp> c_LeakyReLU{
     Ort::Custom::CreateLiteCustomOp(
         "LeakyRelu", "ROCMExecutionProvider", rocm_leaky_relu_forward)};
 domain.Add(c_LeakyReLU.get());

 static const std::unique_ptr<OrtLiteCustomOp> c_Conv{
     Ort::Custom::CreateLiteCustomOp("Conv", "ROCMExecutionProvider", rocm_conv_forward)};
 domain.Add(c_Conv.get());
}
  
}  // namespace Rocm

#endif}}}
