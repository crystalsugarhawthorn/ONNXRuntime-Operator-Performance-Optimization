// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_ROCM_CTX

#include "rocm_resource.h"
#include "core/providers/custom_op_context.h"
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include <unordered_map>
#include <mutex>
#include <cstddef>

namespace Ort {

namespace Custom {

struct RocmContext : public CustomOpContext {
  hipStream_t hip_stream = {};
  miopenHandle_t miopen_handle = {};
  rocblas_handle rblas_handle = {};

  void Init(const OrtKernelContext& kernel_ctx) {
    const auto& ort_api = Ort::GetApi();
    void* resource = {};
    OrtStatus* status = nullptr;

    status = ort_api.KernelContext_GetResource(
        &kernel_ctx, ORT_ROCM_RESOURCE_VERSION, RocmResource::hip_stream_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch hip stream", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    hip_stream = reinterpret_cast<hipStream_t>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(
        &kernel_ctx, ORT_ROCM_RESOURCE_VERSION, RocmResource::miopen_handle_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch miopen handle", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    miopen_handle = reinterpret_cast<miopenHandle_t>(resource);

    resource = {};
    status = ort_api.KernelContext_GetResource(
        &kernel_ctx, ORT_ROCM_RESOURCE_VERSION, RocmResource::rocblas_handle_t, &resource);
    if (status) {
      ORT_CXX_API_THROW("failed to fetch rocblas handle", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
    rblas_handle = reinterpret_cast<rocblas_handle>(resource);
  }

  // Get or allocate a device scratch buffer with at least `bytes` bytes.
  // This caches one buffer per size-bucket (rounded to power-of-two) to avoid
  // repeated device allocations (hipMalloc/hipFree) across operator calls.
  // Returns nullptr on allocation failure.
  void* GetOrAllocScratchBuffer(size_t bytes) const {
    if (bytes == 0) return nullptr;
    size_t bucket = RoundUpPow2(bytes);
    std::lock_guard<std::mutex> lk(scratch_mutex_);
    auto it = scratch_buffers_.find(bucket);
    if (it != scratch_buffers_.end()) {
      return it->second;
    }
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, bucket);
    if (err != hipSuccess) {
      return nullptr;
    }
    scratch_buffers_[bucket] = ptr;
    return ptr;
  }

  // Release all cached scratch buffers. Safe to call multiple times.
  void ReleaseAllScratchBuffers() {
    std::lock_guard<std::mutex> lk(scratch_mutex_);
    for (auto &p : scratch_buffers_) {
      if (p.second) {
        hipFree(p.second);
      }
    }
    scratch_buffers_.clear();
  }

  ~RocmContext() {
    ReleaseAllScratchBuffers();
  }

 private:
  // Round up to next power of two
  static size_t RoundUpPow2(size_t v) {
    if (v == 0) return 0;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFULL
    v |= v >> 32;
#endif
    v += 1;
    return v;
  }

  // mutable so GetOrAllocScratchBuffer can be const
  mutable std::unordered_map<size_t, void*> scratch_buffers_;
  mutable std::mutex scratch_mutex_;
};

}  // namespace Custom
}  // namespace Ort
