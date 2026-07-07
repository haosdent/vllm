// Persistent TopK global-index kernel for DeepSeek V3 sparse attention indexer.
// See persistent_topk_global.cuh for kernel implementation.

#include <cuda_runtime.h>
#include <algorithm>

#include "torch_utils.h"

#ifndef USE_ROCM
  #include "persistent_topk_global.cuh"
#endif

void persistent_topk_global(const torch::stable::Tensor& logits,
                             const torch::stable::Tensor& lengths,
                             torch::stable::Tensor& output,
                             torch::stable::Tensor& valid_count,
                             const torch::stable::Tensor& block_table,
                             const torch::stable::Tensor& req_id,
                             int64_t k, int64_t max_seq_len,
                             int64_t block_size, int64_t num_pool_blocks) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  STD_TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  STD_TORCH_CHECK(valid_count.is_cuda(), "valid_count must be CUDA tensor");
  STD_TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA tensor");
  STD_TORCH_CHECK(req_id.is_cuda(), "req_id must be CUDA tensor");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "Only float32 supported");
  STD_TORCH_CHECK(lengths.scalar_type() == torch::headeronly::ScalarType::Int,
                  "lengths must be int32");
  STD_TORCH_CHECK(output.scalar_type() == torch::headeronly::ScalarType::Int,
                  "output must be int32");
  STD_TORCH_CHECK(valid_count.scalar_type() == torch::headeronly::ScalarType::Int,
                  "valid_count must be int32");
  STD_TORCH_CHECK(block_table.scalar_type() == torch::headeronly::ScalarType::Int,
                  "block_table must be int32");
  STD_TORCH_CHECK(req_id.scalar_type() == torch::headeronly::ScalarType::Int,
                  "req_id must be int32");
  STD_TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  STD_TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
                  "lengths must be 1D or 2D");
  STD_TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  STD_TORCH_CHECK(output.dim() == 2, "output must be 2D");
  STD_TORCH_CHECK(valid_count.dim() == 1, "valid_count must be 1D");
  STD_TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  STD_TORCH_CHECK(req_id.dim() == 1, "req_id must be 1D");

  const int64_t num_rows = logits.size(0);
  const int64_t max_len = logits.stride(0);
  const int64_t bt_s0 = block_table.stride(0);
  const int64_t bt_s1 = block_table.stride(1);
  const int64_t max_blocks = block_table.size(1);

  STD_TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  STD_TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
                  "output size mismatch");
  STD_TORCH_CHECK(valid_count.numel() == num_rows,
                  "valid_count size mismatch");
  STD_TORCH_CHECK(req_id.numel() == num_rows, "req_id size mismatch");
  STD_TORCH_CHECK(
      k == 512 || k == 1024 || k == 2048,
      "persistent_topk_global supports k=512, k=1024, or k=2048, got k=", k);
  STD_TORCH_CHECK(num_rows > 32,
                  "persistent_topk_global only supports FilteredTopK regime num_rows>32");

  static_cast<void>(max_seq_len);

  const cudaStream_t stream = get_current_cuda_stream();
  cudaError_t status = cudaSuccess;

  if (k == 512) {
    status = vllm::FilteredTopKGlobalRaggedTransform<float, int32_t, 512>(
        logits.const_data_ptr<float>(), output.mutable_data_ptr<int32_t>(),
        valid_count.mutable_data_ptr<int32_t>(),
        lengths.const_data_ptr<int32_t>(), static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(k), static_cast<uint32_t>(max_len),
        block_table.const_data_ptr<int32_t>(), req_id.const_data_ptr<int32_t>(),
        static_cast<int>(bt_s0), static_cast<int>(bt_s1),
        static_cast<int>(max_blocks), static_cast<int>(block_size),
        static_cast<int>(num_pool_blocks), stream);
  } else if (k == 1024) {
    status = vllm::FilteredTopKGlobalRaggedTransform<float, int32_t, 1024>(
        logits.const_data_ptr<float>(), output.mutable_data_ptr<int32_t>(),
        valid_count.mutable_data_ptr<int32_t>(),
        lengths.const_data_ptr<int32_t>(), static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(k), static_cast<uint32_t>(max_len),
        block_table.const_data_ptr<int32_t>(), req_id.const_data_ptr<int32_t>(),
        static_cast<int>(bt_s0), static_cast<int>(bt_s1),
        static_cast<int>(max_blocks), static_cast<int>(block_size),
        static_cast<int>(num_pool_blocks), stream);
  } else {
    status = vllm::FilteredTopKGlobalRaggedTransform<float, int32_t, 2048>(
        logits.const_data_ptr<float>(), output.mutable_data_ptr<int32_t>(),
        valid_count.mutable_data_ptr<int32_t>(),
        lengths.const_data_ptr<int32_t>(), static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(k), static_cast<uint32_t>(max_len),
        block_table.const_data_ptr<int32_t>(), req_id.const_data_ptr<int32_t>(),
        static_cast<int>(bt_s0), static_cast<int>(bt_s1),
        static_cast<int>(max_blocks), static_cast<int>(block_size),
        static_cast<int>(num_pool_blocks), stream);
  }
  STD_TORCH_CHECK(status == cudaSuccess,
                  "FilteredTopKGlobal failed: ", cudaGetErrorString(status));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "persistent_topk_global failed: ", cudaGetErrorString(err));
#else
  STD_TORCH_CHECK(false, "not supported on ROCm");
#endif
}
