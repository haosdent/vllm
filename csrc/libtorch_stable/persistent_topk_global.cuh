#ifndef PERSISTENT_TOPK_GLOBAL_CUH_
#define PERSISTENT_TOPK_GLOBAL_CUH_

#include "persistent_topk.cuh"

namespace vllm {

template <typename IdType>
__device__ __forceinline__ IdType remap_global_index(int local, IdType req,
                                                       const IdType* block_table,
                                                       int bt_s0, int bt_s1,
                                                       int max_blocks, int block_size,
                                                       int num_pool_blocks) {
  if (local < 0) return (IdType)-1;
  int block_id = local / block_size;
  if (block_id < 0 || block_id >= max_blocks) return (IdType)-1;
  IdType base = block_table[(int64_t)req * bt_s0 + (int64_t)block_id * bt_s1];
  // Content guard: a stale/garbage block_table entry (padded rows / freed-block
  // reuse under preemption churn) can point past the KV pool. Turn any such
  // out-of-pool physical block into the benign -1 the downstream already
  // handles, so it can never become an out-of-bounds KV-slot address.
  if (base < (IdType)0 || base >= (IdType)num_pool_blocks) return (IdType)-1;
  return (IdType)(base * block_size + (local % block_size));
}

/*!
 * \brief Filtered Top-K kernel for ragged sequences.
 *
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type (int32_t)
 * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
 */
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K = 2048>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKGlobalKernel(const DType* __restrict__ input,
                              IdType* __restrict__ output,
                              IdType* __restrict__ valid_count,
                              const IdType* __restrict__ lengths,
                              uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len,
                              const IdType* __restrict__ block_table,
                              const IdType* __restrict__ req_id, int bt_s0,
                              int bt_s1, int max_blocks, int block_size,
                              int num_pool_blocks) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  const int length =
      (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
  const DType* score = input + bid * max_len;
  IdType* dst = output + bid * top_k;
  const IdType req = req_id[bid];
  __shared__ int s_valid;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    if (tx == 0) s_valid = 0;
    __syncthreads();
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      const IdType mapped = (i < length) ? remap_global_index<IdType>(i, req, block_table, bt_s0, bt_s1, max_blocks, block_size, num_pool_blocks) : (IdType)-1;
      dst[i] = mapped;
      if (mapped != (IdType)-1) atomicAdd(&s_valid, 1);
    }
    __syncthreads();
    if (tx == 0) valid_count[bid] = s_valid;
    return;
  }

  // Static shared memory
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[MAX_K];

  auto& s_histogram = s_histogram_buf[0];

  // Dynamic shared memory for input double buffer
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  using Traits = FilteredTopKTraits<DType>;
  int topk = top_k;

  // Stage 1: 8-bit coarse histogram with vectorized loads
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
  for (int base = tx * VEC_SIZE; base < aligned_length;
       base += BLOCK_SIZE * VEC_SIZE) {
    score_vec.cast_load(&score[base]);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const auto bin = Traits::ToCoarseKey(score_vec[j]);
      atomicAdd(&s_histogram[bin], 1);
    }
  }
  // Handle tail
  for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
    const auto bin = Traits::ToCoarseKey(score[i]);
    atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // Suffix sum
  const auto run_cumsum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  if (topk == 0) {
    // Collect indices where bin > threshold
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const auto bin = static_cast<int>(Traits::ToCoarseKey(score_vec[j]));
        if (bin > threshold_bin) {
          const auto pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = base + j;
        }
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(score[i]));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = i;
      }
    }
    __syncthreads();
  } else {
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // Filter + histogram for refinement
    auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      } else if (bin == threshold_bin) {
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          s_input_idx[0][pos] = index;
          const auto ordered = Traits::ToOrdered(raw_input);
          const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        filter_and_add_to_histogram(score_vec[j], base + j);
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      filter_and_add_to_histogram(score[i], i);
    }
    __syncthreads();

    // Stage 2: refine with 8bit radix passes
#pragma unroll
    for (int round = 0; round < NUM_ROUNDS; ++round) {
      __shared__ int s_last_remain;
      const auto r_idx = round % 2;

      const auto _raw_num_input = s_num_input[r_idx];
      const auto num_input =
          (_raw_num_input < SMEM_INPUT_SIZE) ? _raw_num_input : SMEM_INPUT_SIZE;

      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[r_idx ^ 1] = 0;
        s_last_remain = topk - s_histogram[tx + 1];
      }
      __syncthreads();

      const auto threshold = s_threshold_bin_id;
      topk -= s_histogram[threshold + 1];

      const int offset = FIRST_SHIFT - round * 8;
      const bool is_last_round = (round == NUM_ROUNDS - 1);

      if (topk == 0) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto bin = (Traits::ToOrdered(score[idx]) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          }
        }
        __syncthreads();
        break;
      } else {
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto raw_input = score[idx];
          const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else if (static_cast<int>(bin) == threshold) {
            if (is_last_round) {
              const auto pos = atomicAdd(&s_last_remain, -1);
              if (pos > 0) {
                s_indices[top_k - pos] = idx;
              }
            } else {
              const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
              if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
                s_input_idx[r_idx ^ 1][pos] = idx;
                const auto bin32 = Traits::ToOrdered(raw_input);
                const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                atomicAdd(&s_histogram[sub_bin], 1);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  // Output phase - mode-specific
  if (tx == 0) s_valid = 0;
  __syncthreads();
#pragma unroll 2
  for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
    const int idx = s_indices[base];
    const IdType mapped = remap_global_index<IdType>(idx, req, block_table, bt_s0, bt_s1, max_blocks, block_size, num_pool_blocks);
    dst[base] = mapped;
    if (mapped != (IdType)-1) atomicAdd(&s_valid, 1);
  }
  __syncthreads();
  if (tx == 0) valid_count[bid] = s_valid;
}

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKGlobalRaggedTransform(const DType* input,
                                              IdType* output_indices,
                                              IdType* valid_count,
                                              const IdType* lengths,
                                              uint32_t num_rows, uint32_t top_k_val,
                                              uint32_t max_len,
                                              const IdType* block_table,
                                              const IdType* req_id, int bt_s0,
                                              int bt_s1, int max_blocks,
                                              int block_size, int num_pool_blocks,
                                              cudaStream_t stream = 0) {
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 16 / sizeof(DType);

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,       &output_indices, &valid_count,
                  &lengths,     &num_rows,       &top_k_val,
                  &max_len,     &block_table,    &req_id,
                  &bt_s0,       &bt_s1,          &max_blocks,
                  &block_size,  &num_pool_blocks};

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                               \
  if (vec_size == VS) {                                                     \
    auto kernel = FilteredTopKGlobalKernel<DType, IdType, VS, MAX_K>;       \
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                              \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));   \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, \
                                          smem_size, stream));              \
    return cudaSuccess;                                                     \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

}  // namespace vllm

#endif  // PERSISTENT_TOPK_GLOBAL_CUH_
