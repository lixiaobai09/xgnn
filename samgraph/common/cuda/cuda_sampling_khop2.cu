/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <curand.h>
#include <curand_kernel.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cub/cub.cuh>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_function.h"
#include "cuda_utils.h"

#define ORIGIN_KHOP2
// #define KHOP2_REVERSION_NO_MODIFIED

namespace samgraph {
namespace common {
namespace cuda {

namespace {

#if (defined ORIGIN_KHOP2)

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_khop2(const IdType *indptr, IdType *indices,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  assert(i < num_random_states);
  curandState local_state = random_states[i];

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      const IdType rid = input[index];
      const IdType off = indptr[rid];
      const IdType len = indptr[rid + 1] - indptr[rid];

      if (len <= fanout) {
        size_t j = 0;
        for (; j < len; ++j) {
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = indices[off + j];
        }

        for (; j < fanout; ++j) {
          tmp_src[index * fanout + j] = Constant::kEmptyKey;
          tmp_dst[index * fanout + j] = Constant::kEmptyKey;
        }
      } else {
        for (size_t j = 0; j < fanout; ++j) {
          // now we have $len - j$ cancidate
          size_t selected_j = curand(&local_state) % (len - j);
          IdType selected_node_id = indices[off + selected_j];
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = selected_node_id;
          // swap indices[off+len-j-1] with indices[off+ selected_j];
          indices[off + selected_j] = indices[off+len-j-1];
          indices[off+len-j-1] = selected_node_id;
        }
      }
    }
  }

  random_states[i] = local_state;
}

#elif (defined KHOP2_REVERSION_NO_MODIFIED)

constexpr int HASH_SHIFT = 7;
constexpr int HASH_MASK  = ((1 << HASH_SHIFT) - 1);
constexpr int HASHTABLE_SIZE = (1 << HASH_SHIFT);

struct HashItem {
  IdType key;
  IdType value;
};

__forceinline__ __device__ IdType _FindPos(HashItem* hash_table, IdType key) {
  IdType pos = (key & HASH_MASK);
  IdType offset = 1;
  while(true) {
    IdType old_key = hash_table[pos].key;
    if (old_key == Constant::kEmptyKey || old_key == key) {
      return pos;
    }
    pos = ((pos + offset) & HASH_MASK);
    assert(offset < HASHTABLE_SIZE);
    ++offset;
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_khop2(const IdType *indptr, IdType *indices,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  assert(i < num_random_states);
  curandState local_state = random_states[i];

  assert(2 * fanout < HASHTABLE_SIZE);
  HashItem hash_table[HASHTABLE_SIZE];
  for (IdType j = 0; j < HASHTABLE_SIZE; ++j) {
    hash_table[j].key = Constant::kEmptyKey;
  }

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      const IdType rid = input[index];
      const IdType off = indptr[rid];
      const IdType len = indptr[rid + 1] - indptr[rid];

      if (len <= fanout) {
        size_t j = 0;
        for (; j < len; ++j) {
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = indices[off + j];
        }

        for (; j < fanout; ++j) {
          tmp_src[index * fanout + j] = Constant::kEmptyKey;
          tmp_dst[index * fanout + j] = Constant::kEmptyKey;
        }
      } else {
        for (IdType j = 0; j < fanout; ++j) {
          hash_table[j].key = j;
          hash_table[j].value = j;
        }
        IdType *tmp_idx = (tmp_src + index * fanout);
        for (IdType j = 0; j < fanout; ++j) {
          IdType selected_j = curand(&local_state) % (len -j) + j;
          assert(selected_j >= j && selected_j < len);
          IdType tmp_idx_j = 0;
          if (selected_j < fanout) {
            tmp_idx_j = selected_j;
          }
          else {
            IdType pos = _FindPos(hash_table, selected_j);
            if (hash_table[pos].key == Constant::kEmptyKey) {
              hash_table[pos].key = selected_j;
              hash_table[pos].value = selected_j;
            }
            tmp_idx_j = pos;
          }
          // swap
          IdType old_value = hash_table[tmp_idx_j].value;
          hash_table[tmp_idx_j].value = hash_table[j].value;
          hash_table[j].value = old_value;
          tmp_idx[j] = tmp_idx_j;
        }
        for (IdType j =0; j < fanout; ++j) {
          IdType tmp_idx_j = tmp_idx[j];
          IdType indices_pos = hash_table[j].value;
          assert(indices_pos < len);
          tmp_dst[index * fanout + j] = indices[off + indices_pos];
          tmp_src[index * fanout + j] = rid;
          // reset the hash_table
          if (tmp_idx_j >= fanout) {
            hash_table[tmp_idx_j].key = Constant::kEmptyKey;
            hash_table[tmp_idx_j].value = Constant::kEmptyKey;
          }
          hash_table[j].key = Constant::kEmptyKey;
          hash_table[j].value = Constant::kEmptyKey;
        }
      }
    }
  }

  random_states[i] = local_state;
}

#else
#error SHOULD DEFINE MACRO ORIGIN_KHOP2 / KHOP2_REVERSION_NO_MODIFIED
#endif

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_edge(IdType *edge_src, size_t *item_prefix,
                           const size_t num_input, const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (edge_src[index * fanout + j] != Constant::kEmptyKey) {
          ++count;
        }
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_edge(const IdType *tmp_src, const IdType *tmp_dst,
                             IdType *out_src, IdType *out_dst, size_t *num_out,
                             const size_t *item_prefix, const size_t num_input,
                             const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;

  constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const size_t offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<size_t> prefix_op(0);

  // count successful placements
  for (size_t i = 0; i < VALS_PER_THREAD; ++i) {
    const size_t index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    size_t item_per_thread = 0;
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (tmp_src[index * fanout + j] != Constant::kEmptyKey) {
          item_per_thread++;
        }
      }
    }

    size_t item_prefix_per_thread = item_per_thread;
    BlockScan(temp_space)
        .ExclusiveSum(item_prefix_per_thread, item_prefix_per_thread,
                      prefix_op);
    __syncthreads();

    for (size_t j = 0; j < item_per_thread; j++) {
      out_src[offset + item_prefix_per_thread + j] =
          tmp_src[index * fanout + j];
      out_dst[offset + item_prefix_per_thread + j] =
          tmp_dst[index * fanout + j];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_out = item_prefix[gridDim.x];
  }
}

}  // namespace

void GPUSampleKHop2(const IdType *indptr, IdType *indices,
                    const IdType *input, const size_t num_input,
                    const size_t fanout, IdType *out_src, IdType *out_dst,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    GPURandomStates *random_states, uint64_t task_key) {
  LOG(DEBUG) << "GPUSample: begin with num_input " << num_input
             << " and fanout " << fanout;
  Timer t0;
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  IdType *tmp_src = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  IdType *tmp_dst = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  LOG(DEBUG) << "GPUSample: cuda tmp_src malloc "
             << ToReadableSize(num_input * fanout * sizeof(IdType));
  LOG(DEBUG) << "GPUSample: cuda tmp_dst malloc "
             << ToReadableSize(num_input * fanout * sizeof(IdType));

  sample_khop2<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          indptr, indices, input, num_input, fanout, tmp_src, tmp_dst,
          random_states->GetStates(), random_states->NumStates());
  sampler_device->StreamSync(ctx, stream);
  double sample_time = t0.Passed();

  Timer t1;
  size_t *item_prefix = static_cast<size_t *>(
      sampler_device->AllocWorkspace(ctx, sizeof(size_t) * 2 * (grid.x + 1)));
  size_t *const item_prefix_out = &item_prefix[grid.x + 1];
  LOG(DEBUG) << "GPUSample: cuda item_prefix malloc "
             << ToReadableSize(sizeof(size_t) * 2 * (grid.x + 1));

  count_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(tmp_src, item_prefix, num_input, fanout);
  sampler_device->StreamSync(ctx, stream);
  double count_edge_time = t1.Passed();

  Timer t2;
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
      static_cast<size_t *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(ctx, stream);

  void *workspace = sampler_device->AllocWorkspace(ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix_out, grid.x + 1,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);
  LOG(DEBUG) << "GPUSample: cuda workspace malloc "
             << ToReadableSize(workspace_bytes);

  compact_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(tmp_src, tmp_dst, out_src, out_dst,
                                      num_out, item_prefix_out, num_input, fanout);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t2.Passed();

  sampler_device->FreeWorkspace(ctx, workspace);
  sampler_device->FreeWorkspace(ctx, item_prefix);
  sampler_device->FreeWorkspace(ctx, tmp_src);
  sampler_device->FreeWorkspace(ctx, tmp_dst);

  LOG(DEBUG) << "_debug sample time (key " << task_key << ") " << sample_time;
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopSampleCooTime, sample_time);
  Profiler::Get().LogEpochAdd(task_key, kLogEpochSampleCooTime, sample_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopSampleCountEdgeTime,
                             count_edge_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopSampleCompactEdgesTime,
                             compact_edge_time);

  LOG(DEBUG) << "GPUSample: succeed ";
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
