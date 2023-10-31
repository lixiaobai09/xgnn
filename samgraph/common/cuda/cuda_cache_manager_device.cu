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

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cuda_cache_manager.h"
#include "cuda_utils.h"
#include "ics22_song_dist_feature.h"
#include "../profiler.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_miss_cache(const IdType *hashtable, const IdType *nodes,
                                 const size_t num_nodes, IdType *miss_counts,
                                 IdType *cache_counts) {
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;

  IdType miss_count = 0;
  IdType cache_count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_nodes) {
      if (hashtable[nodes[index]] == Constant::kEmptyKey) {
        miss_count++;
      } else {
        cache_count++;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_miss_space;
  __shared__ typename BlockReduce::TempStorage temp_cache_space;

  miss_count = BlockReduce(temp_miss_space).Sum(miss_count);
  cache_count = BlockReduce(temp_cache_space).Sum(cache_count);

  if (threadIdx.x == 0) {
    miss_counts[blockIdx.x] = miss_count;
    cache_counts[blockIdx.x] = cache_count;
    if (blockIdx.x == 0) {
      miss_counts[gridDim.x] = 0;
      cache_counts[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void get_miss_index(const IdType *hashtable, const IdType *nodes,
                               const size_t num_nodes,
                               IdType *output_miss_dst_index,
                               IdType *output_miss_src_index,
                               const IdType *miss_counts_prefix) {
  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  __shared__ typename BlockScan::TempStorage temp_space;
  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  const IdType offset = miss_counts_prefix[blockIdx.x];

  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    if (index < num_nodes && hashtable[nodes[index]] == Constant::kEmptyKey) {
      flag = 1;
    } else {
      flag = 0;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (index < num_nodes && hashtable[nodes[index]] == Constant::kEmptyKey) {
      const IdType pos = offset + flag;
      assert(pos < num_nodes);
      // new node ID in subgraph
      output_miss_dst_index[pos] = index;
      // old node ID in original graph
      output_miss_src_index[pos] = nodes[index];
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("miss count %u, %u\n", miss_counts_prefix[gridDim.x],
  //          miss_counts_prefix[gridDim.x - 1]);
  // }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE, bool ORIGIN_ICS22_SOLVER = false>
__global__ void get_cache_index(const IdType *hashtable, const IdType *nodes,
                                const size_t num_nodes,
                                IdType *output_cache_dst_index,
                                IdType *output_cache_src_index,
                                const IdType *cache_counts_prefix) {
  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  __shared__ typename BlockScan::TempStorage temp_space;
  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  const IdType offset = cache_counts_prefix[blockIdx.x];

  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    if (index < num_nodes && hashtable[nodes[index]] != Constant::kEmptyKey) {
      flag = 1;
    } else {
      flag = 0;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (index < num_nodes && hashtable[nodes[index]] != Constant::kEmptyKey) {
      const IdType pos = offset + flag;
      // new node ID in subgraph
      output_cache_dst_index[pos] = index;
      // old node ID in original graph
      if (ORIGIN_ICS22_SOLVER == false) {
        output_cache_src_index[pos] = hashtable[nodes[index]];
      } else {
        output_cache_src_index[pos] = nodes[index];
      }
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("cache count %u, %u\n", cache_counts_prefix[gridDim.x],
  //          cache_counts_prefix[gridDim.x - 1]);
  // }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_local_cache(const IdType *cache_src_index, const IdType num_cache, 
                                  IdType *prefix_count,
                                  const long local_part_map,
                                  const size_t num_part,
                                  const IdType compact_bitwidth) {
  size_t start = blockIdx.x * TILE_SIZE;
  size_t end = start + TILE_SIZE;

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;

  IdType local = 0;

#pragma unroll
  for (size_t i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
    if (i < num_cache) {
      int part_id;
      if (compact_bitwidth == 0) {
        part_id = (cache_src_index[i] % num_part);
      } else {
        part_id = (cache_src_index[i] >> compact_bitwidth);
      }
      if (((1 << part_id) & local_part_map) != 0) {
        local++;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage tmp_space;
  local = BlockReduce(tmp_space).Sum(local);
  if (threadIdx.x == 0) {
    prefix_count[blockIdx.x] = local;
    if (blockIdx.x == 0) {
      prefix_count[gridDim.x] = 0;
    }
  }
}

template <typename T>
__global__ void combine_miss_data(void *output, const void *miss,
                                  const IdType *miss_dst_index,
                                  const size_t num_miss, const size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *miss_data = reinterpret_cast<const T *>(miss);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  /** SXN: why need a loop?*/
  /** SXN: ans: this loop is not necessary*/
  while (i < num_miss) {
    size_t col = threadIdx.x;
    const size_t dst_idx = miss_dst_index[i];
    while (col < dim) {
      output_data[dst_idx * dim + col] = miss_data[i * dim + col];
      col += blockDim.x;
    }

    i += stride;
  }
}

template <typename T>
__global__ void extract_miss_data(void *output, const void* src_feat,
                                  const IdType *miss_src_index, const IdType *miss_dst_index, 
                                  const size_t num_miss, size_t dim, size_t mask) {
  auto output_data = reinterpret_cast<T *>(output);
  auto feat_data = reinterpret_cast<const T *>(src_feat);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;
  while (i < num_miss) {
    size_t col = threadIdx.x;
    const size_t src_idx = miss_src_index[i];
    const size_t dst_idx = miss_dst_index[i];
    while (col < dim) {
      output_data[dst_idx * dim + col] = feat_data[(src_idx * dim + col) & mask];
      col += blockDim.x;
    }
    i += stride;
  }
}

template <typename T>
__global__ void combine_cache_data(void *output, const IdType *cache_src_index,
                                   const IdType *cache_dst_index,
                                   const size_t num_cache, const void *cache,
                                   size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *cache_data = reinterpret_cast<const T *>(cache);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_cache) {
    size_t col = threadIdx.x;
    const size_t src_idx = cache_src_index[i];
    const size_t dst_idx = cache_dst_index[i];
    while (col < dim) {
      output_data[dst_idx * dim + col] = cache_data[src_idx * dim + col];
      col += blockDim.x;
    }
    i += stride;
  }
}

template <typename T, typename DeviceFeatureType>
__global__ void combine_cache_data_for_partition(void *output,
                                   const IdType *cache_src_index,
                                   const IdType *cache_dst_index,
                                   const size_t num_cache,
                                   DeviceFeatureType cache,
                                   size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  
  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_cache) {
    size_t col = threadIdx.x;
    const size_t src_idx = cache_src_index[i];
    const size_t dst_idx = cache_dst_index[i];
    while (col < dim) {
      output_data[dst_idx * dim + col] = cache.template Get<T>(src_idx, col);
      col += blockDim.x;
    }
    i += stride;
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void hashtable_insert_nodes(const IdType *const cached_nodes,
                                       const size_t num_cached_nodes,
                                       IdType* gpu_hashtable) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_cached_nodes) {
      gpu_hashtable[cached_nodes[index]] = index;
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void hashtable_reset_nodes(const IdType *const cached_nodes,
                                       const size_t num_cached_nodes,
                                       IdType* gpu_hashtable) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_cached_nodes) {
      gpu_hashtable[cached_nodes[index]] = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void hashtable_reset_nodes(const size_t num_total_nodes,
                                      IdType* gpu_hashtable) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_total_nodes) {
      gpu_hashtable[index] = Constant::kEmptyKey;
    }
  }
}

}  // namespace

void GPUCacheManager::GetMissCacheIndex(
    IdType *output_miss_src_index, IdType *output_miss_dst_index,
    size_t *num_output_miss, IdType *output_cache_src_index,
    IdType *output_cache_dst_index, size_t *num_output_cache,
    const IdType *nodes, const size_t num_nodes, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto sampler_device = Device::Get(_sampler_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  sampler_device->SetDevice(_sampler_ctx);

  IdType *miss_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));
  IdType *cache_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));
  IdType *exclusive_sum_tmp = 
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));

  // LOG(DEBUG) << "GetMissCacheIndex num nodes " << num_nodes;

  CUDA_CALL(cudaSetDevice(_sampler_ctx.device_id));
  count_miss_cache<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(_sampler_gpu_hashtable, nodes, num_nodes,
                                      miss_prefix_counts, cache_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);

  void *workspace =
      sampler_device->AllocWorkspace(_sampler_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, miss_prefix_counts, exclusive_sum_tmp,
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);
  std::swap(miss_prefix_counts, exclusive_sum_tmp);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, cache_prefix_counts, exclusive_sum_tmp,
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);
  std::swap(cache_prefix_counts, exclusive_sum_tmp);

  get_miss_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _sampler_gpu_hashtable, nodes, num_nodes, output_miss_dst_index,
          output_miss_src_index, miss_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  if (RunConfig::use_ics22_song_solver
      && (RunConfig::ics22_compact_mode == false)) {
    get_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize, true>
        <<<grid, block, 0, cu_stream>>>(
            _sampler_gpu_hashtable, nodes, num_nodes, output_cache_dst_index,
            output_cache_src_index, cache_prefix_counts);
  } else {
    get_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, cu_stream>>>(
            _sampler_gpu_hashtable, nodes, num_nodes, output_cache_dst_index,
            output_cache_src_index, cache_prefix_counts);
  }
  sampler_device->StreamSync(_sampler_ctx, stream);

  IdType num_miss;
  IdType num_cache;
  sampler_device->CopyDataFromTo(miss_prefix_counts + grid.x, 0, &num_miss, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->CopyDataFromTo(cache_prefix_counts + grid.x, 0, &num_cache, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->StreamSync(_sampler_ctx, stream);

  *num_output_miss = num_miss;
  *num_output_cache = num_cache;

  sampler_device->FreeWorkspace(_sampler_ctx, workspace);
  sampler_device->FreeWorkspace(_sampler_ctx, exclusive_sum_tmp);
  sampler_device->FreeWorkspace(_sampler_ctx, cache_prefix_counts);
  sampler_device->FreeWorkspace(_sampler_ctx, miss_prefix_counts);
}

void GPUCacheManager::CountLocalCache(size_t task_key, 
                                      const IdType *cache_src_index, 
                                      const size_t num_cache, const size_t num_nodes,
                                      StreamHandle stream) {
  CHECK(RunConfig::run_arch == kArch6
        && (RunConfig::part_cache
            || (RunConfig::use_ics22_song_solver
                && RunConfig::ics22_compact_mode)));
  auto sampler_device = Device::Get(_sampler_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  const dim3 grid(RoundUpDiv((size_t)num_cache, Constant::kCudaTileSize));
  const dim3 block(Constant::kCudaBlockSize);

  IdType *local_cache_counts = sampler_device->AllocArray<IdType>(_sampler_ctx, grid.x + 1);
  IdType *local_cache_counts_prefix_sum = sampler_device->AllocArray<IdType>(_sampler_ctx, grid.x + 1);

  // check long type can be used for device map(local_part_map)
  static_assert(sizeof(long) * 8 >= kMaxDevice);

  long local_part_map = 0;
  if (RunConfig::part_cache) {
    // get the local_part_map and num_part
    DistGraph::GroupConfig group_cfg = _dist_graph->GetGroupConfig(
        _trainer_ctx.device_id);
    size_t num_part = group_cfg.ctx_group.size();
    for (IdType part_id : group_cfg.part_ids) {
      local_part_map = (local_part_map | (1l << part_id));
    }
  } else if (RunConfig::use_ics22_song_solver
             && RunConfig::ics22_compact_mode) {
    IdType device_id = _sampler_ctx.device_id;
    local_part_map = (1l << device_id);
  }
  // get the local_part_map and num_part
  DistGraph::GroupConfig group_cfg = _dist_graph->GetGroupConfig(
      _trainer_ctx.device_id);
  size_t num_part = group_cfg.ctx_group.size();
  for (IdType part_id : group_cfg.part_ids) {
    local_part_map = (local_part_map | (1l << part_id));
  }
  if (RunConfig::part_cache) {
    count_local_cache<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, cu_stream>>>(
          cache_src_index, num_cache, local_cache_counts,
          local_part_map, num_part, 0);
  } else {
    count_local_cache<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, cu_stream>>>(
          cache_src_index, num_cache, local_cache_counts,
          local_part_map, num_part,
          (sizeof(IdType) * 8 - RunConfig::ics22_compact_bitwidth));
  }
  sampler_device->StreamSync(_sampler_ctx, stream);

  size_t workspace_sz;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_sz, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);
  void *workspace = sampler_device->AllocWorkspace(_sampler_ctx, workspace_sz);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_sz, 
      local_cache_counts, local_cache_counts_prefix_sum, 
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);
  
  IdType num_local_cache;
  sampler_device->CopyDataFromTo(
      local_cache_counts_prefix_sum + grid.x, 0,
      &num_local_cache, 0, sizeof(IdType),
      _sampler_ctx, CPU(), stream);
  sampler_device->StreamSync(_sampler_ctx, stream);
  
  sampler_device->FreeWorkspace(_sampler_ctx, local_cache_counts);
  sampler_device->FreeWorkspace(_sampler_ctx, local_cache_counts_prefix_sum);
  sampler_device->FreeWorkspace(_sampler_ctx, workspace);

  Profiler::Get().LogEpochAdd(task_key, kLogEpochLocalCacheBytes,
      GetTensorBytes(_dtype, {num_local_cache, _dim}));
}

void GPUCacheManager::CombineMissData(void *output, const void *miss,
                                      const IdType *miss_dst_index,
                                      const size_t num_miss,
                                      StreamHandle stream) {
  LOG(DEBUG) << "GPUCacheManager::CombineMissData():  num_miss " << num_miss;
  if (num_miss == 0) return;

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_miss, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      combine_miss_data<float><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF64:
      combine_miss_data<double><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF16:
      combine_miss_data<short><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kU8:
      combine_miss_data<uint8_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI32:
      combine_miss_data<int32_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI64:
      combine_miss_data<int64_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}

void GPUCacheManager::GPUExtractMissData(void *output, const IdType *miss_src_index,
                                         const IdType *miss_dst_index, const size_t num_miss,
                                         StreamHandle stream) {
  LOG(DEBUG) << "GPUCacheManager::GPUExtractMissData(): num_miss " << num_miss;
  if (num_miss == 0) return;

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while(static_cast<size_t> (block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_miss, static_cast<size_t>(block.y)));

  size_t mask = static_cast<size_t>(-1);
  if (RunConfig::option_empty_feat) {
    mask = (1ULL << RunConfig::option_empty_feat) - 1;
  }

  switch (_dtype)
  {
  case kF32:
    extract_miss_data<float><<<grid, block, 0, cu_stream>>>(
        output, _cpu_src_data, miss_src_index, miss_dst_index, num_miss, _dim, mask);
    break;
  case kF64:
    extract_miss_data<double><<<grid, block, 0, cu_stream>>>(
        output, _cpu_src_data, miss_src_index, miss_dst_index, num_miss, _dim, mask);
    break;
  case kF16:
    extract_miss_data<short><<<grid, block, 0, cu_stream>>>(
        output, _cpu_src_data, miss_src_index, miss_dst_index, num_miss, _dim, mask);
    break;
  case kU8:
    extract_miss_data<uint8_t><<<grid, block, 0, cu_stream>>>(
        output, _cpu_src_data, miss_src_index, miss_dst_index, num_miss, _dim, mask);
    break;
  case kI32:
    extract_miss_data<int32_t><<<grid, block, 0, cu_stream>>>(
        output, _cpu_src_data, miss_src_index, miss_dst_index, num_miss, _dim, mask);
    break;
  case kI64:
    extract_miss_data<int64_t><<<grid, block, 0, cu_stream>>>(
        output, _cpu_src_data, miss_src_index, miss_dst_index, num_miss, _dim, mask);
    break;
  default:
    CHECK(0);
  }
  device->StreamSync(_trainer_ctx, stream);
  LOG(DEBUG) << "GPUCacheManager::GPUExtractMissData(): after extract_miss_data";
}

template <typename T>
inline void dispatch_combine_cache(dim3 grid, dim3 block, cudaStream_t stream,
                          void *output, const IdType *cache_src_index,
                          const IdType *cache_dst_index, const size_t num_cache,
                          size_t dim,
                          const void *norm_cache, const DistGraph *dist_graph) {
  if (RunConfig::run_arch == kArch6 && (RunConfig::part_cache
                                        || RunConfig::use_ics22_song_solver)) {
    if (RunConfig::use_ics22_song_solver) {
      auto ics22_dist_graph_ptr
             = dynamic_cast<const ICS22SongDistGraph*>(dist_graph);
      CHECK(ics22_dist_graph_ptr != nullptr);
      combine_cache_data_for_partition<T, DeviceICS22SongDistFeature>
        <<<grid, block, 0, stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          ics22_dist_graph_ptr->DeviceFeatureHandle(), dim);
    } else {
      combine_cache_data_for_partition<T, DeviceDistFeature>
        <<<grid, block, 0, stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          dist_graph->DeviceFeatureHandle(), dim);
    }
  } else {
    combine_cache_data<T><<<grid, block, 0, stream>>>(
      output, cache_src_index, cache_dst_index, num_cache,
      norm_cache, dim);
  }
}

void GPUCacheManager::CombineCacheData(void *output,
                                       const IdType *cache_src_index,
                                       const IdType *cache_dst_index,
                                       const size_t num_cache,
                                       StreamHandle stream) {
  CHECK_LE(num_cache, _num_cached_nodes);
  if (num_cache == 0) return;

  LOG(DEBUG) << "CombineCacheData";

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_cache, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      dispatch_combine_cache<float>(grid, block, cu_stream, 
          output, cache_src_index, cache_dst_index, num_cache, _dim, 
          _trainer_cache_data, _dist_graph);
      // combine_cache_data<float><<<grid, block, 0, cu_stream>>>(
      //     output, cache_src_index, cache_dst_index, num_cache,
      //     _trainer_cache_data, _dim);
      break;
    case kF64:
      dispatch_combine_cache<double>(grid, block, cu_stream, 
          output, cache_src_index, cache_dst_index, num_cache, _dim, 
          _trainer_cache_data, _dist_graph);
      // combine_cache_data<double><<<grid, block, 0, cu_stream>>>(
      //     output, cache_src_index, cache_dst_index, num_cache,
      //     _trainer_cache_data, _dim);
      break;
    case kF16:
      dispatch_combine_cache<short>(grid, block, cu_stream, 
          output, cache_src_index, cache_dst_index, num_cache, _dim, 
          _trainer_cache_data, _dist_graph);
      // combine_cache_data<short><<<grid, block, 0, cu_stream>>>(
      //     output, cache_src_index, cache_dst_index, num_cache,
      //     _trainer_cache_data, _dim);
      break;
    case kU8:
      dispatch_combine_cache<uint8_t>(grid, block, cu_stream, 
          output, cache_src_index, cache_dst_index, num_cache, _dim, 
          _trainer_cache_data, _dist_graph);
      // combine_cache_data<uint8_t><<<grid, block, 0, cu_stream>>>(
      //     output, cache_src_index, cache_dst_index, num_cache,
      //     _trainer_cache_data, _dim);
      break;
    case kI32:
      dispatch_combine_cache<int32_t>(grid, block, cu_stream, 
          output, cache_src_index, cache_dst_index, num_cache, _dim, 
          _trainer_cache_data, _dist_graph);
      // combine_cache_data<int32_t><<<grid, block, 0, cu_stream>>>(
      //     output, cache_src_index, cache_dst_index, num_cache,
      //     _trainer_cache_data, _dim);
      break;
    case kI64:
      dispatch_combine_cache<int64_t>(grid, block, cu_stream, 
          output, cache_src_index, cache_dst_index, num_cache, _dim, 
          _trainer_cache_data, _dist_graph);
      // combine_cache_data<int64_t><<<grid, block, 0, cu_stream>>>(
      //     output, cache_src_index, cache_dst_index, num_cache,
      //     _trainer_cache_data, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
  LOG(DEBUG) << "CombineCacheData successfully";
}

void GPUDynamicCacheManager::GetMissCacheIndex(
    IdType *output_miss_src_index, IdType *output_miss_dst_index,
    size_t *num_output_miss, IdType *output_cache_src_index,
    IdType *output_cache_dst_index, size_t *num_output_cache,
    const IdType *nodes, const size_t num_nodes, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto sampler_device = Device::Get(_sampler_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  sampler_device->SetDevice(_sampler_ctx);

  IdType *miss_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));
  IdType *cache_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));
  IdType *exclusive_sum_tmp = 
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));

  // LOG(DEBUG) << "GetMissCacheIndex num nodes " << num_nodes;

  CUDA_CALL(cudaSetDevice(_sampler_ctx.device_id));
  count_miss_cache<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(_sampler_gpu_hashtable, nodes, num_nodes,
                                      miss_prefix_counts, cache_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);

  void *workspace =
      sampler_device->AllocWorkspace(_sampler_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, miss_prefix_counts, exclusive_sum_tmp,
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);
  std::swap(miss_prefix_counts, exclusive_sum_tmp);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, cache_prefix_counts, exclusive_sum_tmp,
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);
  std::swap(cache_prefix_counts, exclusive_sum_tmp);

  get_miss_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _sampler_gpu_hashtable, nodes, num_nodes, output_miss_dst_index,
          output_miss_src_index, miss_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  get_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _sampler_gpu_hashtable, nodes, num_nodes, output_cache_dst_index,
          output_cache_src_index, cache_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  IdType num_miss;
  IdType num_cache;
  sampler_device->CopyDataFromTo(miss_prefix_counts + grid.x, 0, &num_miss, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->CopyDataFromTo(cache_prefix_counts + grid.x, 0, &num_cache, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->StreamSync(_sampler_ctx, stream);

  *num_output_miss = num_miss;
  *num_output_cache = num_cache;

  sampler_device->FreeWorkspace(_sampler_ctx, workspace);
  sampler_device->FreeWorkspace(_sampler_ctx, exclusive_sum_tmp);
  sampler_device->FreeWorkspace(_sampler_ctx, cache_prefix_counts);
  sampler_device->FreeWorkspace(_sampler_ctx, miss_prefix_counts);
}

void GPUDynamicCacheManager::CombineMissData(void *output, const void *miss,
                                             const IdType *miss_dst_index,
                                             const size_t num_miss,
                                             StreamHandle stream) {
  LOG(DEBUG) << "GPUDynamicCacheManager::CombineMissData():  num_miss "
             << num_miss;

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_miss, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      combine_miss_data<float><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF64:
      combine_miss_data<double><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF16:
      combine_miss_data<short><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kU8:
      combine_miss_data<uint8_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI32:
      combine_miss_data<int32_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI64:
      combine_miss_data<int64_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}

void GPUDynamicCacheManager::CombineCacheData(void *output,
                                              const IdType *cache_src_index,
                                              const IdType *cache_dst_index,
                                              const size_t num_cache,
                                              StreamHandle stream) {
  if (_trainer_cache_data == nullptr) {
    CHECK_EQ(num_cache, 0);
    return;
  }
  CHECK_LE(num_cache, _trainer_cache_data->Shape()[0]);
  const void * train_cache_data = _trainer_cache_data->Data();

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_cache, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      combine_cache_data<float><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kF64:
      combine_cache_data<double><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kF16:
      combine_cache_data<short><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kU8:
      combine_cache_data<uint8_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kI32:
      combine_cache_data<int32_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kI64:
      combine_cache_data<int64_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}

/**
 * @brief 
 * 
 * @param nodes must be in cpu
 * @param features must be in train gpu
 */
 void GPUDynamicCacheManager::ReplaceCacheGPU(TensorPtr nodes, TensorPtr features, StreamHandle stream) {
  Timer t;

  // _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});

  // auto cpu_device = Device::Get(CPU());
  auto sampler_gpu_device = Device::Get(_sampler_ctx);
  // auto trainer_gpu_device = Device::Get(_trainer_ctx);

  // IdType *tmp_cpu_hashtable = static_cast<IdType *>(
  //     cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
  // void *tmp_cpu_data = cpu_device->AllocDataSpace(CPU(), _cache_nbytes);

  auto cu_stream = static_cast<cudaStream_t>(stream);

  CHECK_EQ(nodes->Ctx().device_type, _sampler_ctx.device_type);
  CHECK_EQ(nodes->Ctx().device_id, _sampler_ctx.device_id);
  CHECK_EQ(features->Ctx().device_type, _trainer_ctx.device_type);
  CHECK_EQ(features->Ctx().device_id, _trainer_ctx.device_id);

  CHECK_EQ(features->Shape()[0], nodes->Shape()[0]);


  sampler_gpu_device->SetDevice(_sampler_ctx);
  // 1. Initialize the gpu hashtable
  if (_cached_nodes != nullptr) {
    const IdType* old_cached_nodes = static_cast<const IdType*>(_cached_nodes->Data());
    size_t num_old_cached_nodes = _cached_nodes->Shape()[0];
    const size_t old_num_tiles = RoundUpDiv(num_old_cached_nodes, Constant::kCudaTileSize);
    const dim3 old_grid(old_num_tiles);
    const dim3 old_block(Constant::kCudaBlockSize);
    hashtable_reset_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<old_grid, old_block, 0, cu_stream>>>(
            old_cached_nodes, num_old_cached_nodes, _sampler_gpu_hashtable);
    sampler_gpu_device->StreamSync(_sampler_ctx, stream);
  }
  
  // sampler_gpu_device->CopyDataFromTo(
  //   _cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
  //   sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx, stream);
  // sampler_gpu_device->StreamSync(_sampler_ctx, stream);


  // 2. Populate the cpu hashtable
  const IdType* new_cached_nodes = static_cast<const IdType*>(nodes->Data());
  size_t num_new_cached_nodes = nodes->Shape()[0];
  const size_t num_tiles = RoundUpDiv(num_new_cached_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  hashtable_insert_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(new_cached_nodes, num_new_cached_nodes, _sampler_gpu_hashtable);
  sampler_gpu_device->StreamSync(_sampler_ctx, stream);

  _trainer_cache_data = features;
  _cached_nodes = nodes;

  // 3. Populate the cache in cpu memory
  // cpu::CPUExtract(tmp_cpu_data, _cpu_src_data, nodes, _num_cached_nodes, _dim,
  //                 _dtype);

  // 4. Copy the cache from the cpu memory to gpu memory
  // sampler_gpu_device->CopyDataFromTo(
  //     _cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
  //     sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx);
  // trainer_gpu_device->CopyDataFromTo(tmp_cpu_data, 0, _trainer_cache_data, 0,
  //                                    _cache_nbytes, CPU(), _trainer_ctx);

  // 5. Free the cpu tmp cache data
  // cpu_device->FreeDataSpace(CPU(), tmp_cpu_hashtable);
  // cpu_device->FreeDataSpace(CPU(), tmp_cpu_data);

  LOG(INFO) << "GPU dynamic cache "
            << features->Shape()[0] << " / " << _num_nodes << " nodes ( "
            << ToPercentage(features->Shape()[0] / (float)_num_nodes) << " | "
            << ToReadableSize(features->NumBytes()) << " | " << t.Passed()
            << " secs )";
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
