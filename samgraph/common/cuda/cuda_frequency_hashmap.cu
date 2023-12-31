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

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_frequency_hashmap.h"
#include "cuda_utils.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

size_t TableSize(const size_t num, const size_t scale) {
  /**
   * Calculate the number of buckets in the hashtable. To guarantee we can
   * fill the hashtable in the worst case, we must use a number of buckets which
   * is a power of two.
   * https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
   */
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

class MutableDeviceFrequencyHashmap : public DeviceFrequencyHashmap {
 public:
  typedef typename DeviceFrequencyHashmap::NodeBucket *NodeIterator;
  typedef typename DeviceFrequencyHashmap::EdgeBucket *EdgeIterator;

  explicit MutableDeviceFrequencyHashmap(FrequencyHashmap *const host_map)
      : DeviceFrequencyHashmap(host_map->DeviceHandle()) {}

  inline __device__ NodeIterator SearchNode(const IdType id) {
    return GetMutableNode(id);
  }

  inline __device__ EdgeIterator SearchEdge(const IdType node_idx,
                                            const IdType dst) {
    const IdType pos = SearchEdgeForPosition(node_idx, dst);
    return GetMutableEdge(pos);
  }

  inline __device__ bool AttemptInsertEdgeAt(const IdType pos, const IdType src,
                                             const IdType dst,
                                             const IdType index) {
    EdgeIterator edge_iter = GetMutableEdge(pos);
    const IdType key = atomicCAS(&edge_iter->key, Constant::kEmptyKey, dst);
    if (key == Constant::kEmptyKey || key == dst) {
      atomicAdd(&edge_iter->count, 1U);
      /** SXN: remove atomic by checking swapped out key */
      if (key == Constant::kEmptyKey) {
        edge_iter->index = index;
        NodeIterator node_iter = SearchNode(PosToNodeIdx(pos));
        atomicAdd(node_iter, 1U);
      }
      return true;
    } else {
      return false;
    }
  }

  inline __device__ EdgeIterator InsertEdge(const IdType node_idx,
                                            const IdType src, const IdType dst,
                                            const IdType index) {
    IdType start_off = node_idx * _per_node_etable_size;
    IdType pos = EdgeHash(dst);

    IdType delta = 1;
    while (!AttemptInsertEdgeAt(start_off + pos, src, dst, index)) {
      pos = EdgeHash(pos + delta);
      delta += 1;
    }

    return GetMutableEdge(start_off + pos);
  }

  inline __device__ NodeIterator GetMutableNode(const IdType pos) {
    assert(pos < _ntable_size);
    return const_cast<NodeIterator>(_node_table + pos);
  }

  inline __device__ EdgeIterator GetMutableEdge(const IdType pos) {
    assert(pos < _etable_size);
    return const_cast<EdgeIterator>(_edge_table + pos);
  }

  inline __device__ IdType GetRelativePos(const EdgeIterator iter) {
    return iter - _edge_table;
  }
};

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void init_node_table(MutableDeviceFrequencyHashmap table,
                                const size_t num_bucket) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using NodeIterator = typename MutableDeviceFrequencyHashmap::NodeIterator;

#pragma unroll
  for (IdType index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_bucket) {
      NodeIterator node_iter = table.GetMutableNode(index);
      *node_iter = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void init_edge_table(MutableDeviceFrequencyHashmap table,
                                const size_t num_bucket) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (IdType index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_bucket) {
      EdgeIterator edge_iter = table.GetMutableEdge(index);
      edge_iter->key = Constant::kEmptyKey;
      edge_iter->count = 0;
      edge_iter->index = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void init_unique_range(IdType *_unique_range,
                                  const size_t unique_list_size) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (IdType index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < unique_list_size) {
      _unique_range[index] = index;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_node_table(MutableDeviceFrequencyHashmap table,
                                 const IdType *nodes, const size_t num_nodes) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using NodeIterator = typename MutableDeviceFrequencyHashmap::NodeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_nodes) {
      NodeIterator node_iter = table.SearchNode(index);
      *node_iter = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_edge_table(MutableDeviceFrequencyHashmap table,
                                 IdType *unique_node_idx, IdType *unique_dst,
                                 const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType node_idx = unique_node_idx[index];
      IdType dst = unique_dst[index];
      EdgeIterator edge_iter = table.SearchEdge(node_idx, dst);
      edge_iter->key = Constant::kEmptyKey;
      edge_iter->count = 0;
      edge_iter->index = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_edge_table_revised(MutableDeviceFrequencyHashmap table,
                                 IdType *unique_pos, IdType *unique_dst,
                                 const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType dst = unique_dst[index];
      EdgeIterator edge_iter = table.GetMutableEdge(unique_pos[index]);
      assert(edge_iter->key == dst);
      edge_iter->key = Constant::kEmptyKey;
      edge_iter->count = 0;
      edge_iter->index = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_frequency(const IdType *input_src,
                                const IdType *input_dst,
                                const size_t num_input_edge,
                                const size_t edges_per_node,
                                IdType *item_prefix,
                                MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      IdType node_idx = index / edges_per_node;
      EdgeIterator edge_iter =
          table.InsertEdge(node_idx, input_src[index], input_dst[index], index);
      if (edge_iter->index == index) {
        ++count;
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
__global__ void count_frequency_revised(
                                IdType *input_src,
                                IdType *input_dst,
                                const size_t num_input_edge,
                                const size_t edges_per_node,
                                IdType *item_prefix,
                                MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      IdType node_idx = index / edges_per_node;
      EdgeIterator edge_iter =
          table.InsertEdge(node_idx, input_src[index], input_dst[index], index);
      input_src[index] = Constant::kEmptyKey;
      if (edge_iter->index == index) {
        input_src[index] = table.GetRelativePos(edge_iter);
        ++count;
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
__global__ void generate_unique_edges(
    const IdType *input_src, const IdType *input_dst,
    const size_t num_input_edge, IdType *item_prefix, IdType *unique_node_idx,
    IdType *unique_src, IdType *unique_dst, IdType *unique_count,
    size_t *num_unique, const size_t edges_per_node,
    MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    IdType node_idx = index / edges_per_node;
    FlagType flag;
    EdgeBucket *bucket;
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      bucket = table.SearchEdge(node_idx, input_dst[index]);
      flag = (bucket->index == index);
    } else {
      flag = 0;
    }

    if (!flag) {
      bucket = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (bucket) {
      const IdType pos = offset + flag;
      unique_node_idx[pos] = node_idx;
      unique_src[pos] = input_src[index];
      unique_dst[pos] = input_dst[index];
      unique_count[pos] = bucket->count;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique = item_prefix[gridDim.x];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_unique_edges_pos(
    IdType *input_src,
    const IdType *input_nodes,
    const size_t num_input_node,
    const size_t num_input_edge, IdType *item_prefix,
    Id64Type * unique_combination_key,
    IdType *unique_edge_pos, const size_t edges_per_node,
    MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    IdType node_idx = index / edges_per_node;
    FlagType flag = 0;
    EdgeBucket *bucket = nullptr;
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      /** SXN: optimize: input dst can be modified to location in hash_table,
       * thus no need to search */
      bucket = table.GetMutableEdge(input_src[index]);
      flag = 1;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (bucket) {
      const IdType pos = offset + flag;
      unique_edge_pos[pos] = table.GetRelativePos(bucket);
      unique_combination_key[pos] = 
          (Id64Type)((num_input_node - node_idx) << 32) | 
          ((Id64Type)bucket->count); 
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void construct_unique_edge_list(
    const Id64Type *unique_combination_key,
    const IdType *unique_pos,
    IdType *unique_node_idx,
    IdType *unique_dst,
    const size_t num_unique,
    MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      unique_node_idx[index] = table.PosToNodeIdx(unique_pos[index]);
      unique_dst[index] = table.GetMutableEdge(unique_pos[index])->key;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reorder_unique(
    const IdType *unique_src, const IdType *unique_idx,
    const IdType *tmp_unique_node_idx, const IdType *tmp_unique_dst,
    const IdType *tmp_unique_frequency, IdType *unique_node_idx,
    IdType *unique_dst, IdType *unique_frequency,
    Id64Type *unique_combination_key, const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType origin_idx = unique_idx[index];
      unique_node_idx[index] = tmp_unique_node_idx[origin_idx];
      unique_dst[index] = tmp_unique_dst[origin_idx];
      unique_frequency[index] = tmp_unique_frequency[origin_idx];
      unique_combination_key[index] =
          (((Id64Type)unique_src[index]) << 32) |
          ((Id64Type)tmp_unique_frequency[origin_idx]);
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_num_edge(const IdType *, const size_t num_nodes,
                                  const size_t K, IdType *,
                                  IdType *num_output_prefix,
                                  DeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using NodeBucket = typename DeviceFrequencyHashmap::NodeBucket;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_nodes) {
      const NodeBucket &count = *table.SearchNode(index);
      num_output_prefix[index] = count > K ? K : count;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    num_output_prefix[num_nodes] = 0;
  }
}

__global__ void compact_output_revised(
    const IdType* input_nodes,
    const Id64Type *unique_combination_key,
    const IdType *unique_dst,
    const size_t num_nodes, const size_t K,
    const IdType *num_unique_prefix,
    const IdType *num_output_prefix,
    IdType *output_src, IdType *output_dst,
    IdType *output_data, size_t *num_output) {
  size_t i = blockIdx.x * blockDim.y + threadIdx.y;

  /** SXN: this loop `may` be unnecessary */
  if (i < num_nodes) {
    IdType k = threadIdx.x;
    IdType max_output = num_output_prefix[i + 1] - num_output_prefix[i];
    /** SXN: max_output must <= K, ensured in generate_num_edge */
    while (k < max_output) {
      IdType from_off = num_unique_prefix[i] + k;
      IdType to_off = num_output_prefix[i] + k;
      // IdType src_node_idx = num_nodes - (unique_combination_key[from_off] >> 32);
      // assert(src_node_idx == i);
      output_src[to_off] = input_nodes[i];
      output_data[to_off] = (unique_combination_key[from_off]);
      output_dst[to_off] = unique_dst[from_off];

      k += blockDim.x;
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    *num_output = num_output_prefix[num_nodes];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void validate_combine_w_freq(const IdType *unique_src,
                                        IdType *unique_frequency,
                                        Id64Type *unique_combination_key,
                                        const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType original_node_id = unique_src[index];
      Id64Type ck = unique_combination_key[index];
      IdType extracted_node_id = (ck >> 32) & 0x00000000ffffffff;
      IdType extracted_freq = ck & 0x00000000ffffffff;
      assert(extracted_node_id == original_node_id);
      assert(extracted_freq == unique_frequency[index]);
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void validate_combine(const IdType *unique_src,
                                 Id64Type *unique_combination_key,
                                 const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType original_node_id = unique_src[index];
      Id64Type ck = unique_combination_key[index];
      IdType extracted_node_id = (ck >> 32) & 0x00000000ffffffff;
      assert(extracted_node_id == original_node_id);
    }
  }
}

}  // namespace

DeviceFrequencyHashmap::DeviceFrequencyHashmap(
    const NodeBucket *node_table, const EdgeBucket *edge_table,
    const size_t ntable_size, const size_t etable_size,
    const size_t per_node_etable_size)
    : _node_table(node_table),
      _edge_table(edge_table),
      _ntable_size(ntable_size),
      _etable_size(etable_size),
      _per_node_etable_size(per_node_etable_size) {}

DeviceFrequencyHashmap FrequencyHashmap::DeviceHandle() const {
  return DeviceFrequencyHashmap(_node_table, _edge_table, _ntable_size,
                                _etable_size, _per_node_etable_size);
}

FrequencyHashmap::FrequencyHashmap(const size_t max_nodes,
                                   const size_t edges_per_node, Context ctx,
                                   const size_t node_table_scale,
                                   const size_t edge_table_scale)
    : _ctx(ctx),
      _max_nodes(max_nodes),
      _edges_per_node(edges_per_node),
      _ntable_size(max_nodes + 1),
      _etable_size(max_nodes * TableSize(edges_per_node, edge_table_scale)),
      _per_node_etable_size(TableSize(edges_per_node, edge_table_scale)),
      _num_node(0),
      _node_list_size(max_nodes),
      _num_unique(0),
      _unique_list_size(max_nodes * edges_per_node) {
  auto device = Device::Get(_ctx);
  CHECK_EQ(_ctx.device_type, kGPU);

  _node_table = static_cast<NodeBucket *>(
      device->AllocDataSpace(_ctx, sizeof(NodeBucket) * _ntable_size));
  _edge_table = static_cast<EdgeBucket *>(
      device->AllocDataSpace(_ctx, sizeof(EdgeBucket) * _etable_size));

  _unique_node_idx = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size * 2));
  _unique_dst = _unique_node_idx + _unique_list_size;
  _unique_combination_key = static_cast<Id64Type *>(
      device->AllocDataSpace(_ctx, sizeof(Id64Type) * _unique_list_size));

  auto device_table = MutableDeviceFrequencyHashmap(this);
  dim3 grid0(RoundUpDiv(_ntable_size, Constant::kCudaTileSize));
  dim3 grid1(RoundUpDiv(_etable_size, Constant::kCudaTileSize));
  dim3 block0(Constant::kCudaBlockSize);
  dim3 block1(Constant::kCudaBlockSize);

  init_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0>>>(device_table, _ntable_size);
  init_edge_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid1, block1>>>(device_table, _etable_size);

  LOG(INFO) << "FrequencyHashmap init with node table size: " << _ntable_size
            << " and edge table size: " << _etable_size;
}

FrequencyHashmap::~FrequencyHashmap() {
  auto device = Device::Get(_ctx);

  device->FreeDataSpace(_ctx, _node_table);
  device->FreeDataSpace(_ctx, _edge_table);
  device->FreeDataSpace(_ctx, _unique_node_idx);
  device->FreeDataSpace(_ctx, _unique_combination_key);
}

void FrequencyHashmap::GetTopK(
    IdType *input_src, IdType *input_dst,
    const size_t num_input_edge, const IdType *input_nodes,
    const size_t num_input_node, const size_t K, IdType *output_src,
    IdType *output_dst, IdType *output_data, size_t *num_output,
    StreamHandle stream, uint64_t task_key) {
  const size_t num_tiles0 = RoundUpDiv(num_input_node, Constant::kCudaTileSize);
  const size_t num_tiles1 = RoundUpDiv(num_input_edge, Constant::kCudaTileSize);
  const dim3 grid_input_node(num_tiles0);
  const dim3 grid_input_edge(num_tiles1);

  const dim3 block_input_node(Constant::kCudaBlockSize);
  const dim3 block_input_edge(Constant::kCudaBlockSize);
  dim3 block2(Constant::kCudaBlockSize, 1);
  while (static_cast<size_t>(block2.x) >= 2 * K) {
    block2.x /= 2;
    block2.y *= 2;
  }
  dim3 grid2(RoundUpDiv(num_input_node, static_cast<size_t>(block2.y)));

  auto device_table = MutableDeviceFrequencyHashmap(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  size_t workspace_bytes1;
  size_t workspace_bytes2;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes1, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid_input_edge.x + 1, cu_stream));
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes2, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace1 = device->AllocWorkspace(_ctx, workspace_bytes1);
  void *workspace2 = device->AllocWorkspace(_ctx, workspace_bytes2);

  // 1. count frequency of every unique edge and
  //    count unique edges for every node
  Timer t1;
  IdType *num_unique_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * 2 * (grid_input_edge.x + 1)));
  IdType *const num_unique_prefix_out = &num_unique_prefix[grid_input_edge.x + 1];
  count_frequency_revised<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_edge, block_input_edge, 0, cu_stream>>>(input_src, input_dst, num_input_edge,
                                        _edges_per_node, num_unique_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);
  /** pos in device_table is now stored in input_src */
  double step1_time = t1.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 1 finish";

  // 2. count the number of unique edges.
  //    prefix sum the the array
  Timer t2;

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace1, workspace_bytes1,
                                          num_unique_prefix, num_unique_prefix_out,
                                          grid_input_edge.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);
  double step2_time = t2.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 2 finish";

  // 3. get the array of all unique edges' pos in table
  Timer t3;
  device->CopyDataFromTo(&num_unique_prefix_out[grid_input_edge.x], 0, &_num_unique, 0,
                         sizeof(IdType), _ctx, CPU(), stream);
  device->StreamSync(_ctx, stream);
  LOG(DEBUG) << "FrequencyHashmap::Before gettopk step 3,  number of unique is " << _num_unique;
  /** location in edge table */
  /** now we reuse input_dst as pos */
  IdType *tmp_unique_pos = input_dst;
  input_dst = nullptr;
  generate_unique_edges_pos<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_edge, block_input_edge, 0, cu_stream>>>(input_src, input_nodes, num_input_node, num_input_edge,
                                        num_unique_prefix_out, _unique_combination_key, tmp_unique_pos,
                                        _edges_per_node, device_table);
  device->StreamSync(_ctx, stream);
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 3 finish with number of unique "
             << _num_unique;
  double step3_time = t3.Passed();

  // 4. pair-sort unique array using src as key
  //    construct the array of unique dst, unique node idx
  Timer t4;
  /** now we reuse input_src as sort tmp space */
  IdType *alt_val = input_src;
  input_src = nullptr;
  // TDH ASK: why _unique_node_idx not overflow?
  //          ANS _unique_node_idx's space is double
  cub::DoubleBuffer<Id64Type> keys(_unique_combination_key, reinterpret_cast<Id64Type*>(_unique_node_idx));
  cub::DoubleBuffer<IdType>   vals(tmp_unique_pos, alt_val);

  size_t workspace_bytes4;
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      nullptr, workspace_bytes4, keys, vals, _num_unique, 0, sizeof(Id64Type) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace4 = device->AllocWorkspace(_ctx, workspace_bytes4);
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace4, workspace_bytes4, keys, vals, _num_unique, 0, sizeof(Id64Type) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  _unique_combination_key = keys.Current();
  tmp_unique_pos = vals.Current();
  _unique_node_idx = reinterpret_cast<IdType*>(keys.Alternate());
  _unique_dst = _unique_node_idx + _unique_list_size;
  alt_val = vals.Alternate();

  const size_t num_tiles3 = RoundUpDiv(_num_unique, Constant::kCudaTileSize);
  const dim3 grid_uniq_e(num_tiles3);
  const dim3 block_uniq_e(Constant::kCudaBlockSize);
  construct_unique_edge_list<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_uniq_e, block_uniq_e, 0, cu_stream>>>(
          _unique_combination_key, tmp_unique_pos,
          _unique_node_idx, _unique_dst, _num_unique, device_table);
  device->StreamSync(_ctx, stream);
  double step4_time = t4.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 4 finish";


  // 5. get array unique edge number in the order of src nodes.
  //    also count the number of output edges for each nodes.
  //    prefix sum for array of unique edge number.
  Timer t5;
  IdType *num_edge_prefix = _node_table;
  IdType *num_output_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, (num_input_node + 1) * 2 * sizeof(IdType)));
  generate_num_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_node, block_input_node, 0, cu_stream>>>(input_nodes, num_input_node, K,
                                        num_edge_prefix, num_output_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);
  IdType *const num_output_prefix_out = &num_output_prefix[num_input_node + 1];
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_output_prefix, num_output_prefix_out,
                                          num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);
  /** FIX: only the first num_input_node items are used in num_edge_prefix. reset also only rests these.(notice the grid size) */
  IdType *const num_edge_prefix_out = num_output_prefix;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_edge_prefix, num_edge_prefix_out,
                                          num_input_node, cu_stream));
  device->StreamSync(_ctx, stream);

  double step5_time = t5.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 5 finish";

  // 6. copy the result to the output array and set the value of num_output
  Timer t6;
  compact_output_revised<<<grid2, block2, 0, cu_stream>>>(
      input_nodes,
      _unique_combination_key, _unique_dst, num_input_node, K,
      num_edge_prefix_out, num_output_prefix_out, output_src, output_dst, output_data,
      num_output);
  device->StreamSync(_ctx, stream);

  double step6_time = t6.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 6 finish";

  // 7. reset data
  Timer t7;
  reset_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_node, block_input_node, 0, cu_stream>>>(device_table, input_nodes, num_input_node);
  Device::Get(_ctx)->StreamSync(_ctx, stream);

  reset_edge_table_revised<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_uniq_e, block_uniq_e, 0, cu_stream>>>(device_table, tmp_unique_pos,
                                        _unique_dst, _num_unique);
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  double step7_time = t7.Passed();

  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 7 finish";

  _num_unique = 0;

  device->FreeWorkspace(_ctx, num_output_prefix);
  device->FreeWorkspace(_ctx, workspace4);
  device->FreeWorkspace(_ctx, num_unique_prefix);
  device->FreeWorkspace(_ctx, workspace2);
  device->FreeWorkspace(_ctx, workspace1);

  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep1Time,
                             step1_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep2Time,
                             step2_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep3Time,
                             step3_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep4Time,
                             step4_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep5Time,
                             step5_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep6Time,
                             step6_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep7Time,
                             step7_time);
}
}  // namespace cuda
}  // namespace common
}  // namespace samgraph
