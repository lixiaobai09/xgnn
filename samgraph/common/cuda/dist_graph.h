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

#ifndef SAMGRAPH_DIST_GRAPH_H
#define SAMGRAPH_DIST_GRAPH_H

#include <memory>
#include <cassert>
#include <cuda_runtime_api.h>

#include "../common.h"

namespace samgraph {
namespace common {
namespace cuda {

class DeviceDistGraph {
 public:
  DeviceDistGraph(IdType **part_indptr, IdType **part_indices,
      IdType num_partition,
      IdType num_node)
    : _part_indptr(part_indptr), _part_indices(part_indices),
      _num_partition(num_partition), _num_node(num_node) {};

  inline __device__ IdType NumEdge(IdType node_id) {
    assert(node_id < _num_node);
    IdType part_id, real_id;
    _GetRealPartId(node_id, &part_id, &real_id);
    IdType *indptr = _part_indptr[part_id];
    return (indptr[real_id + 1] - indptr[real_id]);
  }

  inline __device__ const IdType* operator[] (IdType node_id) {
    IdType part_id, real_id;
    _GetRealPartId(node_id, &part_id, &real_id);
    IdType *indptr = _part_indptr[part_id];
    IdType *indices = _part_indices[part_id];
    return (indices + indptr[real_id]);
  }

 private:
  inline __device__ void _GetRealPartId(IdType node_id,
      IdType *part_id, IdType *real_id) {
    *part_id = (node_id % _num_partition);
    *real_id = (node_id / _num_partition);
  }

  IdType **_part_indptr;
  IdType **_part_indices;
  IdType _num_partition;
  IdType _num_node;
};

class DeviceNormalGraph {
 public:
  DeviceNormalGraph(const IdType *indptr, const IdType *indices,
      IdType num_node)
    : _indptr(indptr), _indices(indices), _num_node(num_node) {};

  inline __device__ IdType NumEdge(IdType node_id) {
    assert(node_id < _num_node);
    return (_indptr[node_id + 1] - _indptr[node_id]);
  }

  inline __device__ const IdType* operator[] (IdType node_id) {
    IdType offset = _indptr[node_id];
    return (_indices + offset);
  }

 private:
  const IdType *_indptr;
  const IdType *_indices;
  IdType _num_node;
};

constexpr int kMaxDevice = 32;

class DistGraph {
 public:
  void DatasetLoad(Dataset *dataset, int sampler_id, Context sampler_ctx);
  DeviceDistGraph DeviceHandle() const;

  static void Create(std::vector<Context> ctxes);
  static void Release(DistGraph *dist_graph);
  static std::shared_ptr<DistGraph> Get() {
    CHECK(_inst != nullptr) << "The static instance is not be initialized";
    return _inst;
  }

 private:
  DistGraph() = delete;
  DistGraph(const DistGraph &) = delete;
  DistGraph& operator = (const DistGraph &) = delete;
  ~DistGraph() = delete;

  DistGraph(std::vector<Context> ctxes);
  void _Barrier();
  void _DatasetPartition(const Dataset *dataset, int sampler_id);

  std::vector<TensorPtr> _part_indptr;
  std::vector<TensorPtr> _part_indices;
  std::vector<Context> _ctxes;
  IdType _num_node;

  IdType **_d_part_indptr;
  IdType **_d_part_indices;

  struct SharedData {
    pthread_barrier_t barrier;
    cudaIpcMemHandle_t mem_handle[kMaxDevice];
  };
  SharedData *_shared_data;

  static std::shared_ptr<DistGraph> _inst;
};

} // cuda
} // common
} // namespace samgraph

#endif // SAMGRAPH_DIST_GRAPH_H
