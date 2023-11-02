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
#include <set>
#include <iostream>
#include <fstream>
#include <bitset>

#include "../common.h"

namespace samgraph {
namespace common {
namespace cuda {

constexpr IdType kMaxDevice = 32;

/*
class DeviceP2PComm {
 public:
  static void Init(int num_worker);
  static void Create(int worker_id, int device_id);

  static auto Get() { CHECK(_p2p_comm->_init); return _p2p_comm; }
  auto Rank() const { return _rank; }
  auto CommSize() const { return _comm_size; }
  auto DevId() const { return _dev; }
  auto Peer(IdType peer) const { CHECK(peer < _comm_size); return _peers[peer]; }

  void Barrier() { pthread_barrier_wait(&_shared_data->barrier); }

  cudaIpcMemHandle_t *IpcMemHandle(IdType rk) {
    return &_shared_data->mem_handle[rk];
  }

  ~DeviceP2PComm();
 private:
  DeviceP2PComm(int num_worker);
  std::vector<std::bitset<kMaxDevice>> SplitClique(int device_id, int &my_clique);
  void FindClique(std::bitset<kMaxDevice> clique,
                  std::bitset<kMaxDevice> neighbor,
                  std::bitset<kMaxDevice> none,
                  std::bitset<kMaxDevice> &max_clique);

  struct SharedData {
    pthread_barrier_t barrier;
    int p2p_matrix[kMaxDevice][kMaxDevice];
    cudaIpcMemHandle_t mem_handle[kMaxDevice];
  };

  bool _init;
  IdType _dev;
  IdType _peers[kMaxDevice];
  IdType _comm_size;
  IdType _rank;
  SharedData *_shared_data;

  static DeviceP2PComm *_p2p_comm;
};

class DistArray {
 public:
  DistArray(void *devptr, DeviceP2PComm *comm, StreamHandle stream = 0);

  auto Rank() const { return _comm->Rank(); }
  auto CommSize() const { return _comm->CommSize(); }

  ~DistArray();

  // device handle in cuda kernel
  struct DeviceHandle {
    template<typename T>
    inline __device__ T Get(IdType rk, IdType idx) {
      assert(rk < comm_size);
      assert(devptrs != nullptr);
      auto ptr = static_cast<T *>(devptrs[rk]);
      return ptr[idx];
    }

    IdType rank, comm_size;
    void **devptrs;
  };
  DeviceHandle GetDeviceHandle() const {
    return DistArray::DeviceHandle{_comm->Rank(), _comm->CommSize(), _devptrs_d};
  }

 private:
  void **_devptrs_d;
  void **_devptrs_h;
  DeviceP2PComm *_comm;
};
*/


class DeviceDistGraph {
 public:
  DeviceDistGraph(IdType **part_indptr, IdType **part_indices,
      IdType num_partition,
      IdType num_cache_node,
      IdType num_node)
    : _part_indptr(part_indptr), _part_indices(part_indices),
      _num_partition(num_partition), _num_cache_node(num_cache_node),
      _num_node(num_node) {};

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
    if (node_id < _num_cache_node) {
      *part_id = (node_id % _num_partition);
      *real_id = (node_id / _num_partition);
    } else {
      // the host memory whole graph in the last position
      *part_id = _num_partition;
      *real_id = node_id;
    }
  }

  IdType **_part_indptr;
  IdType **_part_indices;
  IdType _num_partition;
  IdType _num_cache_node;
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

class DeviceDistFeature {
 public:
  DeviceDistFeature() = default;
  DeviceDistFeature(void **part_feature_data, IdType num_part,
      IdType num_cache_node, IdType dim)
    : _part_feature_data(part_feature_data),
      _num_partition(num_part),
      _num_cache_node(num_cache_node),
      _dim(dim) {};

  template<typename T>
  inline __device__ const T& Get(IdType node_id, IdType col) {
    assert(node_id < _num_cache_node);
    assert(col < _dim);
    IdType part_id, real_id;
    _GetRealPartId(node_id, &part_id, &real_id);
    auto part_ptr = static_cast<T*>(_part_feature_data[part_id]);
    return part_ptr[real_id * _dim + col];
  }

 private:
  inline __device__ void _GetRealPartId(IdType node_id,
    IdType *part_id, IdType *real_id) {
    *part_id = (node_id % _num_partition);
    *real_id = (node_id / _num_partition);
  }
  void **_part_feature_data;
  IdType _num_partition;
  IdType _num_cache_node;
  IdType _dim;
};


class DistGraph {
 public:
  struct GroupConfig{
    // for which GPU context
    Context ctx;
    // store which partition IDs
    std::vector<IdType> part_ids;
    // access part 0,1, ... n from ctx_group[0], ctx_group[1] ... ctx_group[n]
    // ctx_group.size() is equal to running GPUs
    std::vector<Context> ctx_group;
    GroupConfig(Context ctx_, const std::vector<IdType> &part_ids_,
        const std::vector<Context> &group_)
      : ctx(ctx_), part_ids(part_ids_), ctx_group(group_) {};
    friend std::ostream& operator<<(std::ostream &os, const GroupConfig &config);
  };

  GroupConfig GetGroupConfig(int device_id) const {
    return _group_configs[device_id];
  }
  void GraphLoad(Dataset *dataset, int sampler_id, Context sampler_ctx,
      IdType num_cache_node);
  virtual void FeatureLoad(int trainer_id, Context trainer_ctx,
      const IdType *cache_rank_node, const IdType num_cache_node,
      DataType dtype, size_t dim,
      const void* cpu_src_feature_data,
      StreamHandle stream = nullptr);
  DeviceDistGraph DeviceGraphHandle() const;
  DeviceDistFeature DeviceFeatureHandle() const;

  static void Create(std::vector<Context> ctxes);
  static void Release(DistGraph *dist_graph);
  static std::shared_ptr<DistGraph> Get() {
    CHECK(_inst != nullptr) << "The static instance is not be initialized";
    return _inst;
  }

 protected:
  IdType _trainer_id;
  IdType _num_feature_cache_node;
  IdType _num_node;
  IdType _feat_dim;
  std::vector<TensorPtr> _part_feature;
  void **_d_part_feature;
  std::vector<Context> _ctxes;

  DistGraph(std::vector<Context> ctxes);
  virtual ~DistGraph() { LOG(ERROR) << "Do not call function in here"; };

  struct SharedData {
    pthread_barrier_t barrier;
    cudaIpcMemHandle_t mem_handle[kMaxDevice][kMaxDevice];
  };
  SharedData *_shared_data;
  void _Barrier();
  static std::shared_ptr<DistGraph> _inst;

 private:
  DistGraph() = delete;
  DistGraph(const DistGraph &) = delete;
  DistGraph& operator = (const DistGraph &) = delete;

  void _DatasetPartition(const Dataset *dataset, Context ctx,
    IdType part_id, IdType num_part, IdType num_part_node);
  void _DataIpcShare(std::vector<TensorPtr> &part_data,
      const std::vector<std::vector<size_t>> &shape_vec,
      const std::vector<IdType> &part_ids,
      Context cur_ctx,
      const std::vector<Context> &ctx_group,
      std::string name);

  int _sampler_id;
  std::vector<TensorPtr> _part_indptr;
  std::vector<TensorPtr> _part_indices;
  std::vector<GroupConfig> _group_configs;
  IdType _num_graph_cache_node;

  IdType **_d_part_indptr;
  IdType **_d_part_indices;

};

class PartitionSolver {
 public:

  struct LinkTopoInfo {
    double bandwidth_matrix[kMaxDevice][kMaxDevice];
    int nvlink_matrix[kMaxDevice][kMaxDevice];
  };
  PartitionSolver(const std::vector<Context> &ctxes);
  std::vector<DistGraph::GroupConfig> solve() const ;
  const LinkTopoInfo* GetLinkTopoInfo() const {
    return &_topo_info;
  }
 private:
  std::vector<Context> _ctxes;
  LinkTopoInfo _topo_info;
  void DetectTopo();
  void DetectTopo_child(const std::string &topo_file);
  void LoadTopoFromFile(std::ifstream &ifs);

};

} // cuda
} // common
} // namespace samgraph

#endif // SAMGRAPH_DIST_GRAPH_H
