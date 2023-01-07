#pragma once

#include <cassert>
#include <vector>
#include <iostream>

#include <curand_kernel.h>

#define CUDA_CALL(func)                             \
 {                                                  \
    cudaError_t err = func;                         \
    if(err != cudaSuccess) {                        \
        std::cerr << __FILE__ << ":" << __LINE__    \
             << " " << #func << " "                 \
             << cudaGetErrorString(err)             \
             << " errnum " << err;                  \
        exit(EXIT_FAILURE);                         \
    }                                               \
 }

using IdType = uint32_t;
constexpr IdType kEmptyKey = static_cast<IdType>(-1);

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
      IdType num_node, IdType num_part)
    : _indptr(indptr), _indices(indices), _num_node(num_node),
      _num_part(num_part) {};

  inline __device__ IdType NumEdge(IdType node_id) {
    node_id = node_id / _num_part;
    assert(node_id < _num_node);
    return (_indptr[node_id + 1] - _indptr[node_id]);
  }

  inline __device__ const IdType* operator[] (IdType node_id) {
    node_id = node_id / _num_part;
    IdType offset = _indptr[node_id];
    return (_indices + offset);
  }

 private:
  const IdType *_indptr;
  const IdType *_indices;
  IdType _num_node;
  IdType _num_part;
};

struct Graph {
  IdType* indptr;
  IdType* indices;
  IdType* train_set;
  IdType num_node;
  IdType num_edge;
  IdType num_train_set;
};

class PartitionGraph {
 public:
  PartitionGraph() = delete;
  PartitionGraph(const Graph* graph, IdType num_part);
  ~PartitionGraph() {
    for (auto data_vec : {_part_d_indptr_vec, _part_d_indices_vec}) {
      for (auto data : data_vec) {
        CUDA_CALL(cudaFree(data));
      }
    }
    CUDA_CALL(cudaFree(reinterpret_cast<void*>(_d_part_indptr)));
    CUDA_CALL(cudaFree(reinterpret_cast<void*>(_d_part_indices)));
  }
  DeviceDistGraph DeviceHandle() const {
    return DeviceDistGraph(
        _d_part_indptr, _d_part_indices, _num_part, _num_node);
  }
  DeviceNormalGraph NormalGraph(IdType part_id) const {
    return DeviceNormalGraph(_part_d_indptr_vec[part_id],
        _part_d_indices_vec[part_id],
        _num_node, _num_part);
  }

 private:
  IdType _num_node;
  IdType _num_edge;
  IdType _num_part;
  IdType** _d_part_indptr;
  IdType** _d_part_indices;
  // used to mark and free GPU data
  std::vector<IdType*> _part_d_indptr_vec;
  std::vector<IdType*> _part_d_indices_vec;
};
