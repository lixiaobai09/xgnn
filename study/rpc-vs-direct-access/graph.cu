#include "graph.h"

#include <cstring>

PartitionGraph::PartitionGraph(const Graph *graph, IdType num_part) {
  auto indptr_data = graph->indptr;
  auto indices_data = graph->indices;
  IdType num_node = graph->num_node;
  std::vector<IdType> part_indptr_count(num_part);
  std::vector<IdType> part_edge_count(num_part);
  for (IdType i = 0; i < num_part; ++i) {
    part_indptr_count[i] = (num_node / num_part +
        (i < num_node & num_part? 1 : 0) + 1);
  }
  for (IdType i = 0; i < num_node; ++i) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    part_edge_count[i % num_part] += num_edge;
  }

  // allocate host part data and partition
  std::vector<IdType*> part_indptr_vec(num_part);
  std::vector<IdType*> part_indices_vec(num_part);
  for (IdType i = 0; i < num_part; ++i) {
    part_indptr_vec[i] = static_cast<IdType*>(
        malloc(part_indptr_count[i] * sizeof(IdType)));
    part_indices_vec[i] = static_cast<IdType*>(
        malloc(part_edge_count[i] * sizeof(IdType)));
  }
  std::fill_n(part_edge_count.begin(), num_part, 0);

  for (IdType i = 0; i < num_node; ++i) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    IdType part_id = (i % num_part);
    IdType real_id = (i / num_part);
    part_indptr_vec[part_id][real_id] = part_edge_count[part_id];
    std::memcpy(
        &part_indices_vec[part_id][part_edge_count[part_id]],
        &indices_data[indptr_data[i]],
        num_edge * sizeof(IdType));
    part_edge_count[part_id] += num_edge;
  }
  for (IdType part_id = 0; part_id < num_part; ++part_id) {
    part_indptr_vec[part_id][part_indptr_count[part_id] - 1] =
      part_edge_count[part_id];
  }

  // move data to GPU
  std::vector<IdType*> part_d_indptr_vec(num_part);
  std::vector<IdType*> part_d_indices_vec(num_part);
  IdType** d_part_indptr;
  IdType** d_part_indices;
  CUDA_CALL(cudaMalloc((void**)&d_part_indptr, num_part * sizeof(IdType*)));
  CUDA_CALL(cudaMalloc((void**)&d_part_indices, num_part * sizeof(IdType*)));
  for (IdType i = 0; i < num_part; ++i) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc((void**)&part_d_indptr_vec[i],
        part_indptr_count[i] * sizeof(IdType)));
    CUDA_CALL(cudaMalloc((void**)&part_d_indices_vec[i],
        part_edge_count[i] * sizeof(IdType)));
    CUDA_CALL(cudaMemcpy(part_d_indptr_vec[i], part_indptr_vec[i],
        part_indptr_count[i] * sizeof(IdType), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(part_d_indices_vec[i], part_indices_vec[i],
        part_edge_count[i] * sizeof(IdType), cudaMemcpyDefault));
  }
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaMemcpy(d_part_indptr, part_d_indptr_vec.data(),
        num_part * sizeof(IdType*), cudaMemcpyDefault));
  CUDA_CALL(cudaMemcpy(d_part_indices, part_d_indices_vec.data(),
        num_part * sizeof(IdType*), cudaMemcpyDefault));

  // free the host part data
  for (auto data_vec : {part_indptr_vec, part_indices_vec}) {
    for (IdType* data : data_vec) {
      free(data);
    }
  }

  // set data of PartitionGraph
  _num_node = num_node;
  _num_edge = graph->num_edge;
  _num_part = num_part;
  _d_part_indptr = d_part_indptr;
  _d_part_indices = d_part_indices;

  _part_d_indptr_vec = part_d_indptr_vec;
  _part_d_indices_vec = part_d_indices_vec;
}
