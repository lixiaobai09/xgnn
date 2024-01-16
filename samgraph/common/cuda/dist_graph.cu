#include "dist_graph.h"

#include <fcntl.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/unistd.h>
#include <sys/wait.h>

#include <cstring>
#include <iomanip>
#include <set>
#include <algorithm>
#include <fstream>
#include <regex>
#include <iostream>

#include "../device.h"
#include "../timer.h"
#include "../run_config.h"
#include "../function.h"
#include "../profiler.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

template<typename T>
std::vector<T> operator- (const std::set<T> &a, const std::multiset<T> &b) {
  std::vector<T> ret;
  for (auto i : a) {
    if (!b.count(i)) {
      ret.emplace_back(i);
    }
  }
  return std::move(ret);
};

void search_access_config(int gpu,
    int part_id,
    int n_part,
    std::vector<int> &access_config,
    std::vector<int> &result,
    double &min_bandwidth,
    const std::vector<std::set<int>> &part_gpu_map,
    const double bandwidth_matrix[][kMaxDevice]) {
  if (part_id == n_part) {
    double max_bandwidth = 0.0;
    const double *bandwidth_list = bandwidth_matrix[gpu];
    int n_gpu = n_part;
    std::vector<int> access_cnt(n_gpu, 0);
    for (auto gpu_i : access_config) {
      access_cnt[gpu_i] += 1;
    }
    for (int i = 0; i < n_gpu; ++i) {
      if (access_cnt[i]
          && (
            static_cast<double>(access_cnt[i]) / bandwidth_list[i]
            > max_bandwidth
          )) {
        max_bandwidth = static_cast<double>(access_cnt[i]) / bandwidth_list[i];
      }
    }
    if (max_bandwidth < min_bandwidth) {
      min_bandwidth = max_bandwidth;
      result = access_config;
    }
    return;
  }
  for (auto gpu_j : part_gpu_map[part_id]) {
    access_config.emplace_back(gpu_j);
    search_access_config(gpu, part_id + 1, n_part,
        access_config, result, min_bandwidth,
        part_gpu_map, bandwidth_matrix);
    access_config.pop_back();
  }
}

using ResultType = std::vector<std::tuple<std::set<int>, std::vector<int>>>;

void solver_recursive(int current_gpu,
    int n_gpu,
    int access_current_id,
    std::vector<int> can_not_access_parts,
    std::vector<std::set<int>> &store_parts,
    std::vector<std::multiset<int>> &can_access_parts,
    ResultType &result,
    double &min_max_bandwidth,
    int &global_min_parts,
    const std::set<int> &parts_universal_set,
    const std::vector<std::set<int>> &neighbor_adjacency,
    const double bandwidth_matrix[][kMaxDevice]) {

  // stop condition
  if (current_gpu == n_gpu) {
    std::vector<std::vector<int>> access_config_list;
    double max_bandwidth = 0.0;
    int max_parts = 0;
    // std::stringstream ss;
    for (int gpu = 0; gpu < store_parts.size(); ++gpu) {
      if (store_parts[gpu].size() > max_parts) {
        max_parts = store_parts[gpu].size();
      }
      // for (auto part_id : store_parts[gpu]) {
      //   ss << part_id;
      // }
      // ss << '|';
    }
    // std::cout << ss.str() << std::endl;
    if (max_parts > global_min_parts) {
      return;
    }
    if (max_parts < global_min_parts) {
      min_max_bandwidth = std::numeric_limits<double>::max();
      global_min_parts = max_parts;
    }
    for (int gpu = 0; gpu < store_parts.size(); ++gpu) {
      // std::cout << "gpu: " << gpu << std::endl;
      std::vector<std::set<int>> part_gpu_map(n_gpu);
      for (auto neighbor : neighbor_adjacency[gpu]) {
        for (auto store_part : store_parts[neighbor]) {
          part_gpu_map[store_part].insert(neighbor);
        }
      }
      std::vector<int> access_config;
      std::vector<int> result;
      double min_bandwidth = std::numeric_limits<double>::max();
      search_access_config(gpu, 0, n_gpu, access_config, result,
          min_bandwidth, part_gpu_map, bandwidth_matrix);
      if (min_bandwidth > max_bandwidth) {
        max_bandwidth = min_bandwidth;
      }
      access_config_list.emplace_back(result);
      // for (auto gpu_j : result) { std::cout << gpu_j << " "; }
      // std::cout << std::endl;
    }
    if (max_bandwidth < min_max_bandwidth) {
      min_max_bandwidth = max_bandwidth;
      result.clear();
      for (int i = 0; i < n_gpu; ++i) {
        result.emplace_back(store_parts[i], access_config_list[i]);
      }
    }
    return;
  }
  if (can_not_access_parts.size() == 0) {
    // get can not access parts for GPU i
    can_not_access_parts =
      (parts_universal_set - can_access_parts[current_gpu]);
    access_current_id = 0;
  }
  if (access_current_id < can_not_access_parts.size()) {
    int need_part = can_not_access_parts[access_current_id];
    // id, stored_parts_size, need_score, if_same_part_in_neighbors
    std::vector<std::tuple<int, int, int, int>> tmp_vec;
    for (auto j : neighbor_adjacency[current_gpu]) {
      int need_score = 0;
      for (auto k : neighbor_adjacency[j]) {
        if (!can_access_parts[k].count(need_part)) {
          ++need_score;
        }
      }
      tmp_vec.emplace_back(j, store_parts[j].size(), need_score,
          // XXX: if need this?
          (can_access_parts[j].count(need_part) == 0? 0 : 1));
    }
    std::sort(tmp_vec.begin(), tmp_vec.end(), [](auto x, auto y){
          // stored_parts_size
          if (std::get<1>(x) != std::get<1>(y)) {
            return std::get<1>(x) < std::get<1>(y);
          }
          // need_score
          if (std::get<2>(x) != std::get<2>(y)) {
            return std::get<2>(x) > std::get<2>(y);
          }
          // if_same_part_in_neighbors 0 or 1
          if (std::get<3>(x) != std::get<3>(y)) {
            return std::get<3>(x) < std::get<3>(y);
          }
          return std::get<0>(x) < std::get<0>(y);
        });
    int last = (tmp_vec.size() - 1);
    auto cmp_equal = [](const std::tuple<int, int, int, int> &x,
        const std::tuple<int, int, int, int> &y) {
      return (std::get<1>(x) == std::get<1>(y)
          && std::get<2>(x) == std::get<2>(y)
          && std::get<3>(x) == std::get<3>(y));
    };
    while (!cmp_equal(tmp_vec[0], tmp_vec[last])) {
      tmp_vec.pop_back();
      --last;
    }
    for (auto item : tmp_vec) {
      int store_gpu = std::get<0>(item);
      store_parts[store_gpu].insert(need_part);
      for (auto neighbor : neighbor_adjacency[store_gpu]) {
        can_access_parts[neighbor].insert(need_part);
      }
      solver_recursive(current_gpu, n_gpu,
          access_current_id + 1, can_not_access_parts,
          store_parts, can_access_parts,
          result, min_max_bandwidth, global_min_parts,
          parts_universal_set,
          neighbor_adjacency, bandwidth_matrix);
      // recover
      store_parts[store_gpu].erase(store_parts[store_gpu].find(need_part));
      for (auto neighbor : neighbor_adjacency[store_gpu]) {
        can_access_parts[neighbor].erase(
            can_access_parts[neighbor].find(need_part));
      }
    }
  } else {
    can_not_access_parts.clear();
    solver_recursive(current_gpu + 1, n_gpu, 0, can_not_access_parts,
        store_parts, can_access_parts,
        result, min_max_bandwidth, global_min_parts,
        parts_universal_set, neighbor_adjacency,
        bandwidth_matrix);
  }

}

}; // namespace

std::shared_ptr<DistGraph> DistGraph::_inst = nullptr;

void DistGraph::_DatasetPartition(const Dataset *dataset, Context ctx,
    IdType part_id, IdType num_part, IdType num_part_node) {
  auto indptr_data = dataset->indptr->CPtr<IdType>();
  auto indices_data = dataset->indices->CPtr<IdType>();
  IdType part_edge_count = 0;
  for (IdType i = part_id; i < num_part_node; i += num_part) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    part_edge_count += num_edge;
  }
  CHECK_NE(part_edge_count, 0);

  std::stringstream ctx_name;
  ctx_name << ctx;

  IdType indptr_size = (num_part_node / num_part +
      (part_id < num_part_node % num_part? 1 : 0) + 1);
  _part_indptr[part_id] = Tensor::Empty(kI32, {indptr_size}, CPU(),
      "indptr in device:cpu" );
  _part_indices[part_id] = Tensor::Empty(kI32, {part_edge_count}, CPU(),
      "indices in device:cpu");
  LOG(DEBUG) << "host memory for partitions: "
    << ToReadableSize((indptr_size + part_edge_count) * sizeof(IdType));
  part_edge_count = 0;

  for (IdType i = part_id; i < num_part_node; i += num_part) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    IdType real_id = (i / num_part);
    _part_indptr[part_id]->Ptr<IdType>()[real_id] = part_edge_count;
    std::memcpy(
        &_part_indices[part_id]->Ptr<IdType>()[part_edge_count],
        &indices_data[indptr_data[i]],
        num_edge * sizeof(IdType));
    part_edge_count += num_edge;
  }
  _part_indptr[part_id]->Ptr<IdType>()[indptr_size - 1] = part_edge_count;

  if (ctx.device_type != kCPU) {
    LOG(DEBUG) << "Load Graph to GPU: " << ctx << " store " 
               << ToReadableSize(_part_indices[part_id]->NumBytes() + _part_indptr[part_id]->NumBytes());
    _part_indptr[part_id] = Tensor::CopyTo(_part_indptr[part_id], ctx,
        nullptr, ctx_name.str(), Constant::kAllocNoScale);
    _part_indices[part_id] = Tensor::CopyTo(_part_indices[part_id], ctx,
        nullptr, ctx_name.str(), Constant::kAllocNoScale);
  }
}

void DistGraph::_DataIpcShare(std::vector<TensorPtr> &part_data,
    const std::vector<std::vector<size_t>> &shape_vec,
    const std::vector<IdType> &part_ids,
    Context cur_ctx,
    const std::vector<Context> &ctx_group,
    std::string name) {

  for (IdType part_id : part_ids) {
    // share self data to others
    CHECK(cur_ctx == part_data[part_id]->Ctx());
    CHECK(shape_vec[part_id] == part_data[part_id]->Shape());
    auto shared_data = part_data[part_id]->CPtr<IdType>();
    cudaIpcMemHandle_t &mem_handle =
      _shared_data->mem_handle[cur_ctx.device_id][part_id];
    CUDA_CALL(cudaIpcGetMemHandle(&mem_handle, (void*)shared_data));
  }
  _Barrier();

  // receive data from others
  int num_part = static_cast<int>(part_data.size());
  for (int i = 0; i < num_part; ++i) {
    if (part_data[i] != nullptr) {
      continue;
    }
    auto ctx = ctx_group[i];
    cudaIpcMemHandle_t &mem_handle = _shared_data->mem_handle[ctx.device_id][i];
    void *ptr;
    CUDA_CALL(cudaIpcOpenMemHandle(
          &ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
    part_data[i] = Tensor::FromBlob(ptr, kI32, shape_vec[i], ctx,
        name + " in device:" + std::to_string(ctx.device_id));
  }
  _Barrier();
}

void DistGraph::GraphLoad(Dataset *dataset, int sampler_id,
    Context sampler_ctx, IdType num_cache_node) {

  CHECK(sampler_ctx == _group_configs[sampler_id].ctx);
  CHECK(num_cache_node <= dataset->num_node);
  _sampler_id = sampler_id;
  _num_graph_cache_node = num_cache_node;
  _num_node = dataset->num_node;

  auto part_ids = _group_configs[sampler_id].part_ids;
  auto ctx_group = _group_configs[sampler_id].ctx_group;
  IdType num_part = ctx_group.size();
  _part_indptr.clear();
  _part_indptr.resize(num_part, nullptr);
  _part_indices.clear();
  _part_indices.resize(num_part, nullptr);

  LOG(DEBUG) << "before graph partition";
  if (RunConfig::dist_graph_part_cpu < 1) {
    for (IdType part_id : part_ids) {
      _DatasetPartition(dataset, sampler_ctx, part_id, num_part,
          _num_graph_cache_node);
    }
  } else {
    auto ctx_group = _group_configs[sampler_id].ctx_group;
    for (int i = 0; i < part_ids.size(); i++) {
      _DatasetPartition(dataset, ctx_group[i], part_ids[i], num_part,
          dataset->num_node);
    }
  }

  LOG(DEBUG) << "send and receive graph partitions";
  if (RunConfig::dist_graph_part_cpu < 1) {
    std::vector<std::vector<size_t>> shape_vec(num_part);
    for (size_t i = 0; i < num_part; ++i) {
      size_t num_part_node = (_num_graph_cache_node / num_part +
          (i < _num_graph_cache_node % num_part? 1 : 0) + 1);
      shape_vec[i] = std::vector<size_t>({num_part_node});
    }
    _DataIpcShare(_part_indptr, shape_vec, part_ids,
        sampler_ctx, ctx_group, "dataset part indptr");

    shape_vec.clear();
    shape_vec.resize(num_part, {0});
    std::vector<size_t> part_size_vec(num_part, 0);
    auto indptr_data = dataset->indptr->CPtr<IdType>();
    for (IdType i = 0; i < _num_graph_cache_node; ++i) {
      IdType num_edge = indptr_data[i + 1] - indptr_data[i];
      IdType tmp_part_id = (i % num_part);
      part_size_vec[tmp_part_id] += num_edge;
    }
    for (IdType i = 0; i < num_part; ++i) {
      shape_vec[i] = std::vector<size_t>({part_size_vec[i]});
    }
    _DataIpcShare(_part_indices, shape_vec, part_ids,
        sampler_ctx, ctx_group, "dataset part indices");
  }

  CUDA_CALL(cudaMalloc((void **)&_d_part_indptr, (num_part + 1) * sizeof(IdType *)));
  CUDA_CALL(cudaMalloc((void **)&_d_part_indices, (num_part + 1) * sizeof(IdType *)));

  IdType **h_part_indptr, **h_part_indices;
  CUDA_CALL(cudaMallocHost(&h_part_indptr, (num_part + 1) * sizeof(IdType*)));
  CUDA_CALL(cudaMallocHost(&h_part_indices, (num_part + 1) * sizeof(IdType*)));
  for (IdType i = 0; i < num_part; i++) {
    h_part_indptr[i] = _part_indptr[i]->Ptr<IdType>();
    h_part_indices[i] = _part_indices[i]->Ptr<IdType>();
  }
  // the last point to host whole graph
  h_part_indptr[num_part] = dataset->indptr->Ptr<IdType>();
  h_part_indices[num_part] = dataset->indices->Ptr<IdType>();
  CUDA_CALL(cudaMemcpy(_d_part_indptr, h_part_indptr, sizeof(IdType *) * (num_part + 1), cudaMemcpyDefault));
  CUDA_CALL(cudaMemcpy(_d_part_indices, h_part_indices, sizeof(IdType *) * (num_part + 1), cudaMemcpyDefault));

  CUDA_CALL(cudaFreeHost(h_part_indptr));
  CUDA_CALL(cudaFreeHost(h_part_indices));
}

void DistGraph::FeatureLoad(int trainer_id, Context trainer_ctx,
    const IdType *cache_rank_node, const IdType num_cache_node,
    DataType dtype, size_t dim,
    const void* cpu_src_feature_data,
    StreamHandle stream) {
  if (getenv("SAMGRAPH_CLIQUE") != nullptr) {
    std::cout << "use clique placement" << std::endl;
#if 1 // used for clique policy
    if (_ctxes.size() == 4) {
      PartitionSolver solver(_ctxes);
      auto nvlink_matrix = solver.GetLinkTopoInfo()->nvlink_matrix;
      if (nvlink_matrix[0][3] == 0) {
        _group_configs.clear();
        _group_configs.emplace_back(GroupConfig(GPU(0), {0, 1},
          {GPU(0), GPU(0), GPU(1), GPU(1)}));
        _group_configs.emplace_back(GroupConfig(GPU(1), {2, 3},
          {GPU(0), GPU(0), GPU(1), GPU(1)}));

        _group_configs.emplace_back(GroupConfig(GPU(2), {0, 1},
          {GPU(2), GPU(2), GPU(3), GPU(3)}));
        _group_configs.emplace_back(GroupConfig(GPU(3), {2, 3},
          {GPU(2), GPU(2), GPU(3), GPU(3)}));
      } else {
        _group_configs.clear();
        _group_configs.emplace_back(GroupConfig(GPU(0), {0},
          {GPU(0), GPU(1), GPU(2), GPU(3)}));
        _group_configs.emplace_back(GroupConfig(GPU(1), {1},
          {GPU(0), GPU(1), GPU(2), GPU(3)}));
        _group_configs.emplace_back(GroupConfig(GPU(2), {2},
          {GPU(0), GPU(1), GPU(2), GPU(3)}));
        _group_configs.emplace_back(GroupConfig(GPU(3), {3},
          {GPU(0), GPU(1), GPU(2), GPU(3)}));
      }
    }
    if (_ctxes.size() == 6) {
      _group_configs.clear();
      /*
      _group_configs.emplace_back(GroupConfig(GPU(0), {0, 1, 2},
        {GPU(0), GPU(0), GPU(0), GPU(1), GPU(1), GPU(1),
         GPU(2), GPU(2), GPU(2), GPU(3), GPU(3), GPU(3)}));
      _group_configs.emplace_back(GroupConfig(GPU(1), {3, 4, 5},
        {GPU(0), GPU(0), GPU(0), GPU(1), GPU(1), GPU(1),
         GPU(2), GPU(2), GPU(2), GPU(3), GPU(3), GPU(3)}));
      _group_configs.emplace_back(GroupConfig(GPU(2), {6, 7, 8},
        {GPU(0), GPU(0), GPU(0), GPU(1), GPU(1), GPU(1),
         GPU(2), GPU(2), GPU(2), GPU(3), GPU(3), GPU(3)}));
      _group_configs.emplace_back(GroupConfig(GPU(3), {9, 10 ,11},
        {GPU(0), GPU(0), GPU(0), GPU(1), GPU(1), GPU(1),
         GPU(2), GPU(2), GPU(2), GPU(3), GPU(3), GPU(3)}));
      */

      _group_configs.emplace_back(GroupConfig(GPU(0), {0, 1, 2, 3, 4, 5},
        {GPU(0), GPU(0), GPU(0), GPU(0), GPU(0), GPU(0),
         GPU(1), GPU(1), GPU(1), GPU(1), GPU(1), GPU(1)}));
      _group_configs.emplace_back(GroupConfig(GPU(1), {6, 7, 8, 9, 10, 11},
        {GPU(0), GPU(0), GPU(0), GPU(0), GPU(0), GPU(0),
         GPU(1), GPU(1), GPU(1), GPU(1), GPU(1), GPU(1)}));

      _group_configs.emplace_back(GroupConfig(GPU(2), {0, 1, 2, 3, 4, 5},
        {GPU(2), GPU(2), GPU(2), GPU(2), GPU(2), GPU(2),
         GPU(3), GPU(3), GPU(3), GPU(3), GPU(3), GPU(3)}));
      _group_configs.emplace_back(GroupConfig(GPU(3), {6, 7, 8, 9, 10, 11},
        {GPU(2), GPU(2), GPU(2), GPU(2), GPU(2), GPU(2),
         GPU(3), GPU(3), GPU(3), GPU(3), GPU(3), GPU(3)}));

      _group_configs.emplace_back(GroupConfig(GPU(4), {0, 1, 2, 3, 4, 5},
        {GPU(4), GPU(4), GPU(4), GPU(4), GPU(4), GPU(4),
         GPU(5), GPU(5), GPU(5), GPU(5), GPU(5), GPU(5)}));
      _group_configs.emplace_back(GroupConfig(GPU(5), {6, 7, 8, 9, 10, 11},
        {GPU(4), GPU(4), GPU(4), GPU(4), GPU(4), GPU(4),
         GPU(5), GPU(5), GPU(5), GPU(5), GPU(5), GPU(5)}));
    }
    if (_ctxes.size() == 8) {
      _group_configs.clear();
      _group_configs.emplace_back(GroupConfig(GPU(0), {0, 1},
          {GPU(0), GPU(0), GPU(1), GPU(1), GPU(2), GPU(2), GPU(3), GPU(3)}));
      _group_configs.emplace_back(GroupConfig(GPU(1), {2, 3},
          {GPU(0), GPU(0), GPU(1), GPU(1), GPU(2), GPU(2), GPU(3), GPU(3)}));
      _group_configs.emplace_back(GroupConfig(GPU(2), {4, 5},
          {GPU(0), GPU(0), GPU(1), GPU(1), GPU(2), GPU(2), GPU(3), GPU(3)}));
      _group_configs.emplace_back(GroupConfig(GPU(3), {6, 7},
          {GPU(0), GPU(0), GPU(1), GPU(1), GPU(2), GPU(2), GPU(3), GPU(3)}));

      _group_configs.emplace_back(GroupConfig(GPU(4), {0, 1},
          {GPU(4), GPU(4), GPU(5), GPU(5), GPU(6), GPU(6), GPU(7), GPU(7)}));
      _group_configs.emplace_back(GroupConfig(GPU(5), {2, 3},
          {GPU(4), GPU(4), GPU(5), GPU(5), GPU(6), GPU(6), GPU(7), GPU(7)}));
      _group_configs.emplace_back(GroupConfig(GPU(6), {4, 5},
          {GPU(4), GPU(4), GPU(5), GPU(5), GPU(6), GPU(6), GPU(7), GPU(7)}));
      _group_configs.emplace_back(GroupConfig(GPU(7), {6, 7},
          {GPU(4), GPU(4), GPU(5), GPU(5), GPU(6), GPU(6), GPU(7), GPU(7)}));
    }
#endif
  }
  CHECK(trainer_ctx == _group_configs[trainer_id].ctx);

  _trainer_id = trainer_id;
  _num_feature_cache_node = num_cache_node;
  _feat_dim = dim;

  auto part_ids = _group_configs[trainer_id].part_ids;
  auto ctx_group = _group_configs[trainer_id].ctx_group;
  IdType num_part = ctx_group.size();
  _part_feature.resize(num_part, nullptr);

  // partition the feature data with part_ids
  auto _PartitionFeature = [&](IdType part_id) {
    auto cpu_device = Device::Get(CPU());
    IdType num_extract_node = num_cache_node / num_part +
      (part_id < num_cache_node % num_part? 1 : 0);
    IdType *extract_nodes = cpu_device->AllocArray<IdType>(
        CPU(), sizeof(IdType) * num_extract_node);
    auto tmp_cpu_tensor = Tensor::Empty(dtype, {num_extract_node, dim},
        CPU(), "feature cache with part." + std::to_string(part_id));
    void *tmp_cpu_data = tmp_cpu_tensor->MutableData();
    IdType extract_node_cnt = 0;
    for (IdType i = part_id; i < num_cache_node; i += num_part) {
      extract_nodes[extract_node_cnt++] = cache_rank_node[i];
    }
    CHECK(extract_node_cnt == num_extract_node);

    // Populate the cache in cpu memory
    if (RunConfig::option_empty_feat != 0) {
      cpu::CPUMockExtract(tmp_cpu_data, cpu_src_feature_data,
          extract_nodes, num_extract_node, dim, dtype);
    } else {
      cpu::CPUExtract(tmp_cpu_data, cpu_src_feature_data,
          extract_nodes, num_extract_node, dim, dtype);
    }

    auto ret_tensor = Tensor::CopyTo(tmp_cpu_tensor, trainer_ctx,
        stream, Constant::kAllocNoScale);
    cpu_device->FreeWorkspace(CPU(), extract_nodes);
    return ret_tensor;
  };

  for (auto part_id : part_ids) {
    _part_feature[part_id] = _PartitionFeature(part_id);
    Profiler::Get().LogInitAdd(kLogInitL1FeatMemory, _part_feature[part_id]->NumBytes());
  }

  // share partition feature cache to others
  std::vector<std::vector<size_t>> shape_vec(num_part);
  for (IdType i = 0; i < num_part; ++i) {
    IdType num_part_node = (num_cache_node / num_part +
        (i < num_cache_node % num_part? 1 : 0));
    shape_vec[i] = std::vector<size_t>({num_part_node, dim});
  }
  _DataIpcShare(_part_feature, shape_vec, part_ids,
      trainer_ctx, ctx_group, "feature part cache");

  CUDA_CALL(cudaMalloc((void**)&_d_part_feature, num_part * sizeof(void*)));

  void* h_part_feature[num_part];
  for (IdType i = 0; i < num_part; ++i) {
    h_part_feature[i] = _part_feature[i]->MutableData();
  }
  CUDA_CALL(cudaMemcpy(_d_part_feature, h_part_feature,
        sizeof(void*) * num_part,
        cudaMemcpyDefault));
}

DeviceDistGraph DistGraph::DeviceGraphHandle() const {
  return DeviceDistGraph(
      _d_part_indptr, _d_part_indices,
      _part_indptr.size(),
      _num_graph_cache_node,
      _num_node);
}

DeviceDistFeature DistGraph::DeviceFeatureHandle() const {
  CHECK(_feat_dim != 0);
  return DeviceDistFeature(
      _d_part_feature,
      _part_feature.size(),
      _num_feature_cache_node,
      _feat_dim);
}

DistGraph::DistGraph(std::vector<Context> ctxes) {
#if 0 // a perfect solution example
  /*
  _group_configs.clear();
  _group_configs.emplace_back(GroupConfig(GPU(0), {0, 4},
      {GPU(0), GPU(6), GPU(2), GPU(3), GPU(0), GPU(1), GPU(6), GPU(3)}));
  _group_configs.emplace_back(GroupConfig(GPU(1), {1, 5},
      {GPU(7), GPU(1), GPU(2), GPU(3), GPU(0), GPU(1), GPU(2), GPU(7)}));
  _group_configs.emplace_back(GroupConfig(GPU(2), {2, 6},
      {GPU(0), GPU(1), GPU(2), GPU(3), GPU(4), GPU(1), GPU(2), GPU(3)}));
  _group_configs.emplace_back(GroupConfig(GPU(3), {3, 7},
      {GPU(0), GPU(1), GPU(2), GPU(3), GPU(0), GPU(5), GPU(2), GPU(3)}));

  _group_configs.emplace_back(GroupConfig(GPU(4), {4, 3},
      {GPU(7), GPU(6), GPU(5), GPU(4), GPU(4), GPU(5), GPU(2), GPU(7)}));
  _group_configs.emplace_back(GroupConfig(GPU(5), {5, 2},
      {GPU(7), GPU(6), GPU(5), GPU(4), GPU(4), GPU(5), GPU(6), GPU(3)}));
  _group_configs.emplace_back(GroupConfig(GPU(6), {6, 1},
      {GPU(0), GPU(6), GPU(5), GPU(4), GPU(0), GPU(5), GPU(6), GPU(7)}));
  _group_configs.emplace_back(GroupConfig(GPU(7), {7, 0},
      {GPU(7), GPU(1), GPU(5), GPU(4), GPU(4), GPU(1), GPU(6), GPU(7)}));
  */
#endif

  _ctxes = ctxes;

  if (RunConfig::dist_graph_part_cpu < 1) {
    PartitionSolver solver(ctxes);
    _group_configs = solver.solve();
    
  } else {
    _group_configs.clear();
    for (int i = 0; i < ctxes.size(); i++) {
      auto ctx = ctxes[i];
      CHECK_EQ(i, ctx.device_id);
      std::vector<IdType> part_ids;
      std::vector<Context> ctx_group;
      for (int j = 0; j < RunConfig::dist_graph_part_cpu; j++) {
        if (j == 0) {
          ctx_group.push_back(ctx);
        } else {
          ctx_group.push_back(CPU());
        }
        part_ids.push_back(j);
      }
      _group_configs.emplace_back(ctx, part_ids, ctx_group);
    }
  }
  for (auto &config : _group_configs) {
    // LOG(INFO) << config;
    std::cout << config << std::endl;
  }

  int num_worker = ctxes.size();
  _sampler_id = static_cast<int>(Constant::kEmptyKey);
  _trainer_id = static_cast<int>(Constant::kEmptyKey);

  _shared_data = static_cast<SharedData*>(mmap(NULL, sizeof(SharedData),
                      PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
  CHECK_NE(_shared_data, MAP_FAILED);
  pthread_barrierattr_t attr;
  pthread_barrierattr_init(&attr);
  pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_barrier_init(&_shared_data->barrier, &attr, num_worker);
  LOG(DEBUG) << "DistGraph initialized";
}

void DistGraph::_Barrier() {
  int err = pthread_barrier_wait(&_shared_data->barrier);
  CHECK(err == PTHREAD_BARRIER_SERIAL_THREAD || err == 0);
}

void DistGraph::Release(DistGraph *dist_graph) {
  if (dist_graph->_sampler_id != Constant::kEmptyKey) {
    for (int i = 0; i < dist_graph->_part_indptr.size(); i++) {
      if (i != dist_graph->_sampler_id) {
        CUDA_CALL(cudaIpcCloseMemHandle(dist_graph->_part_indptr[i]->MutableData()));
        CUDA_CALL(cudaIpcCloseMemHandle(dist_graph->_part_indices[i]->MutableData()));
      }
    }
    LOG(INFO) << "Release DistGraph" << " " << dist_graph->_sampler_id;
    // pthread_barrier_wait(&dist_graph->_shared_data->barrier);

    CUDA_CALL(cudaFree((void*)dist_graph->_d_part_indptr));
    CUDA_CALL(cudaFree((void*)dist_graph->_d_part_indices));
  }
  if (dist_graph->_trainer_id != Constant::kEmptyKey) {
    for (int i = 0; i < dist_graph->_part_feature.size(); i++) {
      if (i != dist_graph->_trainer_id) {
        CUDA_CALL(cudaIpcCloseMemHandle(
              dist_graph->_part_feature[i]->MutableData()));
      }
    }
    LOG(INFO) << "Release DistFeature" << " " << dist_graph->_trainer_id;
    CUDA_CALL(cudaFree((void*)dist_graph->_d_part_feature));
  }
  pthread_barrier_destroy(&dist_graph->_shared_data->barrier);
  munmap(dist_graph->_shared_data, sizeof(SharedData));
}

void DistGraph::Create(std::vector<Context> ctxes) {
  CHECK(_inst == nullptr);
  _inst = std::shared_ptr<DistGraph>(
      new DistGraph(ctxes), Release);
}


PartitionSolver::PartitionSolver(const std::vector<Context> &ctxes)
  : _ctxes(ctxes) {
  std::set<int> set;
  for (auto&ctx : ctxes) {
    set.insert(ctx.device_id);
  }
  CHECK_EQ(set.size(), ctxes.size());
  CHECK_EQ(*set.rbegin() + 1, set.size());
  DetectTopo();
}

void PartitionSolver::DetectTopo() {
  std::string device_order;
  if (auto s = std::getenv("CUDA_DEVICE_ORDER")) {
    device_order = std::string{s};
  } else {
    device_order = "FASTEST_FIRST";
  }

  std::string topo_file = Constant::kDetectTopoFile + "_" + device_order;
  std::ifstream topo_ifs(topo_file);
  if (!topo_ifs.is_open()) {
    Timer t1;
    int pid = fork();
    CHECK(pid != -1);
    if (pid == 0) {
      unsetenv("CUDA_VISIBLE_DEVICES");
      DetectTopo_child(topo_file);
    } else {
      int wstatus;
      waitpid(pid, &wstatus, 0);
      if (WEXITSTATUS(wstatus) != 0 || WIFSIGNALED(wstatus)) {
        CHECK(false);
      }
    }
    double detect_time = t1.Passed();
    LOG(INFO) << "DetectTopo Done, cost time: " << detect_time << "sec.";

    topo_ifs.open(topo_file);
    CHECK(topo_ifs.is_open());
  }
  LoadTopoFromFile(topo_ifs);
  topo_ifs.close();

  std::stringstream ss;
  ss << "Topology Detect Debug: \n";
  for (int i = 0; i < _ctxes.size(); i++) {
    for (int j = 0; j < _ctxes.size(); j++) {
      ss << std::setw(8) << std::fixed << std::setprecision(1) << _topo_info.bandwidth_matrix[i][j] << " ";
    }
    ss << "\n";
  }
  LOG(INFO) << ss.str();
}

std::vector<DistGraph::GroupConfig> PartitionSolver::solve() const  {
  IdType num_ctx = _ctxes.size();
  const auto &bandwidth_matrix = _topo_info.bandwidth_matrix;

  std::vector<std::vector<int>> access_count(
      num_ctx, std::vector<int>(num_ctx, 0));
  std::vector<std::set<int>> store_parts(num_ctx);

  std::vector<std::multiset<int>> can_access_parts(num_ctx);
  // from bandwidth matrix
  std::vector<std::set<int>> neighbor_adjacency(num_ctx);
  std::set<int> parts_universal_set;
  std::vector<std::tuple<int, int>> asc_degree_gpu_order(num_ctx);
  for (int i = 0; i < num_ctx; ++i) {
    parts_universal_set.insert(i);
    store_parts[i].insert(i);
    for (int j = 0; j < num_ctx; ++j) {
      if (std::abs(bandwidth_matrix[i][j]) > 1e-6) {
        can_access_parts[i].insert(j);
        neighbor_adjacency[i].insert(j);
      }
    }
  }
  ResultType result;
  double max_avg_bandwidth = std::numeric_limits<double>::max();
  int global_min_parts = std::numeric_limits<int>::max();
  solver_recursive(0, num_ctx, 0, {},
      store_parts, can_access_parts,
      result, max_avg_bandwidth, global_min_parts,
      parts_universal_set, neighbor_adjacency,
      bandwidth_matrix);

  std::vector<DistGraph::GroupConfig> configs;
  for (int i = 0; i < num_ctx; i++) {
    auto ctx = _ctxes[i];
    IdType device = ctx.device_id;
    CHECK_EQ(i, device);
    std::vector<IdType> part_ids(std::get<0>(result[device]).begin(),
        std::get<0>(result[device]).end());
    std::vector<Context> ctx_group(num_ctx);
    {
      int j = 0;
      for (auto gpu_id : std::get<1>(result[device])) {
        ctx_group[j++] = GPU(gpu_id);
      }
    }
    configs.emplace_back(ctx, part_ids, ctx_group);
  }
  return configs;
}

void PartitionSolver::DetectTopo_child(const std::string &topo_file) {
  LinkTopoInfo topo_info;
  // 128M buffer for bandwidth test to detect backbone link
  size_t nbytes = (1<<27);
  IdType *buffers[kMaxDevice], *buffersD2D[kMaxDevice];
  cudaStream_t stream[kMaxDevice];

  int num_device;
  CUDA_CALL(cudaGetDeviceCount(&num_device));
  CHECK(num_device <= kMaxDevice);

  std::vector<cudaDeviceProp> universal_devices(num_device);
  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaGetDeviceProperties(&universal_devices[i], i));
  }

  for (int device = 0; device < num_device; device++) {
    CUDA_CALL(cudaSetDevice(device));
    // XXX: long time cost to lazy create ctx
    CUDA_CALL(cudaMalloc(&buffers[device], nbytes));
    CUDA_CALL(cudaMalloc(&buffersD2D[device], nbytes));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream[device], cudaStreamNonBlocking));
    for (int peer = 0; peer < num_device; peer++) {
      topo_info.bandwidth_matrix[device][peer] = 0;
      if (device == peer) {
        topo_info.nvlink_matrix[device][peer] = 1;
        continue;
      }
      int can_access = false;
      CUDA_CALL(cudaDeviceCanAccessPeer(&can_access, device, peer));
      if (!can_access) {
        topo_info.nvlink_matrix[device][peer] = 0;
      } else {
        topo_info.nvlink_matrix[device][peer] = 1;
      }
    }
  }
  for (int device = 0; device < num_device; device++) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemcpyAsync(buffers[device], buffersD2D[device], nbytes, cudaMemcpyDefault, stream[device]));
    CUDA_CALL(cudaStreamSynchronize(stream[device]));
    for (int peer = 0; peer < num_device; peer++) {
      if (device != peer && topo_info.nvlink_matrix[device][peer]) {
        CUDA_CALL(cudaDeviceEnablePeerAccess(peer, 0));
      }
    }
    for (int peer = 0; peer < num_device; peer++) {
      if (topo_info.nvlink_matrix[device][peer]) {
        Timer t0;
        CUDA_CALL(cudaMemcpyAsync(buffers[device], buffersD2D[peer], nbytes, cudaMemcpyDefault, stream[device]));
        CUDA_CALL(cudaStreamSynchronize(stream[device]));
        auto sec = t0.Passed();
        if (device == peer) {
          topo_info.bandwidth_matrix[device][peer] = 2 * nbytes / sec / 1e9;
        } else {
          topo_info.bandwidth_matrix[device][peer] = nbytes / sec / 1e9;
        }
      }
    }
    for (int peer = 0; peer < num_device; peer++) {
      if (device != peer && topo_info.nvlink_matrix[device][peer]) {
        CUDA_CALL(cudaDeviceDisablePeerAccess(peer));
      }
    }
  }

  // release resouce
  for (int device = 0; device < num_device; device++) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaStreamDestroy(stream[device]));
    CUDA_CALL(cudaFree(buffers[device]));
    CUDA_CALL(cudaFree(buffersD2D[device]));
  }

  std::ofstream ofs(topo_file);
  CHECK(ofs.is_open()) << "cannot open " << topo_file;
  ofs << "GPU Count " << universal_devices.size() << "\n";
  ofs << "Device Order " << topo_file.substr(Constant::kDetectTopoFile.size() + 1) << "\n";
  for (int i = 0; i < universal_devices.size(); i++) {
    const auto &prop = universal_devices[i];
    ofs << "GPU [" << i << "] " << prop.name;
    ofs << " (UUID: ";
    for (int j = 0; j < 16; j++) {
      uint8_t x = prop.uuid.bytes[j];
      ofs << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(x) << std::setfill(' ');
    }
    ofs << ")\n";
  }
  ofs << "\n\nP2P Matrix\n";
  for (int i = 0; i < num_device; i++) {
    for (int j = 0; j < num_device; j++) {
      ofs << std::setw(4) << topo_info.nvlink_matrix[i][j] << " ";
    }
    ofs << "\n";
  }
  ofs << "\n\nBandwidth Matrix\n";
  for (int i = 0; i < num_device; i++) {
    for (int j = 0; j < num_device; j++) {
      ofs << std::setw(8) << std::fixed << std::setprecision(2) << topo_info.bandwidth_matrix[i][j] << " ";
    }
    ofs << "\n";
  }
  ofs.close();
  exit(0);
}

void PartitionSolver::LoadTopoFromFile(std::ifstream &ifs) {
  LinkTopoInfo universal_topo_info;

  std::string line;
  std::getline(ifs, line);
  std::smatch num_device_match;
  CHECK(std::regex_search(line, num_device_match,std::regex{"GPU Count ([0-9]+)"})) << "cannot determine #GPU";
  int num_device = std::stoi(num_device_match[1].str());
  while (std::getline(ifs, line)) {
    if (line.find(std::string{"P2P Matrix"}) != std::string::npos) {
      for (int i= 0; i < num_device; i++) {
        std::getline(ifs, line);
        std::stringstream ss(line, std::ios::in);
        for (int j = 0; j < num_device; j++) {
          ss >> universal_topo_info.nvlink_matrix[i][j];
        }
      }
    } else if (line.find(std::string{"Bandwidth Matrix"}) != std::string::npos) {
      for (int i = 0; i < num_device; i++) {
        std::getline(ifs, line);
        std::stringstream ss(line, std::ios::in);
        for (int j = 0; j < num_device; j++) {
          ss >> universal_topo_info.bandwidth_matrix[i][j];
        }
      }
    }
    if (ifs.eof())
      break;
  }

  auto visiable_device = std::getenv("CUDA_VISIBLE_DEVICES");
  if (visiable_device == nullptr) {
    std::memcpy(&_topo_info, &universal_topo_info, sizeof(LinkTopoInfo));
    return;
  }
  std::smatch d_match;
  std::string visiable_device_str{visiable_device};
  std::vector<int> devices;
  while(std::regex_search(visiable_device_str, d_match, std::regex{"[0-9]+"})) {
    devices.push_back(std::stoi(d_match[0].str()));
    visiable_device_str = d_match.suffix();
  }
  CHECK(devices.size() > 0) << "cannot find device in CUDA_VISIBLE_DEVICES";

  for (int i = 0; i < _ctxes.size(); i++) {
    int device = devices[i];
    for (int j = 0; j < _ctxes.size(); j++) {
      int peer = devices[j];
      _topo_info.nvlink_matrix[i][j] = universal_topo_info.nvlink_matrix[device][peer];
      _topo_info.bandwidth_matrix[i][j] = universal_topo_info.bandwidth_matrix[device][peer];
    }
  }
}

std::ostream& operator<<(std::ostream &os, const DistGraph::GroupConfig &config) {
  std::stringstream part_ss;
  std::stringstream peer_ss;
  for (auto part : config.part_ids)
    part_ss << part << " ";
  for (auto &ctx : config.ctx_group)
    peer_ss << ctx << " ";
  os << "GPU[" << config.ctx.device_id << "]"
     << " part: [ " << part_ss.str() << "]"
     << " peer: [ " << peer_ss.str() << "]";
  return os;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
