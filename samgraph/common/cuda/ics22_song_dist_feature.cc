#include <algorithm>

#include "../run_config.h"
#include "../function.h"

#include "dist_graph.h"
#include "ics22_song_dist_feature.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

void ICS22SongPlacementSolver(const std::vector<double> &sample_prob,
    const IdType num_cached_node,
    const std::vector<Context> &ctxes,
    const double alpha,
    const IdType clique_size,
    std::vector<TensorPtr> &device_map_vec,
    std::vector<TensorPtr> &new_idx_map_vec,
    std::vector<TensorPtr> &device_cached_nodes_vec) {

  device_map_vec.clear();
  new_idx_map_vec.clear();
  device_map_vec.resize(ctxes.size());
  new_idx_map_vec.resize(ctxes.size());
  IdType num_node = sample_prob.size();
  using SortType = std::pair<double, IdType>; // first: weight
                                              // second : id
  std::vector<SortType> sample_idx_vec;
  sample_idx_vec.reserve(num_node);
  for (IdType i = 0; i < num_node; ++i) {
    sample_idx_vec.emplace_back(sample_prob[i], i);
  }
  // XXX: speed up it
  std::sort(sample_idx_vec.begin(), sample_idx_vec.end(),
      [](const SortType &a, const SortType &b) {
        if (std::get<0>(a) != std::get<0>(b)) {
          return std::get<0>(a) > std::get<0>(b);
        }
        return std::get<1>(a) > std::get<1>(b);
      });
  TensorPtr origin_tensor = Tensor::Empty(DataType::kI32, {num_node},
      CPU(), "");
  auto origin_tensor_data = origin_tensor->Ptr<IdType>();
  TensorPtr cached_nodes_tensor = Tensor::Empty(DataType::kI32,
      {num_cached_node}, CPU(), "");
  auto cached_nodes_tensor_data = cached_nodes_tensor->Ptr<IdType>();
  for (IdType i = 0; i < num_node; ++i) {
    origin_tensor_data[i] = Constant::kEmptyKey;
  }
  for (IdType i = 0; i < num_cached_node; ++i) {
    origin_tensor_data[std::get<1>(sample_idx_vec[i])] = i;
    cached_nodes_tensor_data[i] = std::get<1>(sample_idx_vec[i]);
  }
  IdType num_cliques = (ctxes.size() / clique_size);
  for (const Context &ctx : ctxes) {
    int device_id = ctx.device_id;
    CHECK(device_id < static_cast<int>(device_map_vec.size()));
    new_idx_map_vec[device_id] = Tensor::CopyTo(origin_tensor, CPU(),
        nullptr, "new index map of nodes in device "
          + std::to_string(device_id));
    device_cached_nodes_vec[device_id] = Tensor::CopyTo(cached_nodes_tensor,
        CPU(), nullptr, "cached nodes in device " + std::to_string(device_id));
    device_map_vec[device_id] = Tensor::Empty(DataType::kI32, {num_node},
        CPU(), "device map of nodes in device " + std::to_string(device_id));
    auto device_map = device_map_vec[device_id]->Ptr<IdType>();
    for (IdType i = 0; i < num_node; ++i) {
      device_map[i] = Constant::kEmptyKey;
    }
    for (IdType i = 0; i < num_cached_node; ++i) {
      device_map[std::get<1>(sample_idx_vec[i])] = device_id;
    }
  }
  std::vector<SortType> p_accum(clique_size);
  for (IdType i = 0; i < clique_size; ++i) {
    p_accum[i] = {0.0, i};
  }
  auto _GetDeviceOrder = [&p_accum]() {
    std::sort(p_accum.begin(), p_accum.end());
    std::vector<IdType> ret(p_accum.size());
    for (IdType i = 0; i < p_accum.size(); ++i) {
      ret[i] = std::get<1>(p_accum[i]);
    }
    return ret;
  };
  std::vector<IdType> device_idx_order = _GetDeviceOrder();
  IdType last_i = (num_node - num_cached_node);
  for (IdType i = 0; i < last_i; ++i) {
    if (i % (clique_size - 1) == 0) {
      device_idx_order = _GetDeviceOrder();
    }
    IdType candidate_node = std::get<1>(sample_idx_vec[num_cached_node + i]);
    IdType new_node_idx = num_cached_node - 1 - (i / (clique_size - 1));
    IdType node_to_be_replaced = std::get<1>(sample_idx_vec[new_node_idx]);
    if (sample_prob[candidate_node] >= alpha * sample_prob[node_to_be_replaced]) {
      IdType cur_dev_idx = device_idx_order[i % (clique_size - 1)];
      std::get<0>(p_accum[cur_dev_idx]) += sample_prob[candidate_node];
      // for each clique
      for (IdType j = 0; j < num_cliques; ++j) {
        for (IdType k = 0; k < clique_size; ++k) {
          device_map_vec[ctxes[j * clique_size + k].device_id]
            ->Ptr<IdType>()[candidate_node]
              = ctxes[j * clique_size + cur_dev_idx].device_id;
          new_idx_map_vec[ctxes[j * clique_size + k].device_id]
            ->Ptr<IdType>()[candidate_node]
              = new_node_idx;
        }
        device_map_vec[ctxes[j * clique_size + cur_dev_idx].device_id]
          ->Ptr<IdType>()[node_to_be_replaced]
            = ctxes[(j + 1) * clique_size - 1].device_id;
        device_cached_nodes_vec[ctxes[j * clique_size + cur_dev_idx].device_id]
          ->Ptr<IdType>()[new_node_idx] = candidate_node;
      }
    } else {
      break;
    }
  }
};

} // namespace

ICS22SongDistGraph::ICS22SongDistGraph(std::vector<Context> ctxes,
    IdType clique_size,
    Dataset *dataset,
    const double alpha,
    IdType num_feature_cached_node)
  : DistGraph(ctxes) {
  _ctxes = ctxes;
  _num_feature_cache_node = num_feature_cached_node;
  auto nvlink_matrix = PartitionSolver(ctxes).GetLinkTopoInfo()->nvlink_matrix;
  IdType num_cliques = (ctxes.size() / clique_size);
  for (IdType i = 0; i < num_cliques; ++i) {
    for (IdType j = 0; j < clique_size; ++j) {
      for (IdType k = 0; k < clique_size; ++k) {
        CHECK(nvlink_matrix[i * clique_size + j][i * clique_size + k])
          << "clique needs full-connected";
      }
    }
  }
  TensorPtr indptr = dataset->indptr;
  auto indptr_data = indptr->CPtr<IdType>();
  IdType num_node = dataset->num_node;
  std::vector<double> sample_prob(num_node);
  for (IdType i = 0; i < num_node; ++i) {
    sample_prob[i] = (indptr_data[i + 1] - indptr_data[i]);
  }
  ICS22SongPlacementSolver(sample_prob, num_feature_cached_node, ctxes,
      alpha, clique_size, _h_device_map_vec, _h_new_idx_map_vec,
      _h_device_cached_nodes_vec);
  _d_device_map = nullptr;
  _d_new_idx_map = nullptr;
  LOG(DEBUG) << "ICS22SongDistGraph initialized!";
};

void ICS22SongDistGraph::FeatureLoad(int trainer_id, Context trainer_ctx,
    const IdType *cache_rank_node, const IdType num_cache_node,
    DataType dtype, size_t dim,
    const void* cpu_src_feature_data,
    StreamHandle stream) {

  CHECK(trainer_ctx.device_id == trainer_id);
  CHECK(num_cache_node == 0) << "num_cache_node is not used";
  _trainer_id = trainer_id;
  _feat_dim = dim;

  IdType num_cached_node = _h_device_cached_nodes_vec[trainer_id]->Shape()[0];
  auto cached_node_data
         = _h_device_cached_nodes_vec[trainer_id]->CPtr<IdType>();
  _part_feature.resize(_ctxes.size(), nullptr);
  auto tmp_cpu_feature_tensor = Tensor::Empty(dtype,
      {num_cached_node, _feat_dim}, CPU(),
      "feature cache in device " + std::to_string(trainer_id));
  void* tmp_cpu_feature_data = tmp_cpu_feature_tensor->MutableData();

  // Populate the cache in cpu memory
  if (RunConfig::option_empty_feat != 0) {
    cpu::CPUMockExtract(tmp_cpu_feature_data, cpu_src_feature_data,
        cached_node_data, num_cached_node, _feat_dim, dtype);
  } else {
    cpu::CPUExtract(tmp_cpu_feature_data, cpu_src_feature_data,
        cached_node_data, num_cached_node, _feat_dim, dtype);
  }
  TensorPtr trainer_feat_tensor = Tensor::CopyTo(tmp_cpu_feature_tensor,
      trainer_ctx, stream, Constant::kAllocNoScale);
  CHECK(trainer_ctx.device_type == DeviceType::kGPU)
      << "Trainer context should be GPU";
  _part_feature.resize(_ctxes.size(), nullptr);
  _part_feature[trainer_id] = trainer_feat_tensor;
  { // IPC share pointers
    {
      auto feature_data = _part_feature[trainer_id]->MutableData();
      cudaIpcMemHandle_t &mem_handle =
        _shared_data->mem_handle[trainer_id][0];
      CUDA_CALL(cudaIpcGetMemHandle(&mem_handle, (void*)feature_data));
    }
    _Barrier();
    for (auto ctx : _ctxes) {
      if (ctx.device_id == trainer_id) {
        continue;
      }
      cudaIpcMemHandle_t &mem_handle
                          = _shared_data->mem_handle[ctx.device_id][0];
      void *ptr;
      CUDA_CALL(cudaIpcOpenMemHandle(
            &ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
      _part_feature[ctx.device_id] = Tensor::FromBlob(
          ptr, dtype, {num_cached_node, _feat_dim}, ctx,
          "cached feature data in device " + std::to_string(ctx.device_id));
    }
    _Barrier();
  }
  // move pointer array to GPU
  CUDA_CALL(cudaMalloc((void**)&_d_part_feature,
        _ctxes.size() * sizeof(void*)));

  void* h_part_feature[_ctxes.size()];
  for (IdType i = 0; i < _ctxes.size(); ++i) {
    h_part_feature[i] = _part_feature[i]->MutableData();
  }
  CUDA_CALL(cudaMemcpy(_d_part_feature, h_part_feature,
        sizeof(void*) * _ctxes.size(),
        cudaMemcpyDefault));
  // move map structure to GPU
  CUDA_CALL(cudaMalloc((void**)&_d_device_map, sizeof(void*)));
  CUDA_CALL(cudaMemcpy(_d_device_map,
        _h_device_map_vec[trainer_id]->MutableData(),
        sizeof(void*), cudaMemcpyDefault));
  CUDA_CALL(cudaMalloc((void**)&_d_new_idx_map, sizeof(void*)));
  CUDA_CALL(cudaMemcpy(_d_new_idx_map,
        _h_new_idx_map_vec[trainer_id]->MutableData(),
        sizeof(void*), cudaMemcpyDefault));
}

DeviceICS22SongDistFeature ICS22SongDistGraph::DeviceFeatureHandle() const {
  CHECK(_feat_dim != 0);
  return DeviceICS22SongDistFeature(
      _d_part_feature,
      _d_device_map,
      _d_new_idx_map,
      _num_feature_cache_node,
      _feat_dim);
}

void ICS22SongDistGraph::Release(DistGraph *dist_graph) {
  dist_graph->Release(dist_graph);
}

void ICS22SongDistGraph::Create(std::vector<Context> ctxes,
    IdType clique_size,
    Dataset *dataset,
    const double alpha,
    IdType num_feature_cached_node) {
  CHECK(_inst == nullptr);
  _inst = std::shared_ptr<DistGraph>(
      new ICS22SongDistGraph(ctxes, clique_size, dataset, alpha, num_feature_cached_node),
      Release);
}

} // cuda
} // common
} // namespace samgraph
