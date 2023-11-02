#ifdef __GNUC__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif


#include "../timer.h"
#include "../run_config.h"
#include "../function.h"
#include "../device.h"

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
    const DataType dtype,
    std::vector<TensorPtr> &device_map_vec,
    std::vector<TensorPtr> &new_idx_map_vec,
    std::vector<TensorPtr> &device_cached_nodes_vec,
    TensorPtr &ranking_tensor,
    IdType &total_cached_node) {

  device_map_vec.clear();
  new_idx_map_vec.clear();
  device_cached_nodes_vec.clear();
  device_map_vec.resize(ctxes.size(), nullptr);
  new_idx_map_vec.resize(ctxes.size(), nullptr);
  device_cached_nodes_vec.resize(ctxes.size(), nullptr);
  IdType num_node = sample_prob.size();
  Context cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  using SortType = std::pair<double, IdType>; // first: weight
                                              // second : id
  std::vector<SortType> sample_idx_vec;
  sample_idx_vec.reserve(num_node);
  for (IdType i = 0; i < num_node; ++i) {
    sample_idx_vec.emplace_back(sample_prob[i], i);
  }
  Timer t;
#ifdef __GNUC__
  __gnu_parallel::sort(sample_idx_vec.begin(), sample_idx_vec.end(),
      std::greater<SortType>());
#else
  std::sort(sample_idx_vec.begin(), sample_idx_vec.end(),
      std::greater<SortType>());
#endif
  LOG(DEBUG) << "Sort sample_prob time cost: " << t.Passed() << " sec.";
  TensorPtr origin_new_idx_tensor = Tensor::Empty(dtype, {num_node},
      cpu_ctx, "");
  TensorPtr origin_cached_tensor = Tensor::Empty(dtype, {num_cached_node},
      cpu_ctx, "");
  auto origin_new_idx_tensor_data = origin_new_idx_tensor->Ptr<IdType>();
  auto ranking_tensor_data = ranking_tensor->Ptr<IdType>();
  auto origin_cached_tensor_data = origin_cached_tensor->Ptr<IdType>();
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (IdType i = 0; i < num_node; ++i) {
    origin_new_idx_tensor_data[i] = Constant::kEmptyKey;
    ranking_tensor_data[i] = std::get<1>(sample_idx_vec[i]);
  }
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (IdType i = 0; i < num_cached_node; ++i) {
    origin_new_idx_tensor_data[std::get<1>(sample_idx_vec[i])] = i;
    origin_cached_tensor_data[i] = std::get<1>(sample_idx_vec[i]);
  }
  LOG(DEBUG) << "Initialized origin_new_idx_tensor with num_node=" << num_node
    << ", origin_cached_tensor with num_cached_node=" << num_cached_node;
  for (const Context &ctx : ctxes) {
    int device_id = ctx.device_id;
    CHECK(device_id < static_cast<int>(ctxes.size()));
    new_idx_map_vec[device_id] = Tensor::CopyTo(origin_new_idx_tensor, cpu_ctx,
        nullptr, "new index map of nodes in device "
          + std::to_string(device_id));
    device_cached_nodes_vec[device_id] = Tensor::CopyTo(origin_cached_tensor,
        cpu_ctx, nullptr, "cached nodes in device " + std::to_string(device_id));
    device_map_vec[device_id] = Tensor::Empty(dtype, {num_node},
        cpu_ctx, "device map of nodes in device " + std::to_string(device_id));
    auto device_map = device_map_vec[device_id]->Ptr<IdType>();
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (IdType i = 0; i < num_node; ++i) {
      device_map[i] = Constant::kEmptyKey;
    }
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (IdType i = 0; i < num_cached_node; ++i) {
      device_map[std::get<1>(sample_idx_vec[i])] = device_id;
    }
  }
  total_cached_node = num_cached_node;
  if (clique_size == 1) return;
  std::vector<double> p_accum(clique_size, 0.0);
  auto _GetDeviceOrder = [](const std::vector<double> &p_accum) {
    std::vector<SortType> tmp_vec;
    tmp_vec.reserve(p_accum.size());
    for (IdType i = 0; i < p_accum.size(); ++i) {
      tmp_vec.emplace_back(p_accum[i], i);
    }
    std::sort(tmp_vec.begin(), tmp_vec.end());
    std::vector<IdType> ret(p_accum.size());
    for (IdType i = 0; i < p_accum.size(); ++i) {
      ret[i] = std::get<1>(tmp_vec[i]);
    }
    return ret;
  };
  std::vector<IdType> device_idx_order = _GetDeviceOrder(p_accum);
  IdType last_i = (num_node - num_cached_node);
  IdType num_cliques = (ctxes.size() / clique_size);
  LOG(DEBUG) << "Start to replacement with num_cliques=" << num_cliques;
  {
    IdType i = 0;
    for (; i < last_i; ++i) {
      if (i % (clique_size - 1) == 0) {
        device_idx_order = _GetDeviceOrder(p_accum);
      }
      IdType candidate_node = std::get<1>(sample_idx_vec[num_cached_node + i]);
      IdType new_node_idx = num_cached_node - 1 - (i / (clique_size - 1));
      if (new_node_idx < 0 || new_node_idx >= num_cached_node) {
        break;
      }
      IdType node_to_be_replaced = std::get<1>(sample_idx_vec[new_node_idx]);
      if (sample_prob[candidate_node] >= alpha * sample_prob[node_to_be_replaced]) {
        IdType cur_dev_idx = device_idx_order[i % (clique_size - 1)];
        p_accum[cur_dev_idx] += sample_prob[candidate_node];
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
              = ctxes[j * clique_size + device_idx_order.back()].device_id;
          device_cached_nodes_vec[ctxes[j * clique_size + cur_dev_idx].device_id]
            ->Ptr<IdType>()[new_node_idx] = candidate_node;
        }
      } else {
        break;
      }
    }
    std::cout << "i: " << i << std::endl;
    total_cached_node = num_cached_node + i;
    /*
    for (auto ctx : ctxes) {
      auto device_id = ctx.device_id;
      double sample_prob_total = 0.0;
      auto device_cached_nodes_data = device_cached_nodes_vec[device_id]->CPtr<IdType>();
      for (IdType j = 0; j < total_cached_node; ++j) {
        sample_prob_total += sample_prob[device_cached_nodes_data[j]];
      }
      std::cout << "device " << device_id << ": " << sample_prob_total << std::endl;
    }
    */
  }
};

} // namespace

ICS22SongDistGraph::ICS22SongDistGraph(std::vector<Context> ctxes,
    IdType clique_size,
    const Dataset *dataset,
    const double alpha,
    IdType num_feature_cached_node)
  : DistGraph(ctxes),
    _clique_size(clique_size) {
  std::stringstream ss;
  ss << "[ ";
  for (const auto &ctx : ctxes) {
    ss << ctx.device_id << " ";
  }
  ss << "]";
  LOG(INFO) << "Initial ICS22SongDistGraph with: devices=" << ss.str()
    << ", clique_size=" << clique_size
    << ", alpha=" << alpha
    << ", num_feature_cached_node=" << num_feature_cached_node;
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
  TensorPtr indices = dataset->indices;
  auto indices_data = indices->CPtr<IdType>();
  DataType dtype = indices->Type();
  size_t num_edge = dataset->num_edge;
  _num_node = dataset->num_node;
  std::vector<std::vector<IdType>> degree_per_thread(
      RunConfig::omp_thread_num, std::vector<IdType>(_num_node, 0));
  std::vector<double> sample_prob(_num_node, 0.0f);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_edge; ++i) {
    auto thread_idx = omp_get_thread_num();
    degree_per_thread[thread_idx][indices_data[i]] += 1;
  }
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_node; ++i) {
    for (size_t j = 0; j < RunConfig::omp_thread_num; ++j) {
      sample_prob[i] += degree_per_thread[j][i];
    }
  }
  _ranking_node_tensor = Tensor::Empty(dtype, {_num_node},
      CPU(CPU_CLIB_MALLOC_DEVICE), "ranking node tensor");
  ICS22SongPlacementSolver(sample_prob, num_feature_cached_node, ctxes,
      alpha, clique_size, dtype, _h_device_map_vec, _h_new_idx_map_vec,
      _h_device_cached_nodes_vec, _ranking_node_tensor, _total_cached_node);
  LOG(DEBUG) << "ICS22SongDistGraph initialized!";
};

void ICS22SongDistGraph::FeatureLoad(int trainer_id, Context trainer_ctx,
    const IdType *cache_rank_node, const IdType num_cache_node,
    DataType dtype, size_t dim,
    const void* cpu_src_feature_data,
    StreamHandle stream) {

  LOG(DEBUG) << "ICS22SongDistGraph: start to load feature data";
  CHECK(trainer_ctx.device_id == trainer_id);
  _trainer_id = trainer_id;
  _feat_dim = dim;

  IdType num_cached_node = _h_device_cached_nodes_vec[trainer_id]->Shape()[0];
  CHECK(num_cache_node == num_cached_node) << "check num_cache_node";
  auto cached_node_data
         = _h_device_cached_nodes_vec[trainer_id]->CPtr<IdType>();
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
  _part_feature.resize(_ctxes.size(), Tensor::Null());
  _part_feature[trainer_id] = trainer_feat_tensor;
  { // IPC share pointers
    {
      auto feature_data = _part_feature[trainer_id]->MutableData();
      cudaIpcMemHandle_t &mem_handle =
        _shared_data->mem_handle[trainer_id][0];
      CUDA_CALL(cudaIpcGetMemHandle(&mem_handle, (void*)feature_data));
    }
    _Barrier();
    IdType start_j = (trainer_id / _clique_size * _clique_size);
    IdType end_j = start_j + _clique_size;
    for (IdType j = start_j; j < end_j; ++j) {
      auto ctx = _ctxes[j];
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
  if (RunConfig::ics22_compact_mode == false) {
    _d_device_map_tensor = Tensor::CopyTo(_h_device_map_vec[trainer_id],
        trainer_ctx, stream, Constant::kAllocNoScale);
    _d_new_idx_map_tensor = Tensor::CopyTo(_h_new_idx_map_vec[trainer_id],
        trainer_ctx, stream, Constant::kAllocNoScale);
  } else {
    _d_new_idx_map_tensor = nullptr;
    _d_device_map_tensor = nullptr;
  }
  LOG(DEBUG) << "ICS22SongDistGraph: load feature data successfully";
}

DeviceICS22SongDistFeature ICS22SongDistGraph::DeviceFeatureHandle() const {
  CHECK(_feat_dim != 0);
  if (RunConfig::ics22_compact_mode == true) {
    return DeviceICS22SongDistFeature(
        _d_part_feature,
        nullptr,
        nullptr,
        _num_node,
        _feat_dim,
        RunConfig::ics22_compact_bitwidth);
  }
  return DeviceICS22SongDistFeature(
      _d_part_feature,
      _d_device_map_tensor->Ptr<IdType>(),
      _d_new_idx_map_tensor->Ptr<IdType>(),
      _num_node,
      _feat_dim);
}

void ICS22SongDistGraph::Release(DistGraph *dist_graph) {
  dist_graph->Release(dist_graph);
}

void ICS22SongDistGraph::Create(std::vector<Context> ctxes,
    IdType clique_size,
    const Dataset *dataset,
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
