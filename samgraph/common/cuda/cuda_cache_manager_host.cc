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

#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cuda_cache_manager.h"
#include "ics22_song_dist_feature.h"
#include "../profiler.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

/** cache is only used for feature, so we can do mock here */
template <typename T>
void extract_miss_data(void *output_miss, const IdType *miss_src_index,
                       const size_t num_miss, const void *src, size_t dim) {
  T *output_miss_data = reinterpret_cast<T *>(output_miss);
  const T *cpu_src_data = reinterpret_cast<const T *>(src);
  size_t idx_mock_mask = 0xffffffffffffffff;
  if (RunConfig::option_empty_feat != 0) {
    idx_mock_mask = (1ull << RunConfig::option_empty_feat) - 1;
  }

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_miss; i++) {
    size_t src_idx = miss_src_index[i] & idx_mock_mask;
#pragma omp simd
    for (size_t j = 0; j < dim; j++) {
      output_miss_data[i * dim + j] = cpu_src_data[src_idx * dim + j];
    }
  }
}

}  // namespace

GPUCacheManager::GPUCacheManager(Context sampler_ctx, Context trainer_ctx,
                                 const void *cpu_src_data, DataType dtype,
                                 size_t dim, const IdType *nodes,
                                 size_t num_nodes, double cache_percentage)
    : _sampler_ctx(sampler_ctx),
      _trainer_ctx(trainer_ctx),
      _num_nodes(num_nodes),
      _num_cached_nodes(num_nodes * cache_percentage),
      _cache_percentage(cache_percentage),
      _dtype(dtype),
      _dim(dim),
      _cpu_src_data(cpu_src_data){
  Timer t;

  _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});

  auto cpu_device = Device::Get(CPU());
  auto sampler_gpu_device = Device::Get(_sampler_ctx);
  auto trainer_gpu_device = Device::Get(_trainer_ctx);

  IdType *tmp_cpu_hashtable = static_cast<IdType *>(
      cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
  _sampler_gpu_hashtable =
      static_cast<IdType *>(sampler_gpu_device->AllocDataSpace(
          _sampler_ctx, sizeof(IdType) * _num_nodes));

  void *tmp_cpu_data = cpu_device->AllocDataSpace(CPU(), _cache_nbytes);
  _trainer_cache_data =
      trainer_gpu_device->AllocDataSpace(_trainer_ctx, _cache_nbytes);
  Profiler::Get().LogInit(kLogInitL1FeatMemory, _cache_nbytes);

  // 1. Initialize the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    tmp_cpu_hashtable[i] = Constant::kEmptyKey;
  }

  // 2. Populate the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_cached_nodes; i++) {
    tmp_cpu_hashtable[nodes[i]] = i;
  }

  // 3. Populate the cache in cpu memory
  if (RunConfig::option_empty_feat != 0) {
    cpu::CPUMockExtract(tmp_cpu_data, _cpu_src_data, nodes, _num_cached_nodes,
                        _dim, _dtype);
  } else {
    cpu::CPUExtract(tmp_cpu_data, _cpu_src_data, nodes, _num_cached_nodes, _dim,
                    _dtype);
  }

  // 4. Copy the cache from the cpu memory to gpu memory
  sampler_gpu_device->CopyDataFromTo(
      tmp_cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
      sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx);
  trainer_gpu_device->CopyDataFromTo(tmp_cpu_data, 0, _trainer_cache_data, 0,
                                     _cache_nbytes, CPU(), _trainer_ctx);

  // 5. Free the cpu tmp cache data
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_hashtable);
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_data);

  LOG(INFO) << "GPU cache (policy: " << RunConfig::cache_policy
            << ") " << _num_cached_nodes << " / " << _num_nodes << " nodes ( "
            << ToPercentage(_cache_percentage) << " | "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs )";
}

GPUCacheManager::GPUCacheManager(IdType worker_id,
                                 Context sampler_ctx, Context trainer_ctx,
                                 const void* cpu_src_data, DataType dtype, size_t dim,
                                 const IdType* nodes, size_t num_nodes,
                                 double cache_percentage,
                                 DistGraph *dist_graph)
  : _sampler_ctx(sampler_ctx),
    _trainer_ctx(trainer_ctx),
    _num_nodes(num_nodes),
    _num_cached_nodes(num_nodes * cache_percentage),
    _cache_percentage(cache_percentage),
    _dtype(dtype),
    _dim(dim),
    _cpu_src_data(cpu_src_data),
    _trainer_cache_data(nullptr),
    _dist_graph(dist_graph)
{
  Timer t;

  if (RunConfig::run_arch == kArch6) {
    CHECK(static_cast<int>(worker_id) == sampler_ctx.device_id);
    CHECK(static_cast<int>(worker_id) == trainer_ctx.device_id);
  }

  auto cpu_device = Device::Get(CPU());
  auto sampler_device = Device::Get(sampler_ctx);

  IdType *part_cache_rank_node = cpu_device->AllocArray<IdType>(
      CPU(), sizeof(IdType) * _num_nodes);
  IdType real_num_cached_node = _num_cached_nodes;
  // each worker see same ranking nodes
  if (RunConfig::use_ics22_song_solver == false) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < _num_nodes; i++) {
      part_cache_rank_node[i] = nodes[i];
    }
    std::mt19937 eg(real_num_cached_node);
    std::shuffle(part_cache_rank_node,
        part_cache_rank_node + real_num_cached_node, eg);
  } else {
    CHECK(RunConfig::cache_policy == CachePolicy::kCacheByDegree)
      << "ics22_song_solver only support degree at present platform";
    LOG(DEBUG) << "GPUCacheManager: use ics22_song_solver";
    auto ics22_dist_graph = dynamic_cast<ICS22SongDistGraph*>(_dist_graph);
    CHECK(ics22_dist_graph != nullptr);
    auto ranking_node = ics22_dist_graph->GetRankingNode()->CPtr<IdType>();
    real_num_cached_node = ics22_dist_graph->GetRealCachedNodeNum();
    std::cout << "num_cached_nodes vs real_num_cached_node: "
      << _num_cached_nodes << " " << real_num_cached_node << std::endl;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < real_num_cached_node; i++) {
      // std::cout << "rank node " << i << " " << ranking_node[i] << std::endl;
      part_cache_rank_node[i] = ranking_node[i];
    }
  }
  LOG(DEBUG) << "use partition cache, collective cache percent " << _cache_percentage * 100 << "%"
            << " num cached nodes " << real_num_cached_node;

  IdType *tmp_cpu_hashtable = static_cast<IdType *>(
      cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
  _sampler_gpu_hashtable =
      static_cast<IdType *>(sampler_device->AllocDataSpace(
          _sampler_ctx, sizeof(IdType) * _num_nodes));

  // 1. Initialize the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    tmp_cpu_hashtable[i] = Constant::kEmptyKey;
  }
  // 2. Populate the cpu hashtable
  if (RunConfig::use_ics22_song_solver && RunConfig::ics22_compact_mode) {
    auto ics22_dist_graph = dynamic_cast<ICS22SongDistGraph*>(_dist_graph);
    CHECK(ics22_dist_graph != nullptr);
    CHECK(ics22_dist_graph->GetIdxMap(sampler_ctx)->Ctx().device_type
        == DeviceType::kCPU);
    CHECK(ics22_dist_graph->GetDeviceMap(sampler_ctx)->Ctx().device_type
        == DeviceType::kCPU);
    auto idx_map_data = ics22_dist_graph->GetIdxMap(sampler_ctx)
                          ->CPtr<IdType>();
    auto device_map_data = ics22_dist_graph->GetDeviceMap(sampler_ctx)
                              ->CPtr<IdType>();
    IdType device_mask = (1 << RunConfig::ics22_compact_bitwidth) - 1;
    IdType shift_width = (sizeof(IdType) * 8
                            - RunConfig::ics22_compact_bitwidth);
    CHECK((real_num_cached_node & (device_mask << shift_width)) == 0);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < real_num_cached_node; i++) {
      IdType tmp_node = part_cache_rank_node[i];
      tmp_cpu_hashtable[tmp_node] = ((device_map_data[tmp_node] << shift_width)
                                     | idx_map_data[tmp_node]);
    }
  } else {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < real_num_cached_node; i++) {
      tmp_cpu_hashtable[part_cache_rank_node[i]] = i;
    }
  }

  // 3. Load the feature cache
  _dist_graph->FeatureLoad(worker_id, trainer_ctx,
      part_cache_rank_node, _num_cached_nodes,
      _dtype, _dim,
      cpu_src_data);

  // 4. Copy the cache hashtable from the cpu memory to gpu memory
  sampler_device->CopyDataFromTo(
      tmp_cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
      sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx);

  // 5. Free the cpu tmp cache data
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_hashtable);
  cpu_device->FreeWorkspace(CPU(), part_cache_rank_node);


  _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});
  LOG(INFO) << "Partition GPU cache (policy: " << RunConfig::cache_policy << ") "
            << "global cache: "
            << _num_cached_nodes << " / " << _num_nodes << " nodes ( "
            << ToPercentage(_cache_percentage) << " ) "
            << ToReadableSize(_cache_nbytes) << " | "
            << t.Passed() << " secs";
}

GPUCacheManager::~GPUCacheManager() {
  auto sampler_device = Device::Get(_sampler_ctx);
  auto trainer_device = Device::Get(_trainer_ctx);

  if (_sampler_gpu_hashtable != nullptr) {
    sampler_device->FreeDataSpace(_sampler_ctx, _sampler_gpu_hashtable);
  }
  if (_trainer_cache_data != nullptr) {
    trainer_device->FreeDataSpace(_trainer_ctx, _trainer_cache_data);
  }
}

void GPUCacheManager::ExtractMissData(void *output_miss,
                                      const IdType *miss_src_index,
                                      const size_t num_miss) {
  if (num_miss == 0) return;
  switch (_dtype) {
    case kF32:
      extract_miss_data<float>(output_miss, miss_src_index, num_miss,
                               _cpu_src_data, _dim);
      break;
    case kF64:
      extract_miss_data<double>(output_miss, miss_src_index, num_miss,
                                _cpu_src_data, _dim);
      break;
    case kF16:
      extract_miss_data<short>(output_miss, miss_src_index, num_miss,
                               _cpu_src_data, _dim);
      break;
    case kU8:
      extract_miss_data<uint8_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    case kI32:
      extract_miss_data<int32_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    case kI64:
      extract_miss_data<int64_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    default:
      CHECK(0);
  }
}

GPUDynamicCacheManager::GPUDynamicCacheManager(Context sampler_ctx, Context trainer_ctx,
                                 const void *cpu_src_data,
                                 DataType dtype,
                                 size_t dim,
                                //  const IdType *nodes,
                                 size_t num_nodes
                                //  , double cache_percentage
                                 )
    : _sampler_ctx(sampler_ctx),
      _trainer_ctx(trainer_ctx),
      _num_nodes(num_nodes),
      // _num_cached_nodes(num_nodes * cache_percentage),
      // _cache_percentage(cache_percentage),
      _dtype(dtype),
      _dim(dim),
      _cpu_src_data(cpu_src_data) {

  auto sampler_gpu_device = Device::Get(_sampler_ctx);
  auto cpu_device = Device::Get(CPU());
  _sampler_gpu_hashtable =
      static_cast<IdType *>(sampler_gpu_device->AllocDataSpace(
          _sampler_ctx, sizeof(IdType) * _num_nodes));
  _cpu_hashtable = static_cast<IdType *>(
      cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    _cpu_hashtable[i] = Constant::kEmptyKey;
  }
  sampler_gpu_device->CopyDataFromTo(
      _cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
      sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx);
}
/**
 * @brief
 *
 * @param nodes must be in cpu
 * @param features must be in train gpu
 */
void GPUDynamicCacheManager::ReplaceCache(TensorPtr nodes, TensorPtr features) {
  Timer t;

  // _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});

  // auto cpu_device = Device::Get(CPU());
  auto sampler_gpu_device = Device::Get(_sampler_ctx);
  // auto trainer_gpu_device = Device::Get(_trainer_ctx);

  // IdType *tmp_cpu_hashtable = static_cast<IdType *>(
  //     cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
  // void *tmp_cpu_data = cpu_device->AllocDataSpace(CPU(), _cache_nbytes);

  _trainer_cache_data = features;
  _cached_nodes = nodes;

  // 1. Initialize the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    _cpu_hashtable[i] = Constant::kEmptyKey;
  }

  const IdType* const cached_nodes = static_cast<const IdType*>(nodes->Data());
  // 2. Populate the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < features->Shape()[0]; i++) {
    _cpu_hashtable[cached_nodes[i]] = i;
  }

  // 3. Populate the cache in cpu memory
  // cpu::CPUExtract(tmp_cpu_data, _cpu_src_data, nodes, _num_cached_nodes, _dim,
  //                 _dtype);

  // 4. Copy the cache from the cpu memory to gpu memory
  sampler_gpu_device->CopyDataFromTo(
      _cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
      sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx);
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

GPUDynamicCacheManager::~GPUDynamicCacheManager() {
  auto sampler_device = Device::Get(_sampler_ctx);
  auto cpu_device = Device::Get(CPU());
  // auto trainer_device = Device::Get(_trainer_ctx);

  sampler_device->FreeDataSpace(_sampler_ctx, _sampler_gpu_hashtable);
  cpu_device->FreeDataSpace(CPU(), _cpu_hashtable);
  // trainer_device->FreeDataSpace(_trainer_ctx, _trainer_cache_data);
}

void GPUDynamicCacheManager::ExtractMissData(void *output_miss,
                                      const IdType *miss_src_index,
                                      const size_t num_miss) {
  switch (_dtype) {
    case kF32:
      extract_miss_data<float>(output_miss, miss_src_index, num_miss,
                               _cpu_src_data, _dim);
      break;
    case kF64:
      extract_miss_data<double>(output_miss, miss_src_index, num_miss,
                                _cpu_src_data, _dim);
      break;
    case kF16:
      extract_miss_data<short>(output_miss, miss_src_index, num_miss,
                               _cpu_src_data, _dim);
      break;
    case kU8:
      extract_miss_data<uint8_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    case kI32:
      extract_miss_data<int32_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    case kI64:
      extract_miss_data<int64_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    default:
      CHECK(0);
  }
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
