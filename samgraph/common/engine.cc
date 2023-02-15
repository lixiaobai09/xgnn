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

#include "engine.h"

#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <parallel/algorithm>
#include <parallel/numeric>

#include "common.h"
#include "constant.h"
#include "cpu/cpu_engine.h"
#include "cuda/cuda_engine.h"
#include "cuda/um_pre_sampler.h"
#include "dist/dist_engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"
#include "device.h"

namespace samgraph {
namespace common {

namespace {

void shuffle(uint32_t * data, size_t num_data, uint64_t seed= 0x1234567890abcdef) {
  auto g = std::default_random_engine(seed);
  for (size_t i = num_data - 1; i > 0; i--) {
    std::uniform_int_distribution<size_t> d(0, i);
    size_t candidate = d(g);
    std::swap(data[i], data[candidate]);
  }
}

};

Engine* Engine::_engine = nullptr;

void Engine::Create() {
  if (_engine) {
    return;
  }

  switch (RunConfig::run_arch) {
    case kArch0:
      LOG(INFO) << "Use CPU Engine (Arch " << RunConfig::run_arch << ")";
      _engine = new cpu::CPUEngine();
      break;
    case kArch1:
    case kArch2:
    case kArch3:
    case kArch4:
    case kArch7:
      LOG(INFO) << "Use GPU Engine (Arch " << RunConfig::run_arch << ")";
      _engine = new cuda::GPUEngine();
      break;
    case kArch5:
    case kArch6:
    case kArch9:
      LOG(INFO) << "Use Dist Engine (Arch " << RunConfig::run_arch << ")";
      _engine = new dist::DistEngine();
      break;
    default:
      CHECK(0);
  }
}

TensorPtr ConverToAnonMmap(TensorPtr tensor) {
  size_t nbytes = tensor->NumBytes();
  int device_id = 0;
  if (RunConfig::option_huge_page) {
    size_t hugepage_size = (1l << 21);
    nbytes = (nbytes + hugepage_size - 1) / hugepage_size * hugepage_size;
    device_id = MMAP_HUGEPAGE;
  }
  auto data = mmap(NULL, nbytes,
      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
  std::memcpy(data, tensor->Data(), tensor->NumBytes());
  int ret_val = mprotect(data, nbytes, PROT_READ);
  CHECK(ret_val == 0);
  auto ret = Tensor::FromBlob(data, tensor->Type(), tensor->Shape(),
      MMAP(device_id), tensor->Name());
  return ret;
}

void Engine::LoadGraphDataset() {
  Timer t;
  // Load graph dataset from disk by mmap and copy the graph
  // topology data into the target CUDA device.
  _dataset = new Dataset;
  std::unordered_map<std::string, size_t> meta;
  std::unordered_map<std::string, Context> ctx_map = GetGraphFileCtx();

  // default feature type is 32-bit float.
  // legacy dataset doesnot have this meta
  DataType feat_data_type = kF32;

  if (_dataset_path.back() != '/') {
    _dataset_path.push_back('/');
  }

  // Parse the meta data
  std::ifstream meta_file(_dataset_path + Constant::kMetaFile);
  std::string line;
  while (std::getline(meta_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> kv{std::istream_iterator<std::string>{iss},
                                std::istream_iterator<std::string>{}};

    if (kv.size() < 2) {
      break;
    }

    if (kv[0] == Constant::kMetaFeatDataType) {
      feat_data_type = DataTypeParseName(kv[1]);
    } else {
      meta[kv[0]] = std::stoull(kv[1]);
    }
  }

  CHECK(meta.count(Constant::kMetaNumNode) > 0);
  CHECK(meta.count(Constant::kMetaNumEdge) > 0);
  CHECK(meta.count(Constant::kMetaFeatDim) > 0);
  CHECK(meta.count(Constant::kMetaNumClass) > 0);
  CHECK(meta.count(Constant::kMetaNumTrainSet) > 0);
  CHECK(meta.count(Constant::kMetaNumTestSet) > 0);
  CHECK(meta.count(Constant::kMetaNumValidSet) > 0);

  CHECK(ctx_map.count(Constant::kIndptrFile) > 0);
  CHECK(ctx_map.count(Constant::kIndicesFile) > 0);
  CHECK(ctx_map.count(Constant::kFeatFile) > 0);
  CHECK(ctx_map.count(Constant::kLabelFile) > 0);
  CHECK(ctx_map.count(Constant::kTrainSetFile) > 0);
  CHECK(ctx_map.count(Constant::kTestSetFile) > 0);
  CHECK(ctx_map.count(Constant::kValidSetFile) > 0);
  CHECK(ctx_map.count(Constant::kAliasTableFile) > 0);
  CHECK(ctx_map.count(Constant::kProbTableFile) > 0);
  CHECK(ctx_map.count(Constant::kInDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kOutDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByHeuristicFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByDegreeHopFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByFakeOptimalFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByRandomFile) > 0);

  _dataset->num_node = meta[Constant::kMetaNumNode];
  _dataset->num_edge = meta[Constant::kMetaNumEdge];
  _dataset->num_class = meta[Constant::kMetaNumClass];

  if (ctx_map[Constant::kIndptrFile].device_type != DeviceType::kGPU_UM) {
    _dataset->indptr =
        Tensor::FromMmap(_dataset_path + Constant::kIndptrFile, DataType::kI32,
                        {meta[Constant::kMetaNumNode] + 1},
                        ctx_map[Constant::kIndptrFile], "dataset.indptr");
    _dataset->indptr = ConverToAnonMmap(_dataset->indptr);
  } else {
    _dataset->indptr =
        Tensor::UMFromMmap(_dataset_path + Constant::kIndptrFile, DataType::kI32,
                          {meta[Constant::kMetaNumNode] + 1},
                          RunConfig::unified_memory_ctxes, "dataset.indptr");
  }
  if (ctx_map[Constant::kIndicesFile].device_type != DeviceType::kGPU_UM) {
    _dataset->indices =
        Tensor::FromMmap(_dataset_path + Constant::kIndicesFile, DataType::kI32,
                        {meta[Constant::kMetaNumEdge]},
                        ctx_map[Constant::kIndicesFile], "dataset.indices");
    _dataset->indices = ConverToAnonMmap(_dataset->indices);
  } else {
    _dataset->indices =
        Tensor::UMFromMmap(_dataset_path + Constant::kIndicesFile, DataType::kI32,
                          {meta[Constant::kMetaNumEdge]},
                          RunConfig::unified_memory_ctxes, "dataset.indices");
  }

  std::vector<size_t> empty_feat_shape;
  if (!FileExist(_dataset_path + Constant::kFeatFile) ||
      RunConfig::option_fake_feat_dim != 0 ||
      RunConfig::option_empty_feat != 0) {
    if (RunConfig::option_fake_feat_dim != 0) {
      meta[Constant::kMetaFeatDim] = RunConfig::option_fake_feat_dim;
      empty_feat_shape = {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]};
    } else if (RunConfig::option_empty_feat != 0) {
      empty_feat_shape =
          {1ull << RunConfig::option_empty_feat, meta[Constant::kMetaFeatDim]};
    } else {
      empty_feat_shape =
          {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]};
    }
  }

  double gpu_extract_time = 0.0;
  if (empty_feat_shape.size()) {
    if (RunConfig::gpu_extract) {
      Timer tt;
      size_t mmap_nbytes = GetTensorBytes(feat_data_type, empty_feat_shape);
      int mmap_device_id = 0;
      if (RunConfig::option_huge_page) {
        size_t hugepage_size = (1l << 21);
        mmap_nbytes = (mmap_nbytes + hugepage_size - 1) / hugepage_size * hugepage_size;
        mmap_device_id = MMAP_HUGEPAGE;
      }
      auto feat = mmap(NULL, mmap_nbytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
      CHECK_NE(feat, MAP_FAILED);
      int ret_val = mprotect(feat, mmap_nbytes, PROT_READ);
      CHECK(ret_val == 0);
      _dataset->feat = Tensor::FromBlob(feat, feat_data_type,
          empty_feat_shape, MMAP(mmap_device_id), "dataset.feat");
      gpu_extract_time += tt.Passed();
    } else {
      _dataset->feat = Tensor::EmptyNoScale(feat_data_type, empty_feat_shape,
          ctx_map[Constant::kFeatFile], "dataset.feat");
    }
  } else {
    _dataset->feat = Tensor::FromMmap(
        _dataset_path + Constant::kFeatFile, feat_data_type,
        {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
        ctx_map[Constant::kFeatFile], "dataset.feat");
    if (RunConfig::gpu_extract) {
      Timer tt;
      size_t mmap_nbytes = _dataset->feat->NumBytes();
      if (RunConfig::option_huge_page) {
        size_t hugepage_size = (1l << 21);
        mmap_nbytes = (mmap_nbytes + hugepage_size - 1) / hugepage_size * hugepage_size;
      }
      auto feat = mmap(NULL, mmap_nbytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
      CHECK_NE(feat, MAP_FAILED);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for (size_t i = 0; i < _dataset->feat->Shape()[0]; i++) {
        size_t nbytes = GetDataTypeBytes(_dataset->feat->Type()) * _dataset->feat->Shape()[1];
        size_t off = i * nbytes;
        std::memcpy(feat + off, _dataset->feat->Data() + off, nbytes);
      }
      int mmap_device_id = 0;
      if (RunConfig::option_huge_page) {
        mmap_device_id = MMAP_HUGEPAGE;
      }
      int ret_val = mprotect(feat, mmap_nbytes, PROT_READ);
      CHECK(ret_val == 0);
      _dataset->feat = Tensor::FromBlob(feat, _dataset->feat->Type(), _dataset->feat->Shape(), MMAP(mmap_device_id), _dataset->feat->Name());
      gpu_extract_time += tt.Passed();
    }
  }

  if (FileExist(_dataset_path + Constant::kLabelFile)) {
    _dataset->label =
        Tensor::FromMmap(_dataset_path + Constant::kLabelFile, DataType::kI64,
                         {meta[Constant::kMetaNumNode]},
                         ctx_map[Constant::kLabelFile], "dataset.label");
    if (RunConfig::gpu_extract) {
      Timer tt;
      auto label = mmap(NULL, _dataset->label->NumBytes(), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
      CHECK_NE(label, MAP_FAILED);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for (size_t i = 0; i < _dataset->label->Shape()[0]; i++) {
        size_t nbytes = GetDataTypeBytes(_dataset->label->Type());
        size_t off = i * nbytes;
        std::memcpy(label + off, _dataset->label->Data() + off, nbytes);
      }
      _dataset->label = Tensor::FromBlob(label, _dataset->label->Type(), _dataset->label->Shape(), MMAP(), _dataset->label->Name());
      gpu_extract_time += tt.Passed();
    }
  } else {
    if (RunConfig::gpu_extract) {
      Timer tt;
      auto nbytes = GetTensorBytes(DataType::kI64, {meta[Constant::kMetaNumNode]});
      auto label = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
      CHECK_NE(label, MAP_FAILED);
      _dataset->label = Tensor::FromBlob(label, DataType::kI64,
          {meta[Constant::kMetaNumNode]}, MMAP(), "dataset.label");
      gpu_extract_time += tt.Passed();
    } else {
      _dataset->label =
          Tensor::EmptyNoScale(DataType::kI64, {meta[Constant::kMetaNumNode]},
                               ctx_map[Constant::kLabelFile], "dataset.label");
    }
  }

  if (RunConfig::gpu_extract) {
    LOG(INFO) << "GPU Extract, load feature and label to shm, "
      << gpu_extract_time << " sec";
  }

  _dataset->train_set =
      Tensor::FromMmap(_dataset_path + Constant::kTrainSetFile, DataType::kI32,
                       {meta[Constant::kMetaNumTrainSet]},
                       ctx_map[Constant::kTrainSetFile], "dataset.train_set");
  _dataset->test_set =
      Tensor::FromMmap(_dataset_path + Constant::kTestSetFile, DataType::kI32,
                       {meta[Constant::kMetaNumTestSet]},
                       ctx_map[Constant::kTestSetFile], "dataset.test_set");
  _dataset->valid_set =
      Tensor::FromMmap(_dataset_path + Constant::kValidSetFile, DataType::kI32,
                       {meta[Constant::kMetaNumValidSet]},
                       ctx_map[Constant::kValidSetFile], "dataset.valid_set");

  if (RunConfig::option_train_set_slice_mode != "") {
    // first, create an writable copy of train set
    const uint32_t origin_num_train_set = meta[Constant::kMetaNumTrainSet];
    if (RunConfig::option_train_set_slice_mode == "percent" &&
        meta[Constant::kMetaNumNode] * RunConfig::option_train_set_percent / 100 > origin_num_train_set) {
      // expected train set exceeds original train set. so we should rebuild one from entire nodes.
      // degree cache file is a good choice...
      _dataset->train_set = Tensor::FromMmap(
          _dataset_path + Constant::kCacheByDegreeFile, DataType::kI32,
          {meta[Constant::kMetaNumNode]}, CPU(CPU_CLIB_MALLOC_DEVICE), "dataset.train_set");
    } else if (_dataset->train_set->Ctx().device_type != kCPU) {
      // the size of original train set meets our requirement.
      // but it is mapped and we cannot alter it, or it is in gpu.
      _dataset->train_set = Tensor::CopyTo(_dataset->train_set, CPU(CPU_CLIB_MALLOC_DEVICE));
    }

    // do a shuffle. because 1. original train set is sorted by id 2. cache file is also ranked
    // make sure train set is randomly spreaded.
    shuffle(static_cast<uint32_t*>(_dataset->train_set->MutableData()), _dataset->train_set->Shape()[0]);
    uint32_t begin = 0,end = 0;
    if (RunConfig::option_train_set_slice_mode == "percent") {
      end = meta[Constant::kMetaNumNode] * RunConfig::option_train_set_percent / 100;
    } else if (RunConfig::option_train_set_slice_mode == "part") {
      const uint32_t part_idx = RunConfig::option_train_set_part_idx;
      const uint32_t part_num = RunConfig::option_train_set_part_num;
      const uint32_t train_set_part_size = (origin_num_train_set + part_num - 1) / part_num;
      begin = train_set_part_size * part_idx;
      end = train_set_part_size * (part_idx + 1);
      if (end > origin_num_train_set) end = origin_num_train_set;
    } else {
      CHECK(false) << "Unknown train set slice mode " << RunConfig::option_train_set_slice_mode;
    }
    meta[Constant::kMetaNumTrainSet] = end - begin;
    _dataset->train_set = Tensor::CopyBlob(
        _dataset->train_set->Data() + begin * GetDataTypeBytes(kI32),
        DataType::kI32, {end - begin}, CPU(CPU_CLIB_MALLOC_DEVICE), ctx_map[Constant::kTrainSetFile], "dataset.train_set");
    std::cout << "reducing trainset from " << origin_num_train_set
              << " to " << meta[Constant::kMetaNumTrainSet]
              << " (" << meta[Constant::kMetaNumTrainSet] * 100.0 / meta[Constant::kMetaNumNode] << ")\n";
  }

  if (RunConfig::sample_type == kWeightedKHop || RunConfig::sample_type == kWeightedKHopHashDedup) {
    _dataset->prob_table = Tensor::FromMmap(
        _dataset_path + Constant::kProbTableFile, DataType::kF32,
        {meta[Constant::kMetaNumEdge]}, ctx_map[Constant::kProbTableFile],
        "dataset.prob_table");

    _dataset->alias_table = Tensor::FromMmap(
        _dataset_path + Constant::kAliasTableFile, DataType::kI32,
        {meta[Constant::kMetaNumEdge]}, ctx_map[Constant::kAliasTableFile],
        "dataset.alias_table");
    _dataset->prob_prefix_table = Tensor::Null();
  } else if (RunConfig::sample_type == kWeightedKHopPrefix){
    _dataset->prob_table = Tensor::Null();
    _dataset->alias_table = Tensor::Null();
    _dataset->prob_prefix_table = Tensor::FromMmap(
        _dataset_path + Constant::kProbPrefixTableFile, DataType::kF32,
        {meta[Constant::kMetaNumEdge]}, ctx_map[Constant::kProbTableFile],
        "dataset.prob_prefix_table");
  } else {
    _dataset->prob_table = Tensor::Null();
    _dataset->alias_table = Tensor::Null();
    _dataset->prob_prefix_table = Tensor::Null();
  }

  if (RunConfig::option_log_node_access) {
    _dataset->in_degrees = Tensor::FromMmap(
        _dataset_path + Constant::kInDegreeFile, DataType::kI32,
        {meta[Constant::kMetaNumNode]}, ctx_map[Constant::kInDegreeFile],
        "dataset.in_degrees");
    _dataset->out_degrees = Tensor::FromMmap(
        _dataset_path + Constant::kOutDegreeFile, DataType::kI32,
        {meta[Constant::kMetaNumNode]}, ctx_map[Constant::kOutDegreeFile],
        "dataset.out_degrees");
  }

  if (RunConfig::UseGPUCache()) {
    switch (RunConfig::cache_policy) {
      case kCacheByDegree:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByDegreeFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByDegreeFile], "dataset.ranking_nodes");
        break;
      case kCacheByHeuristic:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByHeuristicFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByHeuristicFile], "dataset.ranking_nodes");
        break;
      case kCacheByPreSample:
      case kCacheByPreSampleStatic:
        break;
      case kCacheByDegreeHop:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByDegreeHopFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByDegreeHopFile], "dataset.ranking_nodes");
        break;
      case kCacheByFakeOptimal:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByFakeOptimalFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByFakeOptimalFile], "dataset.ranking_nodes");
        break;
      case kCacheByRandom:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByRandomFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByRandomFile], "dataset.ranking_nodes");
        break;
      case kDynamicCache:
        break;
      default:
        CHECK(0);
    }
  }

  double loading_time = t.Passed();
  LOG(INFO) << "SamGraph loaded dataset(" << _dataset_path << ") successfully ("
            << loading_time << " secs)";
  LOG(DEBUG) << "dataset(" << _dataset_path << ") has "
             << _dataset->num_node << " nodes, "
             << _dataset->num_edge << " edges ";
}

bool Engine::IsAllThreadFinish(int total_thread_num) {
  int k = _joined_thread_cnt.fetch_add(0);
  return (k == total_thread_num);
};

void Engine::ForwardBarrier() {
  outer_counter++;
}
void Engine::ForwardInnerBarrier() {
  inner_counter++;
}

}  // namespace common
}  // namespace samgraph
