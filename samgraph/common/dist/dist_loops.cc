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

#include "dist_loops.h"

#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"

#include "dist_engine.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_function.h"
#include "../cuda/dist_graph.h"

namespace samgraph {
namespace common {
namespace dist {

TaskPtr DoShuffle() {
  // auto s = DistEngine::Get()->GetShuffler();
  Shuffler* s;
  StreamHandle sample_stream;
  if (RunConfig::run_arch != RunArch::kArch9) {
    s = DistEngine::Get()->GetShuffler();
    sample_stream = DistEngine::Get()->GetSampleStream();
  } else {
    auto tid = std::this_thread::get_id();
    s = DistEngine::Get()->GetUMSamplerByTid(tid)->GetShuffler();
    sample_stream = DistEngine::Get()->GetUMSamplerByTid(tid)->SampleStream();
  }
  auto batch = s->GetBatch(sample_stream);

  if (batch) {
    auto task = std::make_shared<Task>();
    // global key
    task->key = DistEngine::Get()->GetBatchKey(s->Epoch(), s->Step());
    task->output_nodes = batch;
    LOG(DEBUG) << "DoShuffle: process task with key " << task->key;
    return task;
  } else {
    return nullptr;
  }
}

void DoGPUSample(TaskPtr task) {
  auto fanouts = DistEngine::Get()->GetFanout();
  auto num_layers = fanouts.size();
  auto last_layer_idx = num_layers - 1;

  auto dataset = DistEngine::Get()->GetGraphDataset();


  Context sampler_ctx;
  Device* sampler_device;
  StreamHandle sample_stream;

  cuda::GPURandomStates* random_states = nullptr;
  cuda::FrequencyHashmap* frequency_hashmap = nullptr;
  cuda::OrderedHashTable *hash_table = nullptr;

  if (RunConfig::run_arch != RunArch::kArch9) {
    sampler_ctx = DistEngine::Get()->GetSamplerCtx();
    sampler_device = Device::Get(sampler_ctx);
    sample_stream = DistEngine::Get()->GetSampleStream();
    random_states = DistEngine::Get()->GetRandomStates();
    frequency_hashmap = DistEngine::Get()->GetFrequencyHashmap();
    hash_table = DistEngine::Get()->GetHashtable();
  } else {
    DistUMSampler* sampler;
    if (!DistEngine::Get()->IsInitialized()) {
      sampler = DistEngine::Get()->GetUMSamplers()[0];
    } else {
      auto tid= std::this_thread::get_id();
      sampler = DistEngine::Get()->GetUMSamplerByTid(tid);
    }
    sampler_ctx = sampler->Ctx();
    sampler_device = Device::Get(sampler_ctx);
    sample_stream = sampler->SampleStream();
    random_states = sampler->GetGPURandomStates();
    frequency_hashmap = sampler->GetFrequencyHashmap();
    hash_table = sampler->GetHashTable();
  }
  hash_table->Reset(sample_stream);

  Timer t;
  auto output_nodes = task->output_nodes;
  size_t num_train_node = output_nodes->Shape()[0];
  hash_table->FillWithDupRevised(output_nodes->CPtr<IdType>(), num_train_node,
      sample_stream);
  task->graphs.resize(num_layers);
  double fill_unique_time = t.Passed();

  const IdType *indptr = dataset->indptr->CPtr<IdType>();
  const IdType *indices = dataset->indices->CPtr<IdType>();
  const IdType num_node = dataset->num_node;
  const float *prob_table = dataset->prob_table->CPtr<float>();
  const IdType *alias_table = dataset->alias_table->CPtr<IdType>();
  const float *prob_prefix_table = dataset->prob_prefix_table->CPtr<float>();

  auto cur_input = task->output_nodes;
  size_t total_num_samples = 0;

  for (int i = last_layer_idx; i >= 0; i--) {
    Timer tlayer;
    Timer t0;
    const size_t fanout = fanouts[i];
    const IdType *input = cur_input->CPtr<IdType>();
    const size_t num_input = cur_input->Shape()[0];
    LOG(DEBUG) << "DoGPUSample: begin sample layer=" << i
               << " ctx=" << sampler_ctx
               << " num_input=" << num_input;

    IdType *out_src = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    IdType *out_dst = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    IdType *out_data = nullptr;
    if (RunConfig::sample_type == kRandomWalk) {
      out_data = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    }
    size_t *num_out = sampler_device->AllocArray<size_t>(sampler_ctx, 1);
    size_t num_samples;

    LOG(DEBUG) << "DoGPUSample: size of out_src " << num_input * fanout;
    LOG(TRACE) << "DoGPUSample: cuda out_src malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(TRACE) << "DoGPUSample: cuda out_dst malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(TRACE) << "DoGPUSample: cuda num_out malloc "
               << ToReadableSize(sizeof(size_t));

    // Sample a compact coo graph
    switch (RunConfig::sample_type) {
      case kKHop0:
        {
          if (RunConfig::use_dist_graph) {
            cuda::GPUSampleKHop0<cuda::DeviceDistGraph>(
              cuda::DistGraph::Get()->DeviceGraphHandle(),
              input, num_input, fanout, out_src,
              out_dst, num_out, sampler_ctx, sample_stream,
              random_states, task->key);
          } else {
            cuda::GPUSampleKHop0<cuda::DeviceNormalGraph>(
              cuda::DeviceNormalGraph(indptr, indices, num_node),
              input, num_input, fanout, out_src,
              out_dst, num_out, sampler_ctx, sample_stream,
              random_states, task->key);
          }
        }
        break;
      case kKHop1:
        {
          CHECK(RunConfig::use_dist_graph == false)
            << "this algorithm not support DistGraph engine";
          cuda::GPUSampleKHop1(indptr, indices, input, num_input, fanout, out_src,
                         out_dst, num_out, sampler_ctx, sample_stream,
                         random_states, task->key);
        }
        break;
      case kWeightedKHop:
        {
          CHECK(RunConfig::use_dist_graph == false)
            << "this algorithm not support DistGraph engine";
          cuda::GPUSampleWeightedKHop(indptr, indices, prob_table, alias_table, input,
                                num_input, fanout, out_src, out_dst, num_out,
                                sampler_ctx, sample_stream, random_states,
                                task->key);
        }
        break;
      case kRandomWalk:
        {
          CHECK_EQ(fanout, RunConfig::num_neighbor);
          if (RunConfig::use_dist_graph) {
            cuda::GPUSampleRandomWalk<cuda::DeviceDistGraph>(
              cuda::DistGraph::Get()->DeviceGraphHandle(),
              input, num_input, RunConfig::random_walk_length,
              RunConfig::random_walk_restart_prob, RunConfig::num_random_walk,
              RunConfig::num_neighbor, out_src, out_dst, out_data, num_out,
              frequency_hashmap, sampler_ctx, sample_stream, random_states,
              task->key);
          } else {
            cuda::GPUSampleRandomWalk<cuda::DeviceNormalGraph>(
              cuda::DeviceNormalGraph(indptr, indices, num_node),
              input, num_input, RunConfig::random_walk_length,
              RunConfig::random_walk_restart_prob, RunConfig::num_random_walk,
              RunConfig::num_neighbor, out_src, out_dst, out_data, num_out,
              frequency_hashmap, sampler_ctx, sample_stream, random_states,
              task->key);
          }
        }
        break;
      case kWeightedKHopPrefix:
        {
          CHECK(RunConfig::use_dist_graph == false)
            << "this algorithm not support DistGraph engine";
          cuda::GPUSampleWeightedKHopPrefix(indptr, indices, prob_prefix_table, input,
                                num_input, fanout, out_src, out_dst, num_out,
                                sampler_ctx, sample_stream, random_states,
                                task->key);
        }
        break;
      case kKHop2:
        {
          CHECK(RunConfig::use_dist_graph == false)
            << "this algorithm not support DistGraph engine";
          cuda::GPUSampleKHop2(indptr, const_cast<IdType*>(indices), input, num_input, fanout, out_src,
                         out_dst, num_out, sampler_ctx, sample_stream,
                         random_states, task->key);
        }
        break;
      case kWeightedKHopHashDedup:
        {
          CHECK(RunConfig::use_dist_graph == false)
            << "this algorithm not support DistGraph engine";
          cuda::GPUSampleWeightedKHopHashDedup(indptr, const_cast<IdType*>(indices), const_cast<float*>(prob_table), alias_table, input,
              num_input, fanout, out_src, out_dst, num_out, sampler_ctx, sample_stream, random_states, task->key);
        }
        break;
      case kKHop3:
        {
          if (RunConfig::use_dist_graph) {
            LOG(DEBUG) << "use DeviceDistGraph to sample";
            cuda::GPUSampleKHop3<cuda::DeviceDistGraph>(
                cuda::DistGraph::Get()->DeviceGraphHandle(),
                input, num_input, fanout, out_src,
                out_dst, num_out, sampler_ctx, sample_stream,
                random_states, task->key);
          } else {
            LOG(DEBUG) << "use DeviceNormalGraph to sample";
            cuda::GPUSampleKHop3<cuda::DeviceNormalGraph>(
                cuda::DeviceNormalGraph(indptr, indices, num_node),
                input, num_input, fanout, out_src,
                out_dst, num_out, sampler_ctx, sample_stream,
                random_states, task->key);
          }
        }
        break;
      default:
        CHECK(0);
    }

    // Get nnz
    sampler_device->CopyDataFromTo(num_out, 0, &num_samples, 0, sizeof(size_t),
                                   sampler_ctx, CPU(), sample_stream);
    sampler_device->StreamSync(sampler_ctx, sample_stream);

    LOG(DEBUG) << "DoGPUSample: "
               << "layer " << i << " number of samples " << num_samples;

    double core_sample_time = t0.Passed();

    Timer t1;
    Timer t2;

    // Populate the hash table with newly sampled nodes
    IdType *unique = sampler_device->AllocArray<IdType>(
        sampler_ctx, num_samples + hash_table->NumItems());
    IdType num_unique;

    LOG(TRACE) << "GPUSample: cuda unique malloc "
               << ToReadableSize((num_samples + +hash_table->NumItems()) *
                                 sizeof(IdType));

    hash_table->FillWithDuplicates(out_dst, num_samples, unique, &num_unique,
                                   sample_stream);

    double populate_time = t2.Passed();

    Timer t3;

    // Mapping edges
    IdType *new_src = sampler_device->AllocArray<IdType>(sampler_ctx, num_samples);
    IdType *new_dst = sampler_device->AllocArray<IdType>(sampler_ctx, num_samples);

    LOG(DEBUG) << "GPUSample: size of new_src " << num_samples;
    LOG(TRACE) << "GPUSample: cuda new_src malloc "
               << ToReadableSize(num_samples * sizeof(IdType));
    LOG(TRACE) << "GPUSample: cuda new_dst malloc "
               << ToReadableSize(num_samples * sizeof(IdType));

    cuda::GPUMapEdges(out_src, new_src, out_dst, new_dst, num_samples,
                hash_table->DeviceHandle(), sampler_ctx, sample_stream);

    double map_edges_time = t3.Passed();
    double remap_time = t1.Passed();
    double layer_time = tlayer.Passed();

    auto train_graph = std::make_shared<TrainGraph>();
    train_graph->num_src = num_unique;
    train_graph->num_dst = num_input;
    train_graph->num_edge = num_samples;
    train_graph->col = Tensor::FromBlob(
        new_src, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.row_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    train_graph->row = Tensor::FromBlob(
        new_dst, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.dst_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    if (out_data) {
      train_graph->data = Tensor::FromBlob(
          out_data, DataType::kI32, {num_samples}, sampler_ctx,
          "train_graph.dst_cuda_sample_" + std::to_string(task->key) + "_" +
              std::to_string(i));
    }

    task->graphs[i] = train_graph;

    total_num_samples += num_samples;

    // Do some clean jobs
    sampler_device->FreeWorkspace(sampler_ctx, out_src);
    sampler_device->FreeWorkspace(sampler_ctx, out_dst);
    sampler_device->FreeWorkspace(sampler_ctx, num_out);
    if (i == (int)last_layer_idx) {
        Profiler::Get().LogStep(task->key, kLogL2LastLayerTime,
                                   layer_time);
        Profiler::Get().LogStep(task->key, kLogL2LastLayerSize,
                                   num_unique);
    }
    LOG(DEBUG) << "_debug task_key=" << task->key << " layer=" << i << " ctx=" << sampler_ctx
               << " num_unique=" << num_unique << " num_sample=" << num_samples
               << " remap_time ( " << remap_time << " )"
               << " = populate_time ( " << populate_time << " )"
               << " + edge_map_time ( " << map_edges_time << " )";
    Profiler::Get().LogStepAdd(task->key, kLogL2CoreSampleTime,
                               core_sample_time);
    Profiler::Get().LogStepAdd(task->key, kLogL2IdRemapTime, remap_time);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapPopulateTime,
                               populate_time);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapMapNodeTime, 0);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapMapEdgeTime,
                               map_edges_time);

    cur_input = Tensor::FromBlob(
        (void *)unique, DataType::kI32, {num_unique}, sampler_ctx,
        "cur_input_unique_cuda_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    LOG(DEBUG) << "GPUSample: finish layer " << i;
  }

  task->input_nodes = cur_input;

  Profiler::Get().LogStep(task->key, kLogL1NumNode,
                          static_cast<double>(task->input_nodes->Shape()[0]));
  Profiler::Get().LogStep(task->key, kLogL1NumSample, total_num_samples);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochNumSample, total_num_samples);
  Profiler::Get().LogStepAdd(task->key, kLogL3RemapFillUniqueTime,
                             fill_unique_time);

  LOG(DEBUG) << " sampler(" << sampler_ctx << ") "
             << "SampleLoop: process task with key " << task->key;
}

void DoGetCacheMissIndex(TaskPtr task) {
  // Get index of miss data and cache data
  // Timer t4;
  Context sampler_ctx;
  Device* sampler_device;
  StreamHandle sample_stream;
  if (RunConfig::run_arch != RunArch::kArch9) {
    sampler_ctx = DistEngine::Get()->GetSamplerCtx();
    sampler_device = Device::Get(sampler_ctx);
    sample_stream = DistEngine::Get()->GetSampleStream();
  } else {
    auto tid = std::this_thread::get_id();
    auto sampler = DistEngine::Get()->GetUMSamplerByTid(tid);
    sampler_ctx = sampler->Ctx();
    sampler_device = sampler->GetDevice();
    sample_stream = sampler->SampleStream();
  }

  if (RunConfig::UseGPUCache() && DistEngine::Get()->IsInitialized()) {
    auto input_nodes = task->input_nodes->CPtr<IdType>();
    const size_t num_input = task->input_nodes->Shape()[0];

    IdType *sampler_output_miss_src_index =
        sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
    IdType *sampler_output_miss_dst_index =
        sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
    IdType *sampler_output_cache_src_index =
        sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
    IdType *sampler_output_cache_dst_index =
        sampler_device->AllocArray<IdType>(sampler_ctx, num_input);

    size_t num_output_miss;
    size_t num_output_cache;

    IdType* cache_hash_tb;
    if (RunConfig::run_arch != RunArch::kArch9) {
      cache_hash_tb = DistEngine::Get()->GetCacheHashtable();
    } else {
      cache_hash_tb = DistEngine::Get()
        ->GetUMSamplerByTid(std::this_thread::get_id())
        ->GetCacheHashTable();
    }
    cuda::GetMissCacheIndex(
        cache_hash_tb, sampler_ctx,
        sampler_output_miss_src_index, sampler_output_miss_dst_index,
        &num_output_miss, sampler_output_cache_src_index,
        sampler_output_cache_dst_index, &num_output_cache, input_nodes, num_input,
        sample_stream);

    CHECK_EQ(num_output_miss + num_output_cache, num_input);

    auto dtype = task->input_nodes->Type();
    // To be freed in task queue after serialization
    task->miss_cache_index.miss_src_index =
        Tensor::FromBlob(sampler_output_miss_src_index, dtype,
                         {num_output_miss}, sampler_ctx, "miss_src_index");
    task->miss_cache_index.miss_dst_index =
        Tensor::FromBlob(sampler_output_miss_dst_index, dtype,
                         {num_output_miss}, sampler_ctx, "miss_dst_index");
    task->miss_cache_index.cache_src_index =
        Tensor::FromBlob(sampler_output_cache_src_index, dtype,
                         {num_output_cache}, sampler_ctx, "cache_src_index");
    task->miss_cache_index.cache_dst_index =
        Tensor::FromBlob(sampler_output_cache_dst_index, dtype,
                         {num_output_cache}, sampler_ctx, "cache_dst_index");

    sampler_device->StreamSync(sampler_ctx, sample_stream);

    task->miss_cache_index.num_miss = num_output_miss;
    task->miss_cache_index.num_cache = num_output_cache;
  }
}

void DoGraphCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto copy_ctx = trainer_ctx;
  auto copy_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  for (size_t i = 0; i < task->graphs.size(); i++) {
    auto graph = task->graphs[i];
    Profiler::Get().LogStepAdd(task->key, kLogL1GraphBytes,
                               graph->row->NumBytes() + graph->col->NumBytes());
    if (graph->row->Ctx() == trainer_ctx) {
      CHECK(graph->col->Ctx() == trainer_ctx) << "col ctx needs equal row in graph";
      continue;
    }
    graph->row = Tensor::CopyTo(graph->row, trainer_ctx, copy_stream,
        "train_graph.row_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
    graph->col = Tensor::CopyTo(graph->col, trainer_ctx, copy_stream,
        "train_graph.col_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));

    LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_row malloc "
               << ToReadableSize(graph->row->NumBytes());
    LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_col malloc "
               << ToReadableSize(graph->col->NumBytes());

    if (RunConfig::sample_type == kRandomWalk) {
      CHECK(graph->data != nullptr);
      graph->data = Tensor::CopyTo(graph->data, trainer_ctx, copy_stream,
        "train_graph.data_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
      LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda graph data malloc "
                 << ToReadableSize(graph->data->NumBytes());
    }

    copy_device->StreamSync(copy_ctx, copy_stream);
  }

  LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
}

void DoIdCopy(TaskPtr task) {
  auto copy_ctx = task->input_nodes->Ctx();
  auto copy_device = Device::Get(copy_ctx);
  auto copy_stream = DistEngine::Get()->GetSamplerCopyStream();

  task->input_nodes = Tensor::CopyTo(task->input_nodes, CPU(), copy_stream,
      "task.input_nodes_cpu_" + std::to_string(task->key));
  task->output_nodes = Tensor::CopyTo(task->output_nodes, CPU(), copy_stream,
      "task.output_nodes_cpu_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyDevice2Host input_nodes cpu malloc "
             << ToReadableSize(task->input_nodes->NumBytes());
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cpu malloc "
             << ToReadableSize(task->output_nodes->NumBytes());

  copy_device->StreamSync(copy_ctx, copy_stream);

  Profiler::Get().LogStepAdd(
      task->key, kLogL1IdBytes,
      task->input_nodes->NumBytes() + task->output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoCPUFeatureExtract(TaskPtr task) {
  auto dataset = DistEngine::Get()->GetGraphDataset();

  auto input_nodes = task->input_nodes;
  auto output_nodes = task->output_nodes;

  auto feat = dataset->feat;
  auto label = dataset->label;

  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();
  auto label_type = dataset->label->Type();

  auto input_data = input_nodes->CPtr<IdType>();
  auto output_data = output_nodes->CPtr<IdType>();
  auto num_input = input_nodes->Shape()[0];
  auto num_ouput = output_nodes->Shape()[0];

  task->input_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, CPU(),
                    "task.input_feat_cpu_" + std::to_string(task->key));
  task->output_label =
      Tensor::Empty(label_type, {num_ouput}, CPU(),
                    "task.output_label_cpu" + std::to_string(task->key));

  LOG(DEBUG) << "DoCPUFeatureExtract input_feat cpu malloc "
             << ToReadableSize(task->input_feat->NumBytes());
  LOG(DEBUG) << "DoCPUFeatureExtract output_label cpu malloc "
             << ToReadableSize(task->output_label->NumBytes());

  auto feat_dst = task->input_feat->MutableData();
  auto feat_src = dataset->feat->Data();

  if (RunConfig::option_empty_feat != 0) {
    cpu::CPUMockExtract(feat_dst, feat_src, input_data, num_input, feat_dim,
                    feat_type);
  } else {
    cpu::CPUExtract(feat_dst, feat_src, input_data, num_input, feat_dim,
                    feat_type);
  }

  auto label_dst = task->output_label->MutableData();
  auto label_src = dataset->label->Data();

  cpu::CPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type);

  if (RunConfig::option_log_node_access) {
    Profiler::Get().LogNodeAccess(task->key, input_data, num_input);
  }

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
}

void DoFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  auto cpu_feat = task->input_feat;
  auto cpu_label = task->output_label;

  CHECK_EQ(cpu_feat->Ctx().device_type, CPU().device_type);
  CHECK_EQ(cpu_label->Ctx().device_type, CPU().device_type);

  task->input_feat = Tensor::CopyTo(cpu_feat, trainer_ctx, copy_stream,
      "task.train_feat_cuda_" + std::to_string(task->key));
  task->output_label = Tensor::CopyTo(cpu_label, trainer_ctx, copy_stream,
      "task.train_label_cuda" + std::to_string(task->key));

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          task->input_feat->NumBytes());
  Profiler::Get().LogStep(task->key, kLogL1LabelBytes,
                          task->output_label->NumBytes());
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              task->input_feat->NumBytes());
  Profiler::Get().LogEpochAdd(task->key, kLogEpochMissBytes,
                              task->input_feat->NumBytes());

  LOG(DEBUG) << "FeatureCopyHost2Device: process task with key " << task->key;
}

void DoGPUFeatureExtract(TaskPtr task) {
  CHECK_EQ(RunConfig::run_arch, kArch6);
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto stream = DistEngine::Get()->GetTrainerCopyStream();

  auto dataset = DistEngine::Get()->GetGraphDataset();

  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();
  auto label_type = dataset->label->Type();

  auto input_data = task->input_nodes->CPtr<IdType>();
  auto output_data = task->output_nodes->CPtr<IdType>();
  auto num_input = task->input_nodes->Shape()[0];
  auto num_output = task->output_nodes->Shape()[0];

  task->input_feat = Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                                   "task.input_feat_cuda_" + std::to_string(task->key));
  task->output_label = Tensor::Empty(label_type, {num_output}, trainer_ctx,
                                     "task.output_label_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "DoGPUFeatureExtract input_feat gpu malloc " << ToReadableSize(task->input_feat->NumBytes())
             << " output_label gpu malloc " << ToReadableSize(task->output_label->NumBytes());

  if (RunConfig::option_empty_feat) {
    cuda::GPUMockExtract(
      task->input_feat->MutableData(), dataset->feat->Data(),
      input_data, num_input, feat_dim, feat_type,
      trainer_ctx, stream, task->key
    );
  } else {
    cuda::GPUExtract(
      task->input_feat->MutableData(), dataset->feat->Data(),
      input_data, num_input, feat_dim, feat_type,
      trainer_ctx, stream, task->key
    );
  }
  cuda::GPUExtract(
    task->output_label->MutableData(), dataset->label->Data(),
    output_data, num_output, 1, label_type,
    trainer_ctx, stream, task->key
  );
  LOG(DEBUG) << "GPUFeatureExtract: process task with key " << task->key;

  auto feat_nbytes = task->input_feat->NumBytes();
  auto label_nbytes = task->output_label->NumBytes();
  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes, feat_nbytes);
  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, label_nbytes);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes, feat_nbytes);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochMissBytes, feat_nbytes);
}

void DoCacheIdCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto copy_ctx = trainer_ctx;
  auto copy_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  task->output_nodes = Tensor::CopyTo(task->output_nodes, trainer_ctx, copy_stream,
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyHost2Device output_nodes cuda malloc "
             << ToReadableSize(task->output_nodes->NumBytes());

  copy_device->StreamSync(copy_ctx, copy_stream);

  Profiler::Get().LogStepAdd(task->key, kLogL1IdBytes,
                             task->output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoCacheIdCopyToCPU(TaskPtr task) {
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = DistEngine::Get()->GetSamplerCopyStream();

  task->output_nodes = Tensor::CopyTo(task->output_nodes, CPU(), copy_stream,
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cuda malloc "
             << ToReadableSize(task->output_nodes->NumBytes());

  sampler_device->StreamSync(sampler_ctx, copy_stream);

  Profiler::Get().LogStepAdd(task->key, kLogL1IdBytes,
                             task->output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

// for switcher cache
void DoSwitchCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto trainer_copy_stream = DistEngine::Get()->GetTrainerCopyStream();
  auto cpu_ctx = CPU();
  auto cpu_device = Device::Get(cpu_ctx);

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetGPUCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto input_data = input_nodes->CPtr<IdType>();
  auto num_input = input_nodes->Shape()[0];

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 1. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  // do get_miss_cache_index in trainer GPU
  Timer t0;

  IdType *trainer_output_miss_src_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);
  IdType *trainer_output_miss_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);
  IdType *trainer_output_cache_src_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);
  IdType *trainer_output_cache_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);

  size_t num_output_miss;
  size_t num_output_cache;

  cache_manager->GetMissCacheIndex(
      trainer_output_miss_src_index, trainer_output_miss_dst_index,
      &num_output_miss, trainer_output_cache_src_index,
      trainer_output_cache_dst_index, &num_output_cache, input_data, num_input,
      trainer_copy_stream);

  CHECK_EQ(num_output_miss + num_output_cache, num_input);

  double get_index_time = t0.Passed();

  // 2. Move the miss index

  Timer t1;

  IdType *cpu_output_miss_src_index =
      cpu_device->AllocArray<IdType>(CPU(), num_output_miss);
  trainer_device->CopyDataFromTo(trainer_output_miss_src_index, 0,
                                 cpu_output_miss_src_index, 0,
                                 num_output_miss * sizeof(IdType), trainer_ctx,
                                 CPU(), trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);
  // free the miss src indices in GPU
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss_src_index);

  double copy_idx_time = t1.Passed();

  // 3. Extract the miss data
  Timer t2;

  auto cpu_output_miss = Tensor::Empty(
      feat_type, {num_output_miss, feat_dim}, CPU(), "");

  cache_manager->ExtractMissData(cpu_output_miss->MutableData(),
                                 cpu_output_miss_src_index, num_output_miss);

  double extract_miss_time = t2.Passed();

  // 4. Copy the miss data
  Timer t3;

  auto trainer_output_miss = Tensor::CopyTo(
      cpu_output_miss, trainer_ctx, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  cpu_output_miss = nullptr;

  double copy_miss_time = t3.Passed();

  // 5. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss->Data(),
                                 trainer_output_miss_dst_index, num_output_miss,
                                 trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_miss_time = t4.Passed();

  // 6. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_cache_time = t5.Passed();

  task->input_feat = train_feat;

  // 7. Free space
  cpu_device->FreeWorkspace(CPU(), cpu_output_miss_src_index);
  trainer_output_miss = nullptr;
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss_dst_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_src_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_dst_index);

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(
      task->key, kLogL1MissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime, get_index_time);
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime, copy_idx_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime, copy_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochMissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  LOG(DEBUG) << "DoSwitchCacheFeatureCopy: process task with key " << task->key;
}

void DoCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto trainer_copy_stream = DistEngine::Get()->GetTrainerCopyStream();
  auto cpu_ctx = CPU();
  auto cpu_device = Device::Get(cpu_ctx);

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto num_input =
      task->miss_cache_index.num_cache + task->miss_cache_index.num_miss;

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 1. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;

  size_t num_output_miss = task->miss_cache_index.num_miss;
  size_t num_output_cache = task->miss_cache_index.num_cache;

  const IdType *output_miss_src_index = nullptr;
  const IdType *trainer_output_miss_dst_index = nullptr;
  const IdType *trainer_output_cache_src_index = nullptr;
  const IdType *trainer_output_cache_dst_index = nullptr;
  if (num_output_miss > 0) {
    output_miss_src_index = task->miss_cache_index.miss_src_index->CPtr<IdType>();
    CHECK_EQ(task->miss_cache_index.miss_dst_index->Ctx(), trainer_ctx)
      << "output_miss_dst_index should be in trainer GPU";
    trainer_output_miss_dst_index = task->miss_cache_index.miss_dst_index->CPtr<IdType>();
  }

  if (num_output_cache > 0) {
    CHECK_EQ(task->miss_cache_index.cache_src_index->Ctx(), trainer_ctx)
      << "output_cache_src_index should be in trainer GPU";
    trainer_output_cache_src_index = task->miss_cache_index.cache_src_index->CPtr<IdType>();
    CHECK_EQ(task->miss_cache_index.cache_dst_index->Ctx(), trainer_ctx)
      << "output_cache_dst_index should be in trainer GPU";
    trainer_output_cache_dst_index = task->miss_cache_index.cache_dst_index->CPtr<IdType>();
  }

  double get_index_time = t0.Passed();

  // 2. Move the miss index

  Timer t1;

  const IdType *cpu_output_miss_src_index = output_miss_src_index;

  double copy_idx_time = t1.Passed();

  // 3. Extract and copy the miss data
  Timer t2;

  auto cpu_output_miss = Tensor::Empty(
      feat_type, {num_output_miss, feat_dim}, CPU(), "");

  cache_manager->ExtractMissData(cpu_output_miss->MutableData(), cpu_output_miss_src_index,
                                 num_output_miss);

  double extract_miss_time = t2.Passed();

  // 4. Copy the miss data
  Timer t3;

  auto trainer_output_miss = Tensor::CopyTo(
      cpu_output_miss, trainer_ctx, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  cpu_output_miss = nullptr;

  double copy_miss_time = t3.Passed();

  // 5. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss->Data(),
                                 trainer_output_miss_dst_index, num_output_miss,
                                 trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_miss_time = t4.Passed();

  // 6. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_cache_time = t5.Passed();

  task->input_feat = train_feat;

  // 7. Free space
  trainer_output_miss = nullptr;

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(
      task->key, kLogL1MissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime, get_index_time);
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime, copy_idx_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime, copy_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochMissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  LOG(DEBUG) << "DoCacheFeatureCopy: process task with key " << task->key;
}

void DoGPULabelExtract(TaskPtr task) {
  auto dataset = DistEngine::Get()->GetGraphDataset();

  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto trainer_copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  auto output_nodes = task->output_nodes;
  auto label = dataset->label;
  auto label_type = dataset->label->Type();

  auto output_data = output_nodes->CPtr<IdType>();
  auto num_ouput = output_nodes->Shape()[0];

  auto train_label =
      Tensor::Empty(label_type, {num_ouput}, trainer_ctx,
                    "task.train_label_cuda" + std::to_string(task->key));

  void *label_dst = train_label->MutableData();
  const void *label_src = dataset->label->Data();

  CHECK_EQ(output_nodes->Ctx().device_type, trainer_ctx.device_type);
  CHECK_EQ(output_nodes->Ctx().device_id, trainer_ctx.device_id);
  // CHECK_EQ(dataset->label->Ctx().device_type, trainer_ctx.device_type);
  // CHECK_EQ(dataset->label->Ctx().device_id, trainer_ctx.device_id);

  cuda::GPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type,
             trainer_ctx, trainer_copy_stream, task->key);

  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;

  task->output_label = train_label;

  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());
}

void DoCPULabelExtractAndCopy(TaskPtr task) {
  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  // 1. Extract
  auto output_nodes = task->output_nodes;

  auto label = dataset->label;
  auto label_type = dataset->label->Type();

  auto output_data = output_nodes->CPtr<IdType>();
  auto num_ouput = output_nodes->Shape()[0];

  task->output_label =
      Tensor::Empty(label_type, {num_ouput}, CPU(),
                    "task.output_label_cpu" + std::to_string(task->key));

  LOG(DEBUG) << "DoCPUFeatureExtract output_label cpu malloc "
             << ToReadableSize(task->output_label->NumBytes());

  auto label_dst = task->output_label->MutableData();
  auto label_src = dataset->label->Data();

  cpu::CPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type);

  // 2. Copy
  auto cpu_label = task->output_label;
  CHECK_EQ(cpu_label->Ctx().device_type, CPU().device_type);
  task->output_label = Tensor::CopyTo(cpu_label, trainer_ctx, copy_stream,
      "task.train_label_cuda" + std::to_string(task->key));
  trainer_device->StreamSync(trainer_ctx, copy_stream);

  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, task->output_label->NumBytes());

  LOG(DEBUG) << "DoCPULabelExtractAndCopy: process task with key " << task->key;
}

void DoArch6GetCacheMissIndex(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto stream = DistEngine::Get()->GetTrainerCopyStream();

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetGPUCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;

  auto input_data = input_nodes->CPtr<IdType>();
  auto num_input = input_nodes->Shape()[0];

  IdType *trainer_output_miss_src_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);
  IdType *trainer_output_miss_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);
  IdType *trainer_output_cache_src_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);
  IdType *trainer_output_cache_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_input);

  size_t num_output_miss;
  size_t num_output_cache;

  cache_manager->GetMissCacheIndex(
      trainer_output_miss_src_index, trainer_output_miss_dst_index,
      &num_output_miss, trainer_output_cache_src_index,
      trainer_output_cache_dst_index, &num_output_cache, input_data, num_input,
      stream);

  CHECK_EQ(num_output_miss + num_output_cache, num_input);

  if (RunConfig::part_cache
      || (RunConfig::use_ics22_song_solver && RunConfig::ics22_compact_mode)) {
    cache_manager->CountLocalCache(task->key, trainer_output_cache_src_index,
        num_output_cache, num_input, stream);
  }

  auto dtype = task->input_nodes->Type();
  // To be freed in task queue after serialization
  task->miss_cache_index.miss_src_index =
      Tensor::FromBlob(trainer_output_miss_src_index, dtype, {num_output_miss},
                       trainer_ctx, "miss_src_index");
  task->miss_cache_index.miss_dst_index =
      Tensor::FromBlob(trainer_output_miss_dst_index, dtype, {num_output_miss},
                       trainer_ctx, "miss_dst_index");
  task->miss_cache_index.cache_src_index =
      Tensor::FromBlob(trainer_output_cache_src_index, dtype,
                       {num_output_cache}, trainer_ctx, "cache_src_index");
  task->miss_cache_index.cache_dst_index =
      Tensor::FromBlob(trainer_output_cache_dst_index, dtype,
                       {num_output_cache}, trainer_ctx, "cache_dst_index");

  trainer_device->StreamSync(trainer_ctx, stream);

  task->miss_cache_index.num_miss = num_output_miss;
  task->miss_cache_index.num_cache = num_output_cache;
}

void DoArch6CacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto cpu_device = Device::Get(CPU());
  auto stream = DistEngine::Get()->GetTrainerCopyStream();

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetGPUCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto num_input = input_nodes->Shape()[0];

  CHECK_EQ(input_nodes->Ctx().device_type, trainer_ctx.device_type);
  CHECK_EQ(input_nodes->Ctx().device_id, trainer_ctx.device_id);

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 0. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;

  size_t num_output_miss = task->miss_cache_index.num_miss;
  size_t num_output_cache = task->miss_cache_index.num_cache;

  // LOG(INFO) << "Miss, Hit: " << num_input << " " << num_output_miss << " " << num_output_cache;

  IdType *trainer_output_miss_src_index =
      task->miss_cache_index.miss_src_index->Ptr<IdType>();
  IdType *trainer_output_miss_dst_index =
      task->miss_cache_index.miss_dst_index->Ptr<IdType>();
  IdType *trainer_output_cache_src_index =
      task->miss_cache_index.cache_src_index->Ptr<IdType>();
  IdType *trainer_output_cache_dst_index =
      task->miss_cache_index.cache_dst_index->Ptr<IdType>();

  CHECK_EQ(num_output_miss + num_output_cache, num_input);

  double get_index_time = t0.Passed();

  // 1. Move the miss index

  Timer t1;

  IdType *cpu_output_miss_src_index =
      cpu_device->AllocArray<IdType>(CPU(), num_output_miss);

  trainer_device->CopyDataFromTo(
      trainer_output_miss_src_index, 0, cpu_output_miss_src_index, 0,
      num_output_miss * sizeof(IdType), trainer_ctx, CPU(), stream);

  trainer_device->StreamSync(trainer_ctx, stream);

  double copy_idx_time = t1.Passed();

  // 2. Extract the miss data
  Timer t2;

  auto cpu_output_miss = Tensor::Empty(feat_type, {num_output_miss, feat_dim}, CPU(), "");
  cache_manager->ExtractMissData(cpu_output_miss->MutableData(), cpu_output_miss_src_index,
                                 num_output_miss);

  double extract_miss_time = t2.Passed();

  // 3. Copy the miss data
  Timer t3;
  auto trainer_output_miss = Tensor::CopyTo(
      cpu_output_miss, trainer_ctx, stream);
  trainer_device->StreamSync(trainer_ctx, stream);

  cpu_output_miss = nullptr;

  double copy_miss_time = t3.Passed();

  // 4. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss->Data(),
                                 trainer_output_miss_dst_index, num_output_miss,
                                 stream);
  trainer_device->StreamSync(trainer_ctx, stream);

  double combine_miss_time = t4.Passed();

  // 5. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, stream);
  trainer_device->StreamSync(trainer_ctx, stream);

  double combine_cache_time = t5.Passed();

  task->input_feat = train_feat;

  // 5. Free space
  cpu_device->FreeWorkspace(CPU(), cpu_output_miss_src_index);
  trainer_output_miss = nullptr;
  task->miss_cache_index.miss_src_index = nullptr;
  task->miss_cache_index.miss_dst_index = nullptr;
  task->miss_cache_index.cache_src_index = nullptr;
  task->miss_cache_index.cache_dst_index = nullptr;

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(
      task->key, kLogL1MissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime,
                          get_index_time);  // t0, step 0
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime,
                          copy_idx_time);  // t1, step 1
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time);  // t2, step2
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime,
                          copy_miss_time);  // t3, step 3
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);  // t4, step 4
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time);  // t5, step 5
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochMissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  LOG(DEBUG) << "DoCacheFeatureCopy: process task with key " << task->key;
}

void DoArch6GPUCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto stream = DistEngine::Get()->GetTrainerCopyStream();

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetGPUCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = feat->Shape()[1];
  auto feat_type = feat->Type();

  auto num_input = input_nodes->Shape()[0];

  CHECK(input_nodes->Ctx() == trainer_ctx);

  auto train_feat = Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                                  "task.train_feat_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "step " << DistEngine::Get()->GetShuffler()->Step()
    << " extracting train feat size: "
    << ToReadableSize(train_feat->NumBytes());
  // LOG(INFO) << "free workspace size is: "
  //   << ToReadableSize(trainer_device->FreeWorkspaceSize(trainer_ctx));
  // 0. Get index of miss data and cache data
  Timer t0;

  size_t num_output_miss = task->miss_cache_index.num_miss;
  size_t num_output_cache = task->miss_cache_index.num_cache;

  auto trainer_output_miss_src_index =
      task->miss_cache_index.miss_src_index->Ptr<IdType>();
  auto trainer_output_miss_dst_index =
      task->miss_cache_index.miss_dst_index->Ptr<IdType>();
  auto trainer_output_cache_src_index =
      task->miss_cache_index.cache_src_index->Ptr<IdType>();
  auto trainer_output_cache_dst_index =
      task->miss_cache_index.cache_dst_index->Ptr<IdType>();

  CHECK_EQ(num_output_miss + num_output_cache, num_input);
  double get_index_time = t0.Passed();

  // 1. Extract the miss data
  Timer t1;

  cache_manager->GPUExtractMissData(train_feat->MutableData(),
                                    trainer_output_miss_src_index, trainer_output_miss_dst_index,
                                    num_output_miss, stream);
  trainer_device->StreamSync(trainer_ctx, stream);
  double extract_miss_time = t1.Passed();

  // 2. Combine cache data
  Timer t2;
  cache_manager->CombineCacheData(train_feat->MutableData(),
                                  trainer_output_cache_src_index, trainer_output_cache_dst_index,
                                  num_output_cache, stream);
  trainer_device->StreamSync(trainer_ctx, stream);

  double combine_cache_time = t2.Passed();
  task->input_feat = train_feat;

  task->miss_cache_index.miss_src_index = nullptr;
  task->miss_cache_index.miss_dst_index = nullptr;
  task->miss_cache_index.cache_src_index = nullptr;
  task->miss_cache_index.cache_dst_index = nullptr;

  auto miss_nbytes = GetTensorBytes(feat_type, {num_output_miss, feat_dim});
  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes, train_feat->NumBytes());
  Profiler::Get().LogStep(task->key, kLogL1MissBytes, miss_nbytes);
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime, get_index_time); // t0
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime, extract_miss_time); // t1
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime, combine_cache_time); // t2
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes, train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(task->key, kLogEpochMissBytes, miss_nbytes);

  LOG(DEBUG) << "DoGPUCacheFeatureCopy: process task with key " << task->key;
}

} // dist
} // common
} // samgraph
