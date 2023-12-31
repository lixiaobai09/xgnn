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

#include "operation.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <regex>

#include "common.h"
#include "constant.h"
#include "engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"
#include "cuda/cuda_engine.h"
#include "cuda/dist_graph.h"
#include "cuda/ics22_song_dist_feature.h"
#include "dist/dist_engine.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_config(const char **config_keys, const char **config_values,
                     const size_t num_config_items) {
  using RC = RunConfig;
  CHECK(!RC::is_configured);

  std::unordered_map<std::string, std::string> configs;

  for (size_t i = 0; i < num_config_items; i++) {
    std::string k(config_keys[i]);
    std::string v(config_values[i]);
    configs[k] = v;
  }
  samgraph_config_from_map(configs);
}

void samgraph_config_from_map(std::unordered_map<std::string, std::string>& configs) {
  using RC = RunConfig;
  CHECK(!RC::is_configured);

  CHECK(configs.count("dataset_path"));
  CHECK(configs.count("_arch"));
  CHECK(configs.count("_sample_type"));
  CHECK(configs.count("batch_size"));
  CHECK(configs.count("num_epoch"));
  CHECK(configs.count("_cache_policy"));
  CHECK(configs.count("cache_percentage"));
  CHECK(configs.count("max_sampling_jobs"));
  CHECK(configs.count("max_copying_jobs"));
  CHECK(configs.count("omp_thread_num"));
  CHECK(configs.count("num_layer"));
  CHECK(configs.count("num_hidden"));
  CHECK(configs.count("lr"));
  CHECK(configs.count("dropout"));

  RC::raw_configs = configs;
  RC::dataset_path = configs["dataset_path"];
  RC::run_arch = static_cast<RunArch>(std::stoi(configs["_arch"]));
  RC::sample_type = static_cast<SampleType>(std::stoi(configs["_sample_type"]));
  RC::batch_size = std::stoull(configs["batch_size"]);
  RC::num_epoch = std::stoull(configs["num_epoch"]);
  RC::cache_policy =
      static_cast<CachePolicy>(std::stoi(configs["_cache_policy"]));
  RC::cache_percentage = std::stod(configs["cache_percentage"]);

  RC::max_sampling_jobs = std::stoull(configs["max_sampling_jobs"]);
  RC::max_copying_jobs = std::stoull(configs["max_copying_jobs"]);
  RC::omp_thread_num = std::stoi(configs["omp_thread_num"]);
  RC::hiddem_dim = std::stoi(configs["num_hidden"]);
  RC::lr = std::stod(configs["lr"]);
  RC::dropout = std::stod(configs["dropout"]);
  RC::num_layer = std::stoull(configs["num_layer"]);

  switch (RC::run_arch) {
    case kArch0:
    case kArch1:
    case kArch2:
    case kArch3:
    case kArch4:
      CHECK(configs.count("sampler_ctx"));
      CHECK(configs.count("trainer_ctx"));
      RC::sampler_ctx = Context(configs["sampler_ctx"]);
      RC::trainer_ctx = Context(configs["trainer_ctx"]);
      break;
    case kArch5:
      CHECK(configs.count("num_sample_worker"));
      CHECK(configs.count("num_train_worker"));
      RC::num_sample_worker = std::stoull(configs["num_sample_worker"]);
      RC::num_train_worker = std::stoull(configs["num_train_worker"]);
      if (!configs.count("have_switcher")) {
        configs["have_switcher"] = "0";
      }
      RC::have_switcher = std::stoi(configs["have_switcher"]);
      break;
    case kArch6:
      CHECK(configs.count("num_worker"));
      RC::num_sample_worker = std::stoull(configs["num_worker"]);
      RC::num_train_worker = std::stoull(configs["num_worker"]);
      RC::num_worker = std::stoull(configs["num_worker"]);
      break;
    case kArch7:
      CHECK(configs.count("worker_id"));
      CHECK(configs.count("num_worker"));
      CHECK(configs.count("sampler_ctx"));
      CHECK(configs.count("trainer_ctx"));
      RC::worker_id = std::stoull(configs["worker_id"]);
      RC::num_worker = std::stoull(configs["num_worker"]);
      RC::sampler_ctx = Context(configs["sampler_ctx"]);
      RC::trainer_ctx = Context(configs["trainer_ctx"]);
      break;
    case kArch9:
      CHECK(configs.count("num_sample_worker"));
      CHECK(configs.count("num_train_worker"));
      RC::num_sample_worker = std::stoi(configs["num_sample_worker"]);
      RC::num_train_worker = std::stoi(configs["num_train_worker"]);
      CHECK(configs.count("unified_memory"));
      CHECK(configs.count("unified_memory_ctx"));
      break;
    default:
      CHECK(false);
  }

  if (RC::sample_type != kRandomWalk) {
    // configure khop
    CHECK(configs.count("num_fanout"));
    CHECK(configs.count("fanout"));

    size_t num_fanout = std::stoull(configs["num_fanout"]);
    std::stringstream ss(configs["fanout"]);
    for (size_t i = 0; i < num_fanout; i++) {
      size_t fanout;
      ss >> fanout;
      RC::fanout.push_back(fanout);
    }
  } else {
    // configure random walk
    CHECK(configs.count("random_walk_length"));
    CHECK(configs.count("random_walk_restart_prob"));
    CHECK(configs.count("num_random_walk"));
    CHECK(configs.count("num_neighbor"));

    RC::random_walk_length = std::stoull(configs["random_walk_length"]);
    RC::random_walk_restart_prob =
        std::stod(configs["random_walk_restart_prob"]);
    RC::num_random_walk = std::stoull(configs["num_random_walk"]);
    RC::num_neighbor = std::stoull(configs["num_neighbor"]);
    RC::fanout = std::vector<size_t>(RC::num_layer, RC::num_neighbor);
  }

  if (configs.count("barriered_epoch") > 0) {
    RunConfig::barriered_epoch = std::stoi(configs["barriered_epoch"]);
    LOG(DEBUG) << "barriered_epoch=" << RunConfig::barriered_epoch;
  } else {
    RunConfig::barriered_epoch = 0;
  }

  if (configs.count("presample_epoch") > 0) {
    RunConfig::presample_epoch = std::stoi(configs["presample_epoch"]);
    LOG(DEBUG) << "presample_epoch=" << RunConfig::presample_epoch;
  } else {
    RunConfig::presample_epoch = 0;
  }

  if (configs.count("use_dist_graph") > 0) {
    RC::dist_graph_percentage = std::stod(configs["use_dist_graph"]);
    if (RC::dist_graph_percentage > 0.0) {
      RC::use_dist_graph = true;
      LOG(DEBUG) << "use_dist_graph with dist_graph_percentage = "
        << RC::dist_graph_percentage;
    }

    if (configs.count("dist_graph_part_cpu") > 0) {
      RC::dist_graph_part_cpu = std::stoi(configs["dist_graph_part_cpu"]);
      LOG(DEBUG) << "dist graph partition graph to CPU, GPU/(GPU+CPU) = 1/" << RC::dist_graph_part_cpu;
    }
  }

  if (configs.count("part_cache") > 0 && configs["part_cache"] == "True") {
    if (RC::run_arch == kArch6) {
      RC::part_cache = true;
    } else {
      LOG(FATAL) << "partition cache can only be used in arch6";
    }
  }

  if (configs.count("clique_size") > 0) {
    RC::clique_size = std::stoi(configs["clique_size"]);
  }

  if (configs.count("ics22_song_alpha") > 0) {
    RC::ics22_song_alpha = std::stod(configs["ics22_song_alpha"]);
  }

  if (configs.count("use_ics22_song_solver") > 0
      && configs["use_ics22_song_solver"] == "True") {
    CHECK(RC::use_dist_graph == true)
      << "use ics22_song_solver need to open dist_graph";
    CHECK(RC::part_cache == false)
      << "use ics22_song_solver need to close part cache";
    CHECK(RC::clique_size > 0)
      << "use ics22_song_solver need to set clique size";
    RC::use_ics22_song_solver = true;
  }

  if (configs.count("gpu_extract") > 0 && configs["gpu_extract"] == "True") {
    if (RC::run_arch == kArch6) {
      RC::gpu_extract = true;
    } else {
      LOG(WARNING) << "gpu extract is only supported in arch6";
    }
  }

  if(configs.count("unified_memory") > 0 && configs["unified_memory"] == "True") {
    RunConfig::unified_memory = true;
    LOG(DEBUG) << "unified_memory=True";
  }
  if(configs.count("unified_memory_percentage") > 0) {
    std::stringstream ss(configs["unified_memory_percentage"]);
    double percent;
    RunConfig::unified_memory_percentages.clear();
    while (ss >> percent) {
      RunConfig::unified_memory_percentages.push_back(percent);
    }
    CHECK(std::abs(std::accumulate(RunConfig::unified_memory_percentages.begin(), RunConfig::unified_memory_percentages.end(), 0.0) - 1) < 1e-8);
  }
  if(configs.count("um_policy") > 0) {
    if(configs["um_policy"] == "default") {
      RunConfig::unified_memory_policy = UMPolicy::kDefault;
    } else if(configs["um_policy"] == "degree") {
      RunConfig::unified_memory_policy = UMPolicy::kDegree;
    } else if(configs["um_policy"] == "trainset") {
      RunConfig::unified_memory_policy = UMPolicy::kTrainset;
    } else if(configs["um_policy"] == "random") {
      RunConfig::unified_memory_policy = UMPolicy::kRandom;
    } else if(configs["um_policy"] == "presample") {
      RunConfig::unified_memory_policy = UMPolicy::kPreSample;
    } else {
      LOG(FATAL) << "bad um_policy";
    }
    LOG(INFO) << "unified_memory_policy" << " " << configs["um_policy"];
  }
  if(RunConfig::unified_memory == true && RunConfig::run_arch != kArch6) {
    RunConfig::unified_memory_ctxes.clear();
    if (configs.count("unified_memory_ctx") > 0) {
      std::stringstream ss(configs["unified_memory_ctx"]);
      std::regex cuda_reg("cuda:([0-9]+)");
      std::string ctx;
      while(ss >> ctx) {
        std::smatch device_id;
        if (std::regex_match(ctx, device_id, cuda_reg)) {
          RunConfig::unified_memory_ctxes.push_back(GPU(std::stoi(device_id[1])));
        } else if (ctx == "cpu") {
          RunConfig::unified_memory_ctxes.push_back(CPU());
        }
      }
    } else {
      RunConfig::unified_memory_ctxes.push_back(RunConfig::sampler_ctx);
      RunConfig::unified_memory_ctxes.push_back(CPU());
    }
    LOG(INFO) << "unified_memory_ctxes : "
              << std::accumulate(RunConfig::unified_memory_ctxes.begin(), RunConfig::unified_memory_ctxes.end(),
                  std::string{""},
                [](std::string init, const Context& first) {
                  std::stringstream ss;
                  ss << first << " ";
                  return init + ss.str();
                });
    LOG(INFO) << "unified_memory_percentages : "
              << std::accumulate(RunConfig::unified_memory_percentages.begin(), RunConfig::unified_memory_percentages.end(),
                  std::string{""},
                [](std::string init, double percent) {
                  return init + std::to_string(percent) + " ";
                });
    CHECK(RunConfig::unified_memory_ctxes.size() == RunConfig::unified_memory_percentages.size());
  }
  // check unified memory based on run arch
  if (RC::run_arch == RunArch::kArch9) {
    if (!RC::unified_memory) {
      LOG(FATAL) << "Arch9 should use unified memory for sampling";
    }
    if (RC::unified_memory_ctxes.size() != RC::num_sample_worker) {
      LOG(FATAL) << "UM sampler worker conflicts";
    }
    for (auto ctx : RC::unified_memory_ctxes) {
      if (ctx.device_type != DeviceType::kGPU && ctx.device_type != DeviceType::kGPU_UM) {
        LOG(FATAL) << "UM sampler ctx should be GPU, but find " << ctx;
      }
    }
  } else if (RC::run_arch == RunArch::kArch3) {
    if (RunConfig::unified_memory) {
      CHECK(RunConfig::unified_memory_ctxes[0] == RunConfig::sampler_ctx);
    }
  }

  RC::LoadConfigFromEnv();
  LOG(INFO) << "Use " << RunConfig::sample_type << " sampling algorithm";

  RC::is_configured = true;
}

void samgraph_init() {
  CHECK(RunConfig::is_configured);
  Engine::Create();
  Engine::Get()->Init();

  LOG(INFO) << "SamGraph has been initialied successfully";
}

void samgraph_start() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  if (RunConfig::option_profile_cuda) {
    CUDA_CALL(cudaProfilerStart());
  }

  Engine::Get()->Start();
  LOG(INFO) << "SamGraph has been started successfully";
}

size_t samgraph_num_epoch() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->NumEpoch();
}

size_t samgraph_steps_per_epoch() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->NumStep();
}

size_t samgraph_num_class() {
  // CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->GetGraphDataset()->num_class;
}

size_t samgraph_feat_dim() {
  // CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->GetGraphDataset()->feat->Shape().at(1);
}

uint64_t samgraph_get_next_batch() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());

  // uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Engine::Get()->SetGraphBatch(nullptr);
  auto graph = Engine::Get()->GetGraphPool()->GetGraphBatch();
  uint64_t key = graph->key;

  LOG(DEBUG) << "samgraph_get_next_batch encodeKey with key " << key;
  Engine::Get()->SetGraphBatch(graph);

  return key;
}

void samgraph_sample_once() { Engine::Get()->RunSampleOnce(); }

size_t samgraph_get_graph_num_src(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_src;
}

size_t samgraph_get_graph_num_dst(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_dst;
}

size_t samgraph_get_graph_num_edge(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_edge;
}

void samgraph_shutdown() {
  Engine::Get()->Shutdown();
  if (RunConfig::option_profile_cuda) {
    CUDA_CALL(cudaProfilerStop());
  }
  LOG(INFO) << "SamGraph has been completely shutdown now";
}

void samgraph_log_step(uint64_t epoch, uint64_t step, int item, double val) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Profiler::Get().LogStep(key, static_cast<LogStepItem>(item), val);
}
void samgraph_log_step_by_key(uint64_t key, int item, double val) {
  CHECK_LT(item, kNumLogStepItems);
  Profiler::Get().LogStep(key, static_cast<LogStepItem>(item), val);
}

void samgraph_log_step_add(uint64_t epoch, uint64_t step, int item,
                           double val) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Profiler::Get().LogStepAdd(key, static_cast<LogStepItem>(item), val);
}

void samgraph_log_epoch_add(uint64_t epoch, int item, double val) {
  CHECK_LT(item, kNumLogEpochItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, 0);
  Profiler::Get().LogEpochAdd(key, static_cast<LogEpochItem>(item), val);
}

double samgraph_get_log_init_value(int item) {
  CHECK_LT(item, kNumLogInitItems);
  return Profiler::Get().GetLogInitValue(static_cast<LogInitItem>(item));
}

double samgraph_get_log_step_value(uint64_t epoch, uint64_t step, int item) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  return Profiler::Get().GetLogStepValue(key, static_cast<LogStepItem>(item));
}
double samgraph_get_log_step_value_by_key(uint64_t key, int item) {
  CHECK_LT(item, kNumLogStepItems);
  return Profiler::Get().GetLogStepValue(key, static_cast<LogStepItem>(item));
}

double samgraph_get_log_epoch_value(uint64_t epoch, int item) {
  CHECK_LT(item, kNumLogEpochItems);
  return Profiler::Get().GetLogEpochValue(epoch,
                                          static_cast<LogEpochItem>(item));
}

void samgraph_report_init() {
  Profiler::Get().ReportInit();
  std::cout.flush();
}

void samgraph_report_step(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStep(epoch, step);
}

void samgraph_report_step_average(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStepAverage(epoch, step);
  std::cout.flush();
}

void samgraph_report_epoch(uint64_t epoch) {
  Profiler::Get().ReportEpoch(epoch);
}

void samgraph_report_epoch_average(uint64_t epoch) {
  Profiler::Get().ReportEpochAverage(epoch);
  std::cout.flush();
}

void samgraph_report_node_access() {
  if (RunConfig::option_log_node_access_simple) {
    Profiler::Get().ReportNodeAccessSimple();
  }
  if (RunConfig::option_log_node_access) {
    Profiler::Get().ReportNodeAccess();
  }
  std::cout.flush();
}

void samgraph_trace_step_begin(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item), us);
}

void samgraph_trace_step_end(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item), us);
}

void samgraph_trace_step_begin_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item),
                                 t.TimePointMicro());
}

void samgraph_trace_step_end_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item),
                               t.TimePointMicro());
}

void samgraph_dump_trace() {
  Profiler::Get().DumpTrace(std::cerr);
  std::cerr.flush();
}

void samgraph_forward_barrier() { Engine::Get()->ForwardBarrier(); }

void samgraph_data_init() {
  CHECK(RunConfig::is_configured);
  Engine::Create();
  Engine::Get()->Init();

  if (RunConfig::run_arch == kArch6
      && (RunConfig::use_dist_graph
          || RunConfig::part_cache
          || RunConfig::use_ics22_song_solver)) {
    std::vector<Context> ctxes(RunConfig::num_worker);
    for (int i = 0; i < static_cast<int>(RunConfig::num_worker); ++i) {
      ctxes[i] = Context{kGPU, i};
    }
    if (RunConfig::use_dist_graph && RunConfig::use_ics22_song_solver) {
      cuda::ICS22SongDistGraph::Create(ctxes, RunConfig::clique_size,
          Engine::Get()->GetGraphDataset(),
          RunConfig::ics22_song_alpha,
          RunConfig::cache_percentage * Engine::Get()->GetGraphDataset()->num_node);
    } else {
      cuda::DistGraph::Create(ctxes);
    }
  }

  LOG(INFO) << "SamGraph data has been initialized successfully";
}

void samgraph_sample_init(int worker_id, const char*ctx) {
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->SampleInit(worker_id, Context(std::string(ctx)));

  LOG(INFO) << "SamGraph sample has been initialized successfully";
}

void samgraph_um_sample_init(int num_workers) {
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->UMSampleInit(num_workers);

  LOG(INFO) << "SamGraph um sample has been initialized successfully";
}

void samgraph_train_init(int worker_id, const char*ctx) {
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->TrainInit(worker_id, Context(std::string(ctx)), dist::DistType::Extract);

  LOG(INFO) << "SamGraph train has been initialized successfully";
}

void samgraph_extract_start(int count) {
  dist::DistEngine::Get()->StartExtract(count);
  LOG(INFO) << "SamGraph extract background thread start successfully";
}

void samgraph_switch_init(int worker_id, const char*ctx, double cache_percentage) {
  RunConfig::cache_percentage = cache_percentage;
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->TrainInit(worker_id, Context(std::string(ctx)), dist::DistType::Switch);

  LOG(INFO) << "SamGraph switch has been initialized successfully";
}

size_t samgraph_num_local_step() {
  return Engine::Get()->NumLocalStep();
}

int samgraph_wait_one_child() {
  int child_stat;
  pid_t pid = waitpid(-1, &child_stat, 0);
  if (WEXITSTATUS(child_stat) != 0) {
    LOG(ERROR) << "detect a terminated child " << pid << ", status is "
               << WEXITSTATUS(child_stat);
    return 1;
  } else if (WIFSIGNALED(child_stat)) {
    LOG(ERROR) << "detect an abnormal terminated child, signal is " << strsignal(WTERMSIG(child_stat));
    return 1;
  } else return 0;
}

}  // extern "c"

}  // namespace common
}  // namespace samgraph
