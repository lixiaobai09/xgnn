#include "dist_graph.h"

#include <sys/mman.h>
#include <sys/unistd.h>
#include <sys/wait.h>

#include <cstring>
#include <iomanip>
#include <set>
#include <algorithm>

#include "../device.h"
#include "../timer.h"


namespace samgraph {
namespace common {
namespace cuda {

std::shared_ptr<DistGraph> DistGraph::_inst = nullptr;

void DistGraph::_DatasetPartition(const Dataset *dataset, int sampler_id) {
  auto indptr_data = dataset->indptr->CPtr<IdType>();
  auto indices_data = dataset->indices->CPtr<IdType>();
  auto ctx = _group_configs[sampler_id].ctx;
  IdType num_node = dataset->num_node;
  IdType num_part = _group_configs[sampler_id].ctx_group.size();
  IdType part_id = _group_configs[sampler_id].part_id;
  IdType part_edge_count = 0;
  for (IdType i = part_id; i < num_node; i += num_part) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    part_edge_count += num_edge;
  }
  _part_indptr.clear();
  _part_indptr.resize(num_part, nullptr);
  _part_indices.clear();
  _part_indices.resize(num_part, nullptr);

  IdType indptr_size = (num_node / num_part +
      (part_id < num_node % num_part? 1 : 0) + 1);
  _part_indptr[part_id] = Tensor::Empty(kI32, {indptr_size}, CPU(),
      "indptr in device:" + std::to_string(ctx.device_id));
  _part_indices[part_id] = Tensor::Empty(kI32, {part_edge_count}, CPU(),
      "indices in device:" + std::to_string(ctx.device_id));
  part_edge_count = 0;

  for (IdType i = part_id; i < num_node; i += num_part) {
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

  _part_indptr[part_id] = Tensor::CopyTo(_part_indptr[part_id], ctx,
      nullptr, Constant::kAllocNoScale);
  _part_indices[part_id] = Tensor::CopyTo(_part_indices[part_id], ctx,
      nullptr, Constant::kAllocNoScale);
}

void DistGraph::DatasetLoad(Dataset *dataset, int sampler_id,
    Context sampler_ctx) {

  CHECK(sampler_ctx == _group_configs[sampler_id].ctx);

  _sampler_id = sampler_id;
  _DatasetPartition(dataset, sampler_id);

  auto ctx_group = _group_configs[sampler_id].ctx_group;
  IdType part_id = _group_configs[sampler_id].part_id;
  IdType num_part = ctx_group.size();

  auto DataIpcShare = [&](std::vector<TensorPtr> &part_data,
      std::vector<size_t> part_size_vec,
      std::string name) {

    {
      // share self data to others
      CHECK(sampler_ctx == part_data[part_id]->Ctx());
      auto shared_data = part_data[part_id]->CPtr<IdType>();
      cudaIpcMemHandle_t &mem_handle =
        _shared_data->mem_handle[sampler_ctx.device_id];
      CUDA_CALL(cudaIpcGetMemHandle(&mem_handle, (void*)shared_data));
    }

    _Barrier();

    // receive data from others
    for (int i = 0; i < num_part; ++i) {
      if (i == part_id) {
        continue;
      }
      CHECK(part_data[i] == nullptr);
      auto ctx = ctx_group[i];
      cudaIpcMemHandle_t &mem_handle = _shared_data->mem_handle[ctx.device_id];
      void *ptr;
      CUDA_CALL(cudaIpcOpenMemHandle(
            &ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
      part_data[i] = Tensor::FromBlob(ptr, kI32, {part_size_vec[i]}, ctx,
          name + " in device:" + std::to_string(ctx.device_id));
    }

  };

  IdType num_node = dataset->num_node;
  std::vector<size_t> part_size_vec(num_part);
  for (size_t i = 0; i < num_part; ++i) {
    part_size_vec[i] = (num_node / num_part +
        (i < num_node % num_part? 1 : 0) + 1);
  }
  DataIpcShare(_part_indptr, part_size_vec, "dataset part indptr");

  part_size_vec.clear();
  part_size_vec.resize(num_part, 0);
  auto indptr_data = dataset->indptr->CPtr<IdType>();
  for (IdType i = 0; i < num_node; ++i) {
    IdType num_edge = indptr_data[i + 1] - indptr_data[i];
    IdType part_id = (i % num_part);
    part_size_vec[part_id] += num_edge;
  }
  DataIpcShare(_part_indices, part_size_vec, "dataset part indices");

  CUDA_CALL(cudaMalloc((void **)&_d_part_indptr, num_part * sizeof(IdType *)));
  CUDA_CALL(cudaMalloc((void **)&_d_part_indices, num_part * sizeof(IdType *)));

  IdType **h_part_indptr, **h_part_indices;
  CUDA_CALL(cudaMallocHost(&h_part_indptr, num_part * sizeof(IdType*)));
  CUDA_CALL(cudaMallocHost(&h_part_indices, num_part * sizeof(IdType*)));
  for (IdType i = 0; i < num_part; i++) {
    h_part_indptr[i] = _part_indptr[i]->Ptr<IdType>();
    h_part_indices[i] = _part_indices[i]->Ptr<IdType>();
  }
  CUDA_CALL(cudaMemcpy(_d_part_indptr, h_part_indptr, sizeof(IdType *) * num_part, cudaMemcpyDefault));
  CUDA_CALL(cudaMemcpy(_d_part_indices, h_part_indices, sizeof(IdType *) * num_part, cudaMemcpyDefault));

  CUDA_CALL(cudaFreeHost(h_part_indptr));
  CUDA_CALL(cudaFreeHost(h_part_indices));

  _num_node = dataset->num_node;
}

DeviceDistGraph DistGraph::DeviceHandle() const {
  return DeviceDistGraph(
      _d_part_indptr, _d_part_indices,
      _group_configs[_sampler_id].ctx_group.size(),
      _num_node);
}

DistGraph::DistGraph(std::vector<Context> ctxes) {
  // TODO: from ctxes to get graph parts configs
  // bala bala ...
  std::vector<Context> ctx_group = ctxes;

  PartitionSolver solver(ctx_group);
  auto configs = solver.solve();
  for (auto &config : configs) {
    LOG(INFO) << config;
  }

  _group_configs.clear();
  for (int i = 0; i < ctxes.size(); ++i) {
    _group_configs.emplace_back(ctxes[i], i, ctx_group);
  }


  int num_worker = ctxes.size();
  _sampler_id = static_cast<int>(Constant::kEmptyKey);

  _shared_data = static_cast<SharedData*>(mmap(NULL, sizeof(SharedData),
                      PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
  CHECK_NE(_shared_data, MAP_FAILED);
  pthread_barrierattr_t attr;
  pthread_barrierattr_init(&attr);
  pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_barrier_init(&_shared_data->barrier, &attr, num_worker);
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
  pthread_barrier_destroy(&dist_graph->_shared_data->barrier);
  munmap(dist_graph->_shared_data, sizeof(SharedData));
}

void DistGraph::Create(std::vector<Context> ctxes) {
  CHECK(_inst == nullptr);
  _inst = std::shared_ptr<DistGraph>(
      new DistGraph(ctxes), Release);
}


PartitionSolver::PartitionSolver(const std::vector<Context> &ctx_group) 
  : _ctx_group(ctx_group) {
  std::set<int> set;
  for (auto&ctx : ctx_group) {
    set.insert(ctx.device_id);
  }
  CHECK_EQ(set.size(), ctx_group.size());
  CHECK_EQ(*set.rbegin() + 1, set.size());
  DetectTopo();
}

void PartitionSolver::DetectTopo() {
  // shared memory for transfer detect result
  LinkTopoInfo *shared_data = (LinkTopoInfo*)mmap(NULL, sizeof(LinkTopoInfo), 
    PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  int pid = fork();
  CHECK(pid != -1);
  if (pid == 0) {
    DetectTopo_child(shared_data);
  } else {
    int wstatus;
    waitpid(pid, &wstatus, 0);
    if (WEXITSTATUS(wstatus) != 0 || WIFSIGNALED(wstatus)) {
      CHECK(false);
    }
    std::memcpy(&_topo_info, shared_data, sizeof(LinkTopoInfo));
  }
  munmap(shared_data, sizeof(LinkTopoInfo));

  LOG(INFO) << "DetectTopo Done";
}

std::vector<PartitionSolver::GroupConfig> PartitionSolver::solve() const  {
  std::set<IdType> parts[kMaxDevice];
  IdType access_matrix[kMaxDevice][kMaxDevice] = {0};
  IdType access_cnt[kMaxDevice][kMaxDevice] = {0};
  
  for (int i = 0; i < _ctx_group.size(); i++) {
    int device = _ctx_group[i].device_id;
    parts[device].insert(device);
  }
  
  auto neighborParts = [&](IdType device, IdType part) -> std::vector<IdType> {
    std::vector<IdType> peer_vec;
    for (int peer = 0; peer < this->_ctx_group.size(); peer++) {
      if (this->_topo_info.nvlink_matrix[device][peer]) {
        if (parts[peer].find(part) != parts[peer].end()) {
          peer_vec.push_back(peer);
        } 
      }
    }
    return peer_vec;
  };

  for (int device = 0; device < _ctx_group.size(); device++) {
    std::vector<IdType> miss_parts;
    for (int part = 0; part < _ctx_group.size(); part++) {
      auto peers = neighborParts(device, part);
      if (peers.empty()) {
        miss_parts.push_back(part);
      } else {
        auto peer = ChoosePeer(parts, access_cnt, device, peers, true);
        access_cnt[device][peer]++;
        access_matrix[device][part] = peer;
      }
    }
    for (auto part : miss_parts) {
      IdType rep_peer = FindPalcement(parts, access_cnt, device, part);
      parts[rep_peer].insert(part);
      access_cnt[device][rep_peer]++;
      access_matrix[device][part] = rep_peer;
    }
  }
  std::vector<PartitionSolver::GroupConfig> configs;
  for (int i = 0; i < _ctx_group.size(); i++) {
    auto ctx = _ctx_group[i];
    IdType device = ctx.device_id;
    std::vector<IdType> part_ids(parts[device].begin(), parts[device].end());
    std::vector<Context> group;
    for (int j = 0; j < _ctx_group.size(); j++) {
      group.push_back(GPU(access_matrix[device][j]));
    }
    configs.emplace_back(ctx, part_ids, group);
  }
  return configs;
}

IdType PartitionSolver::FindPalcement(
  const std::set<IdType> parts[], IdType access_cnt[][kMaxDevice],
  IdType device, IdType part) const {
  std::vector<IdType> peers;
  for (IdType peer = 0; peer < _ctx_group.size(); peer++) {
    if (_topo_info.nvlink_matrix[device][peer]) {
      peers.push_back(peer);
    }
  }
  CHECK(peers.size() > 0);
  return ChoosePeer(parts, access_cnt, device, peers, false);
}

IdType PartitionSolver::ChoosePeer(
  const std::set<IdType> parts[], IdType access_cnt[][kMaxDevice],
  IdType device, std::vector<IdType> peers, bool exist) const {
  if (peers.empty()) {
    return -1;
  }
  std::vector<std::pair<IdType, double>> weight;
  for (auto peer : peers) {
    double bw = _topo_info.bandwitdh_matrix[device][peer];
    bw /= (access_cnt[device][peer] + exist);
    weight.push_back({parts[peer].size(), bw});
  }
  std::sort(peers.begin(), peers.end(), [&](IdType x, IdType y) {
    if (weight[x].first != weight[y].first) {
      return weight[x].first < weight[y].first;
    } else {
      return weight[x].second > weight[y].second;
    }
  });
  return peers.front();
}

void PartitionSolver::DetectTopo_child(LinkTopoInfo *topo_info) {
  // 128M buffer for bandwidth test to detect backbone link
  size_t nbytes = (1<<27);
  IdType *buffers[kMaxDevice], *buffersD2D[kMaxDevice];
  cudaStream_t stream[kMaxDevice];
  for (int i = 0; i < _ctx_group.size(); i++) {
    int device = _ctx_group[i].device_id;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMalloc(&buffers[device], nbytes));
    CUDA_CALL(cudaMalloc(&buffersD2D[device], nbytes));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream[device], cudaStreamNonBlocking));
    for (int j = 0; j < _ctx_group.size(); j++) {
      int peer = _ctx_group[j].device_id;
      topo_info->bandwitdh_matrix[device][peer] = 0;
      if (device == peer) {
        topo_info->nvlink_matrix[device][peer] = 1;
        continue;
      }
      int can_access = false;
      CUDA_CALL(cudaDeviceCanAccessPeer(&can_access, device, peer));
      if (!can_access) {
        topo_info->nvlink_matrix[device][peer] = 0;
      } else {
        topo_info->nvlink_matrix[device][peer] = 1;
      }
    }
  }
  for (int i = 0; i < _ctx_group.size(); i++) {
    int device = _ctx_group[i].device_id;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemcpyAsync(buffers[device], buffersD2D[device], nbytes, cudaMemcpyDefault, stream[device]));
    CUDA_CALL(cudaStreamSynchronize(stream[device]));
    for (int j = 0; j < _ctx_group.size(); j++) {
      int peer = _ctx_group[j].device_id;
      if (device != peer && topo_info->nvlink_matrix[device][peer]) {
        CUDA_CALL(cudaDeviceEnablePeerAccess(peer, 0));
      }
    }
    for (int j = 0; j < _ctx_group.size(); j++) {
      int peer = _ctx_group[j].device_id;
      if (topo_info->nvlink_matrix[device][peer]) {
        Timer t0;
        CUDA_CALL(cudaMemcpyAsync(buffers[device], buffersD2D[peer], nbytes, cudaMemcpyDefault, stream[device]));
        CUDA_CALL(cudaStreamSynchronize(stream[device]));
        auto sec = t0.Passed();
        if (device == peer) {
          topo_info->bandwitdh_matrix[device][peer] = 2 * nbytes / sec / 1e9;
        } else {
          topo_info->bandwitdh_matrix[device][peer] = nbytes / sec / 1e9;
        }
      }
    }
    for (int j = 0; j < _ctx_group.size(); j++) {
      int peer = _ctx_group[j].device_id;
      if (device != peer && topo_info->nvlink_matrix[device][peer]) {
        CUDA_CALL(cudaDeviceDisablePeerAccess(peer));
      }
    }
  }

  // release resouce
  for (int i = 0; i < _ctx_group.size(); i++) {
    auto device = _ctx_group[i].device_id;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaStreamDestroy(stream[device]));
    CUDA_CALL(cudaFree(buffers[device]));
    CUDA_CALL(cudaFree(buffersD2D[device]));
    for (int j = 0; j < _ctx_group.size(); j++) {
      auto peer = _ctx_group[j].device_id;
      if (device == peer)
        continue;
    }
  }

  std::stringstream ss;
  ss << "Topology Detect Debug: \n";
  for (int i = 0; i < _ctx_group.size(); i++) {
    for (int j = 0; j < _ctx_group.size(); j++) {
      ss << std::setw(8) << std::fixed << std::setprecision(1) << topo_info->bandwitdh_matrix[i][j] << " ";
    }
    ss << "\n";
  }
  LOG(INFO) << ss.str();

  munmap(topo_info, sizeof(LinkTopoInfo));
  exit(0);
}

std::ostream& operator<<(std::ostream &os, const PartitionSolver::GroupConfig &config) {
  std::stringstream part_ss;
  std::stringstream peer_ss;
  for (auto part : config.part_ids)
    part_ss << part << " ";
  for (auto &ctx : config.ctx_group)
    peer_ss << ctx.device_id << " ";
  os << "GPU[" << config.ctx.device_id << "]"
     << " part: [ " << part_ss.str() << "]"
     << " peer: [ " << peer_ss.str() << "]";
  return os;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
