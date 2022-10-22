#include "dist_graph.h"

#include <sys/mman.h>
#include <sys/unistd.h>
#include <sys/wait.h>

#include <cstring>
#include <iomanip>
#include <set>
#include <algorithm>
#include <fstream>
#include <regex>

#include "../device.h"
#include "../timer.h"


namespace samgraph {
namespace common {
namespace cuda {

std::shared_ptr<DistGraph> DistGraph::_inst = nullptr;

void DistGraph::_DatasetPartition(const Dataset *dataset, Context ctx,
    IdType part_id, IdType num_part) {
  auto indptr_data = dataset->indptr->CPtr<IdType>();
  auto indices_data = dataset->indices->CPtr<IdType>();
  IdType num_node = dataset->num_node;
  IdType part_edge_count = 0;
  for (IdType i = part_id; i < num_node; i += num_part) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    part_edge_count += num_edge;
  }

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

  auto part_ids = _group_configs[sampler_id].part_ids;
  auto ctx_group = _group_configs[sampler_id].ctx_group;
  IdType num_part = ctx_group.size();
  _part_indptr.clear();
  _part_indptr.resize(num_part, nullptr);
  _part_indices.clear();
  _part_indices.resize(num_part, nullptr);

  for (IdType part_id : part_ids) {
    _DatasetPartition(dataset, sampler_ctx, part_id, num_part);
  }

  auto DataIpcShare = [&](std::vector<TensorPtr> &part_data,
      std::vector<size_t> part_size_vec,
      std::string name) {

    {
      for (IdType part_id : part_ids) {
        // share self data to others
        CHECK(sampler_ctx == part_data[part_id]->Ctx());
        CHECK(part_size_vec[part_id] == part_data[part_id]->Shape()[0]);
        auto shared_data = part_data[part_id]->CPtr<IdType>();
        cudaIpcMemHandle_t &mem_handle =
          _shared_data->mem_handle[sampler_ctx.device_id][part_id];
        CUDA_CALL(cudaIpcGetMemHandle(&mem_handle, (void*)shared_data));
      }
    }
    _Barrier();

    // receive data from others
    for (int i = 0; i < num_part; ++i) {
      if (part_data[i] != nullptr) {
        continue;
      }
      auto ctx = ctx_group[i];
      cudaIpcMemHandle_t &mem_handle = _shared_data->mem_handle[ctx.device_id][i];
      void *ptr;
      CUDA_CALL(cudaIpcOpenMemHandle(
            &ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
      part_data[i] = Tensor::FromBlob(ptr, kI32, {part_size_vec[i]}, ctx,
          name + " in device:" + std::to_string(ctx.device_id));
    }
    _Barrier();

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
    IdType tmp_part_id = (i % num_part);
    part_size_vec[tmp_part_id] += num_edge;
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

  PartitionSolver solver(ctxes);
  _group_configs = solver.solve();
  for (auto &config : _group_configs) {
    LOG(INFO) << config;
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
      ss << std::setw(8) << std::fixed << std::setprecision(1) << _topo_info.bandwitdh_matrix[i][j] << " ";
    }
    ss << "\n";
  }
  LOG(INFO) << ss.str();
}

std::vector<DistGraph::GroupConfig> PartitionSolver::solve() const  {
  std::set<IdType> parts[kMaxDevice];
  IdType access_matrix[kMaxDevice][kMaxDevice] = {0};
  IdType access_cnt[kMaxDevice][kMaxDevice] = {0};
  std::memset(access_matrix, -1, sizeof(access_matrix));
  
  for (int i = 0; i < _ctxes.size(); i++) {
    int device = _ctxes[i].device_id;
    parts[device].insert(device);
    access_matrix[device][device] = device;
    access_cnt[device][device]++;
  }
  
  auto neighborParts = [&](IdType device, IdType part) -> std::vector<IdType> {
    std::vector<IdType> peer_vec;
    for (int peer = 0; peer < this->_ctxes.size(); peer++) {
      if (this->_topo_info.nvlink_matrix[device][peer]) {
        if (parts[peer].find(part) != parts[peer].end()) {
          peer_vec.push_back(peer);
        } 
      }
    }
    return peer_vec;
  };

  for (int device = 0; device < _ctxes.size(); device++) {
    std::vector<IdType> miss_parts;
    for (int part = 0; part < _ctxes.size(); part++) {
      if (access_matrix[device][part] != static_cast<IdType>(-1))
        continue;
      auto peers = neighborParts(device, part);
      {
        std::stringstream ss;
        for (auto peer : peers) ss << peer << " ";
        if (peers.empty()) ss << "NULL";
        LOG(DEBUG) << "[device, perr] is [" << device << ", " << part << "] choose peer from " << ss.str();
      }
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
      LOG(DEBUG) << "device " << device << " place part " << part << " into peer " << rep_peer;
      parts[rep_peer].insert(part);
      access_cnt[device][rep_peer]++;
      access_matrix[device][part] = rep_peer;
    }
  }
  std::vector<DistGraph::GroupConfig> configs;
  for (int i = 0; i < _ctxes.size(); i++) {
    auto ctx = _ctxes[i];
    IdType device = ctx.device_id;
    std::vector<IdType> part_ids(parts[device].begin(), parts[device].end());
    std::vector<Context> group;
    for (int j = 0; j < _ctxes.size(); j++) {
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
  for (IdType peer = 0; peer < _ctxes.size(); peer++) {
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
    return static_cast<IdType>(-1);
  }
  std::vector<std::tuple<IdType, IdType, double>> weight;
  for (auto peer : peers) {
    double bw = _topo_info.bandwitdh_matrix[device][peer];
    bw /= access_cnt[device][peer] + 1;
    weight.push_back({peer, parts[peer].size(), bw});
  }
  std::sort(weight.begin(), weight.end(), [&](auto x, auto y) {
    if (!exist && std::get<1>(x) != std::get<1>(y)) {
      return std::get<1>(x) < std::get<1>(y);
    } else {
      return std::get<2>(x) > std::get<2>(y);
    }
  });
  return std::get<0>(weight.front());
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
      topo_info.bandwitdh_matrix[device][peer] = 0;
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
          topo_info.bandwitdh_matrix[device][peer] = 2 * nbytes / sec / 1e9;
        } else {
          topo_info.bandwitdh_matrix[device][peer] = nbytes / sec / 1e9;
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
  for (int i = 0; i < universal_devices.size(); i++) {
    const auto &prop = universal_devices[i];
    ofs << "GPU [" << i << "] " << prop.name;
    ofs << " (UUID: ";
    for (int j = 0; j < 16; j++) {
      uint8_t x = prop.uuid.bytes[j];
      ofs << std::hex << static_cast<int>(x);
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
      ofs << std::setw(8) << std::fixed << std::setprecision(2) << topo_info.bandwitdh_matrix[i][j] << " ";
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
        // for (int j = 0; j < num_device; j++) {
        //   ss >> universal_topo_info.nvlink_matrix[i][j];
        // }
      }
    } else if (line.find(std::string{"Bandwidth Matrix"}) != std::string::npos) {
      for (int i = 0; i < num_device; i++) {
        std::getline(ifs, line);
        std::stringstream ss(line, std::ios::in);
        // for (int j = 0; j < num_device; j++) {
        //   ss >> universal_topo_info.bandwitdh_matrix[i][j];
        // }
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
      _topo_info.bandwitdh_matrix[i][j] = universal_topo_info.bandwitdh_matrix[device][peer];
    }
  }
}

std::ostream& operator<<(std::ostream &os, const DistGraph::GroupConfig &config) {
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
