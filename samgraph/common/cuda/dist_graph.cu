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


namespace samgraph {
namespace common {
namespace cuda {

namespace {

template<typename T>
std::vector<T> operator- (const std::set<T> &a, const std::set<T> &b) {
  std::vector<T> ret;
  for (auto i : a) {
    if (!b.count(i)) {
      ret.emplace_back(i);
    }
  }
  return std::move(ret);
};

}; // namespace

DeviceP2PComm *DeviceP2PComm::_p2p_comm = nullptr;

DeviceP2PComm::DeviceP2PComm(int num_worker) 
  : _init(false), _dev(Constant::kEmptyKey), _comm_size(num_worker),
    _rank(Constant::kEmptyKey) {
  std::memset(_peers, 0xff, sizeof(_peers));
  _shared_data = static_cast<SharedData *>(mmap(
    NULL, sizeof(SharedData), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  pthread_barrierattr_t attr;
  pthread_barrierattr_init(&attr);
  pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_barrier_init(&_shared_data->barrier, &attr, _comm_size);
}

DeviceP2PComm::~DeviceP2PComm() {
  munmap(_shared_data, sizeof(SharedData));
  _init = false;
}

void DeviceP2PComm::Init(int num_worker) {
  if (_p2p_comm == nullptr)
    _p2p_comm = new DeviceP2PComm(num_worker);
}

void DeviceP2PComm::Create(int worker_id, int device_id) {
  CHECK_EQ(worker_id, device_id);
  CHECK_NE(_p2p_comm, nullptr);
  // get p2p access matrix
  CUDA_CALL(cudaSetDevice(device_id));
  for (IdType i = 0; i < _p2p_comm->_comm_size; i++) {
      int& flag = _p2p_comm->_shared_data->p2p_matrix[device_id][i];
    if (i != device_id) {
      CUDA_CALL(cudaDeviceCanAccessPeer(&flag, device_id, i));
    } else {
      flag = 1;
    }
  }
  _p2p_comm->Barrier();
  // find p2p clique
  int my_clique;
  auto cliques = _p2p_comm->SplitClique(worker_id, my_clique);
  auto& clique = cliques[my_clique];
  if (cliques.size() > 1) {
    // need to split p2p communication
    std::stringstream ss;
    int new_rank = -1;
    int new_comm_size = clique.count();
    ss << "p2pComm";
    for (int i = 0, rk = 0; i < _p2p_comm->_comm_size; i++) {
      if (clique[i]) {
        if (i == device_id) new_rank = rk;
        ss << "-" << i;
        rk++;
      }
    }
    SharedData *shared_data = nullptr;
    if (new_rank == 0) {
      shm_unlink(ss.str().c_str());
      auto shm_fd = shm_open(ss.str().c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
      CHECK_NE(shm_fd, -1);
      int err = ftruncate(shm_fd, sizeof(SharedData));
      CHECK_NE(err, -1);
      shared_data = static_cast<SharedData *>(mmap(
        NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
      CHECK_NE(shared_data, MAP_FAILED);
      pthread_barrierattr_t attr;
      pthread_barrierattr_init(&attr);
      pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
      pthread_barrier_init(&shared_data->barrier, &attr, new_comm_size);
      _p2p_comm->Barrier();
    } else {
      _p2p_comm->Barrier();
      auto shm_fd = shm_open(ss.str().c_str(), O_RDWR, S_IRUSR | S_IWUSR);
      CHECK_NE(shm_fd, -1);
      shared_data = static_cast<SharedData *>(mmap(
        NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
      CHECK_NE(shared_data, MAP_FAILED);
    }
    munmap(_p2p_comm->_shared_data, sizeof(SharedData));
    _p2p_comm->_shared_data = shared_data;
    _p2p_comm->_comm_size = new_comm_size;
    _p2p_comm->_rank = new_rank;
  } else { 
    // each pair gpu can p2p access
    _p2p_comm->_rank = worker_id;
  }
  _p2p_comm->_dev = device_id;
  for (int i = 0, j = 0; i < clique.size(); i++) {
    if (clique[i])
      _p2p_comm->_peers[j++] = i;
  }
  _p2p_comm->Barrier();
  _p2p_comm->_init = true;
}

std::vector<std::bitset<kMaxDevice>> DeviceP2PComm::SplitClique(int device_id, int &my_clique) {
  std::bitset<kMaxDevice> gpu_cnt = 0;
  std::vector<std::bitset<kMaxDevice>> cliques;
  // using gpu_peer_info_t = std::pair<std::bitset<kMaxDevice>, IdType>;
  for (IdType i = 0; i < _p2p_comm->_comm_size && gpu_cnt.count() < _p2p_comm->_comm_size; i++) {
    if (gpu_cnt[i]) continue;    
    std::bitset<kMaxDevice> gpu1, cur, none, max_clique;
    for (IdType j = 0; j < _p2p_comm->_comm_size; j++) {
      if (_p2p_comm->_shared_data->p2p_matrix[i][j])
        gpu1[j] = 1;
    }
    cur[i] = 1;
    gpu1 ^= cur;
    gpu1 = gpu1 & (~gpu_cnt);
    FindClique(cur, gpu1, none, max_clique);
    cliques.push_back(max_clique);
    gpu_cnt |= max_clique;
    if (max_clique[device_id])
      my_clique = cliques.size() - 1;
  }
  // std::stringstream ss;
  // for (auto clique : cliques) {
  //   for (int i = 0; i < kMaxDevice; i++) {
  //     if (clique[i]) ss << i << " ";
  //   }
  //   ss << "| ";
  // }
  // LOG(INFO) << "Cliques: " << ss.str();
  return cliques;
}

void DeviceP2PComm::FindClique(std::bitset<kMaxDevice> clique, 
                               std::bitset<kMaxDevice> neighbor, 
                               std::bitset<kMaxDevice> none,
                               std::bitset<kMaxDevice> &max_clique) {
  if (!neighbor.count() && !none.count()) {
    if (clique.count() > max_clique.count())
      max_clique = clique;
  }
  if (!neighbor.count()) return;
  auto get_neighbor = [&](IdType dev) {
    std::bitset<kMaxDevice> nb;
    for (IdType i = 0; i < _p2p_comm->_comm_size; i++) {
      if (_p2p_comm->_shared_data->p2p_matrix[dev][i])
        nb[i] = 1;
    }
    nb[dev] = 0;
    return nb;
  };
  IdType pivot = neighbor._Find_first();
  std::bitset<kMaxDevice> pivot_nb = get_neighbor(pivot);
  auto next_nb = neighbor & (~pivot_nb);
  for (IdType i = 0; i < _p2p_comm->_comm_size; i++) {
    if (next_nb[i]) {
      auto nb = get_neighbor(i);
      auto clique_ = clique;
      clique_[i] = 1;
      FindClique(clique_, neighbor & nb, none & nb, max_clique);
      neighbor[i] = 0;
      none[i] = 1;
    }
  }
}

DistArray::DistArray(void *devptr, DeviceP2PComm *comm, StreamHandle stream) 
  : _comm(comm)
{
  auto cu_stream = static_cast<cudaStream_t>(stream);
  CUDA_CALL(cudaSetDevice(_comm->DevId()));
  CUDA_CALL(cudaMallocHost((void**)&_devptrs_h, sizeof(void*) * CommSize()));
  CUDA_CALL(cudaMalloc((void**)&_devptrs_d, sizeof(void*) * CommSize()));
  CUDA_CALL(cudaIpcGetMemHandle(_comm->IpcMemHandle(Rank()), devptr));
  _comm->Barrier();
  for (size_t i = 0; i < CommSize(); i++) {
    if (i != Rank()) {
      CUDA_CALL(cudaIpcOpenMemHandle(&_devptrs_h[i], *_comm->IpcMemHandle(i), cudaIpcMemLazyEnablePeerAccess));
    } else {
      _devptrs_h[i] = devptr;
    }
  }
  _comm->Barrier();
  CUDA_CALL(cudaMemcpyAsync(_devptrs_d, _devptrs_h, sizeof(void*) * CommSize(), cudaMemcpyDefault, cu_stream));
  CUDA_CALL(cudaStreamSynchronize(cu_stream));
}

DistArray::~DistArray() {
  for (size_t i = 0; i < CommSize(); i++) {
    if (i != Rank()) {
      CUDA_CALL(cudaIpcCloseMemHandle(_devptrs_h[i]));
    }
  }
  CUDA_CALL(cudaFree(_devptrs_d));
  CUDA_CALL(cudaFreeHost(_devptrs_h));
  _devptrs_d = nullptr;
  _devptrs_h = nullptr;
}

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

  std::stringstream ctx_name;
  ctx_name << ctx;

  IdType indptr_size = (num_node / num_part +
      (part_id < num_node % num_part? 1 : 0) + 1);
  _part_indptr[part_id] = Tensor::Empty(kI32, {indptr_size}, CPU(),
      "indptr in device:cpu" );
  _part_indices[part_id] = Tensor::Empty(kI32, {part_edge_count}, CPU(),
      "indices in device:cpu");
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

  if (ctx.device_type != kCPU) {
    LOG(DEBUG) << "Load Graph to GPU: " << ctx << " store " 
               << ToReadableSize(_part_indices[part_id]->NumBytes() + _part_indptr[part_id]->NumBytes());
    _part_indptr[part_id] = Tensor::CopyTo(_part_indptr[part_id], ctx,
        nullptr, ctx_name.str(), Constant::kAllocNoScale);
    _part_indices[part_id] = Tensor::CopyTo(_part_indices[part_id], ctx,
        nullptr, ctx_name.str(), Constant::kAllocNoScale);
  }
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

  if (RunConfig::dist_graph_part_cpu < 1) {
    for (IdType part_id : part_ids) {
      _DatasetPartition(dataset, sampler_ctx, part_id, num_part);
    }
  } else {
    auto ctx_group = _group_configs[sampler_id].ctx_group;
    for (int i = 0; i < part_ids.size(); i++) {
      _DatasetPartition(dataset, ctx_group[i], part_ids[i], num_part);
    }
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

  if (RunConfig::dist_graph_part_cpu < 1) {
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
  }

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
  std::vector<std::vector<int>> access_part_ctx(
      num_ctx, std::vector<int>(num_ctx, -1));
  std::vector<std::set<int>> store_parts(num_ctx);

  std::vector<std::set<int>> can_access_parts(num_ctx);
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
    asc_degree_gpu_order[i] = std::make_tuple(
        i, static_cast<int>(neighbor_adjacency[i].size()));
  }
  // sort nodes by ascending order to iterate
  std::sort(asc_degree_gpu_order.begin(), asc_degree_gpu_order.end(),
      [](auto x, auto y) {
        if (std::get<1>(x) != std::get<1>(y)) {
          return std::get<1>(x) < std::get<1>(y);
        }
        return std::get<0>(x) < std::get<0>(y);
      });
  std::stringstream ss;
  for (auto item : asc_degree_gpu_order) {
    ss << std::get<0>(item) << " ";
  }
  LOG(INFO) << "new node order to iterate: " << ss.str();
  // iterator for each GPU ctx
  for (auto item : asc_degree_gpu_order) {
    int i = std::get<0>(item);
    // get can not access parts for GPU i
    std::vector<int> can_not_access_parts =
      (parts_universal_set - can_access_parts[i]);
    // sort it by default degree
    std::sort(can_not_access_parts.begin(), can_not_access_parts.end(),
        [&neighbor_adjacency](auto x, auto y) {
          if (neighbor_adjacency[x].size() != neighbor_adjacency[y].size()) {
            return neighbor_adjacency[x].size() < neighbor_adjacency[y].size();
          }
          return x < y;
        });
    for (auto need_part : can_not_access_parts) {
      // id, stored_parts_size, need_score, if_same_part_in_neighbors, bandwidth
      std::vector<std::tuple<int, int, int, int, double>> tmp_vec;
      // iterate GPU_i neighbors
      for(auto j : neighbor_adjacency[i]) {
        int need_score = 0;
        for (auto k : neighbor_adjacency[j]) {
          if(!can_access_parts[k].count(need_part)) {
            ++need_score;
          }
        }
        tmp_vec.emplace_back(j, store_parts[j].size(), need_score,
            can_access_parts[j].count(need_part),
            bandwidth_matrix[i][j] / (access_count[i][j] + 1));
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
            // bandwidth
            if (std::get<4>(x) != std::get<4>(y)) {
              return std::get<4>(x) > std::get<4>(y);
            }
            return std::get<0>(x) < std::get<0>(y);
          });
      int choose_gpu_id = std::get<0>(tmp_vec.front());
      store_parts[choose_gpu_id].insert(need_part);
      // update can access parts for choose_gpu_id neighbors
      for (auto neighbor : neighbor_adjacency[choose_gpu_id]) {
        can_access_parts[neighbor].insert(need_part);
      }
    }
    // choose part in which GPU to access
    assert(can_access_parts[i].size() == num_ctx);
    for (int j = 0; j < num_ctx; ++j) {
      int which_gpu;
      double max_bandwidth = 0.0;
      for(auto neighbor : neighbor_adjacency[i]) {
        if (store_parts[neighbor].count(j)) {
          double tmp_bandwidth =
            bandwidth_matrix[i][neighbor] / (access_count[i][neighbor] + 1);
          if (tmp_bandwidth > max_bandwidth) {
            max_bandwidth = tmp_bandwidth;
            which_gpu = neighbor;
          }
        }
      }
      access_part_ctx[i][j] = which_gpu;
      access_count[i][which_gpu] += 1;
    }
  }

  std::vector<DistGraph::GroupConfig> configs;
  for (int i = 0; i < num_ctx; i++) {
    auto ctx = _ctxes[i];
    IdType device = ctx.device_id;
    CHECK_EQ(i, device);
    std::vector<IdType> part_ids(store_parts[device].begin(),
        store_parts[device].end());
    std::vector<Context> ctx_group(num_ctx);
    for (int j = 0; j < num_ctx; ++j) {
      ctx_group[j] = GPU(access_part_ctx[device][j]);
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
