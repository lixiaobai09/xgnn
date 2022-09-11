#include "dist_graph.h"

#include <sys/mman.h>

#include <cstring>

#include "../device.h"

namespace samgraph {
namespace common {
namespace cuda {

std::shared_ptr<DistGraph> DistGraph::_inst = nullptr;

void DistGraph::_DatasetPartition(const Dataset *dataset) {
  auto indptr_data = dataset->indptr->CPtr<IdType>();
  auto indices_data = dataset->indices->CPtr<IdType>();
  IdType num_node = dataset->num_node;
  IdType num_device = _ctxes.size();
  std::vector<IdType> part_edge_count(num_device, 0);
  for (IdType i = 0; i < num_node; ++i) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    IdType part_id = (i % num_device);
    part_edge_count[part_id] += num_edge;
  }
  _part_indptr.clear();
  _part_indptr.resize(num_device, nullptr);
  _part_indices.clear();
  _part_indices.resize(num_device, nullptr);
  for (IdType i = 0; i < num_device; ++i) {
    IdType indptr_size = (num_node / num_device +
        (i < num_node % num_device? 1 : 0) + 1);
    _part_indptr[i] = Tensor::Empty(kI32, {indptr_size}, CPU(),
        "indptr in device:" + std::to_string(_ctxes[i].device_id));
    _part_indices[i] = Tensor::Empty(kI32, {part_edge_count[i]}, CPU(),
        "indices in device:" + std::to_string(_ctxes[i].device_id));
    part_edge_count[i] = 0;
  }
  for (IdType i = 0; i < num_node; ++i) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    IdType part_id = (i % num_device);
    IdType real_id = (i / num_device);
    _part_indptr[part_id]->Ptr<IdType>()[real_id] = part_edge_count[part_id];
    std::memcpy(
        &_part_indices[part_id]->Ptr<IdType>()[part_edge_count[part_id]],
        &indices_data[indptr_data[i]],
        num_edge * sizeof(IdType));
    // XXX: use omp to speedup?
    part_edge_count[part_id] += num_edge;
  }
  for (IdType i = 0; i < num_device; ++i) {
    IdType last_indptr = (num_node / num_device +
        (i < num_node % num_device? 1 : 0));
    _part_indptr[i]->Ptr<IdType>()[last_indptr] = part_edge_count[i];
  }

  for (IdType i = 0; i < num_device; ++i) {
    _part_indptr[i] = Tensor::CopyTo(_part_indptr[i], _ctxes[i],
        nullptr, Constant::kAllocNoScale);
    _part_indices[i] = Tensor::CopyTo(_part_indices[i], _ctxes[i],
        nullptr, Constant::kAllocNoScale);
  }
}

void DistGraph::DatasetLoad(Dataset *dataset, Context sampler_ctx) {
  if (sampler_ctx == _ctxes.front()) {
    _DatasetPartition(dataset);
  }

  // open P2P
  for (IdType i = 0; i < _ctxes.size(); ++i) {
    int device_id = _ctxes[i].device_id;
    CUDA_CALL(cudaSetDevice(device_id));
    for (IdType j = 0; j < _ctxes.size(); ++j) {
      if (j == i) {
        continue;
      }
      int peer_device_id = _ctxes[j].device_id;
      int flag;
      CUDA_CALL(cudaDeviceCanAccessPeer(&flag, device_id, peer_device_id));
      CHECK_EQ(flag, 1);
      CUDA_CALL(cudaDeviceEnablePeerAccess(peer_device_id, 0));
    }
  }
  Device::Get(sampler_ctx)->SetDevice(sampler_ctx);

  auto DataIpcShare = [&](std::vector<TensorPtr> &part_data,
      std::vector<size_t> part_size_vec,
      std::string name) {
    if (sampler_ctx == _ctxes.front()) {
      int num_worker = _ctxes.size();
      for (int i = 0; i < num_worker; ++i) {
        auto ctx = _ctxes[i];
        CHECK(ctx == part_data[i]->Ctx());
        auto shared_data = part_data[i]->CPtr<IdType>();
        auto gpu_device = Device::Get(ctx);
        gpu_device->SetDevice(ctx);
        cudaIpcMemHandle_t &mem_handle = _shared_data->mem_handle[ctx.device_id];
        CUDA_CALL(cudaIpcGetMemHandle(&mem_handle, (void*)shared_data));
      }

      _Barrier();
    } else {
      _Barrier();
      int num_worker = _ctxes.size();
      CHECK(part_data.size() == 0)
        << "_part_indptr need be null in other processes.";
      part_data.resize(num_worker, nullptr);
      for (int i = 0; i < num_worker; ++i) {
        auto ctx = _ctxes[i];
        cudaIpcMemHandle_t &mem_handle = _shared_data->mem_handle[ctx.device_id];
        void *ptr;
        CUDA_CALL(cudaIpcOpenMemHandle(
              &ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
        part_data[i] = Tensor::FromBlob(ptr, kI32, {part_size_vec[i]}, ctx,
            name + " in device:" + std::to_string(ctx.device_id));
      }
    }
    Device::Get(sampler_ctx)->SetDevice(sampler_ctx);
  };

  IdType num_node = dataset->num_node;
  IdType num_device = _ctxes.size();
  std::vector<size_t> part_size_vec(num_device);
  for (size_t i = 0; i < num_device; ++i) {
    part_size_vec[i] = (num_node / num_device +
        (i < num_node % num_device? 1 : 0) + 1);
  }
  DataIpcShare(_part_indptr, part_size_vec, "dataset part indptr");

  part_size_vec.clear();
  part_size_vec.resize(num_device, 0);
  auto indptr_data = dataset->indptr->CPtr<IdType>();
  for (IdType i = 0; i < num_node; ++i) {
    IdType num_edge = indptr_data[i + 1] - indptr_data[i];
    IdType part_id = (i % num_device);
    part_size_vec[part_id] += num_edge;
  }
  DataIpcShare(_part_indices, part_size_vec, "dataset part indices");

  CUDA_CALL(cudaMallocManaged(
        (void**)&_d_part_indptr, num_device * sizeof(IdType*)));
  CUDA_CALL(cudaMallocManaged(
        (void**)&_d_part_indices, num_device * sizeof(IdType*)));
  for (IdType i = 0; i < num_device; ++i) {
    _d_part_indptr[i] = _part_indptr[i]->Ptr<IdType>();
    _d_part_indices[i] = _part_indices[i]->Ptr<IdType>();
  }

  _num_node = dataset->num_node;
}

DeviceDistGraph DistGraph::DeviceHandle() const {
  return DeviceDistGraph(
      _d_part_indptr, _d_part_indices, _ctxes.size(), _num_node);
}

DistGraph::DistGraph(std::vector<Context> ctxes) {
  int num_worker = ctxes.size();
  _ctxes = ctxes;
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
  // XXX: release ipc memory with cudaIpcCloseMemHandle?
  CUDA_CALL(cudaFree((void*)dist_graph->_d_part_indptr));
  CUDA_CALL(cudaFree((void*)dist_graph->_d_part_indices));
  pthread_barrier_destroy(&dist_graph->_shared_data->barrier);
  munmap(dist_graph->_shared_data, sizeof(SharedData));
}

void DistGraph::Create(std::vector<Context> ctxes) {
  CHECK(_inst == nullptr);
  _inst = std::shared_ptr<DistGraph>(
      new DistGraph(ctxes), Release);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph