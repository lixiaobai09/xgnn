#include "dist_graph.h"

#include <sys/mman.h>

#include <cstring>

#include "../device.h"

namespace samgraph {
namespace common {
namespace cuda {

std::shared_ptr<DistGraph> DistGraph::_inst = nullptr;

void DistGraph::_DatasetPartition(const Dataset *dataset, int sampler_id) {
  auto indptr_data = dataset->indptr->CPtr<IdType>();
  auto indices_data = dataset->indices->CPtr<IdType>();
  auto ctx = _ctxes[sampler_id];
  IdType num_node = dataset->num_node;
  IdType num_device = _ctxes.size();
  IdType part_edge_count = 0;
  for (IdType i = sampler_id; i < num_node; i += num_device) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    part_edge_count += num_edge;
  }
  _part_indptr.clear();
  _part_indptr.resize(num_device, nullptr);
  _part_indices.clear();
  _part_indices.resize(num_device, nullptr);

  IdType indptr_size = (num_node / num_device +
      (sampler_id < num_node % num_device? 1 : 0) + 1);
  _part_indptr[sampler_id] = Tensor::Empty(kI32, {indptr_size}, CPU(),
      "indptr in device:" + std::to_string(ctx.device_id));
  _part_indices[sampler_id] = Tensor::Empty(kI32, {part_edge_count}, CPU(),
      "indices in device:" + std::to_string(ctx.device_id));
  part_edge_count = 0;

  for (IdType i = sampler_id; i < num_node; i += num_device) {
    IdType num_edge = (indptr_data[i + 1] - indptr_data[i]);
    IdType part_id = sampler_id;
    IdType real_id = (i / num_device);
    _part_indptr[part_id]->Ptr<IdType>()[real_id] = part_edge_count;
    std::memcpy(
        &_part_indices[part_id]->Ptr<IdType>()[part_edge_count],
        &indices_data[indptr_data[i]],
        num_edge * sizeof(IdType));
    part_edge_count += num_edge;
  }
  IdType last_indptr = (num_node / num_device +
      (sampler_id < num_node % num_device? 1 : 0));
  _part_indptr[sampler_id]->Ptr<IdType>()[last_indptr] = part_edge_count;

  _part_indptr[sampler_id] = Tensor::CopyTo(_part_indptr[sampler_id], ctx,
      nullptr, Constant::kAllocNoScale);
  _part_indices[sampler_id] = Tensor::CopyTo(_part_indices[sampler_id], ctx,
      nullptr, Constant::kAllocNoScale);
}

void DistGraph::DatasetLoad(Dataset *dataset, int sampler_id,
    Context sampler_ctx) {
  _DatasetPartition(dataset, sampler_id);

  auto DataIpcShare = [&](std::vector<TensorPtr> &part_data,
      std::vector<size_t> part_size_vec,
      std::string name) {

    int num_worker = _ctxes.size();
    {
      // share self data to others
      CHECK(sampler_ctx == part_data[sampler_id]->Ctx());
      auto shared_data = part_data[sampler_id]->CPtr<IdType>();
      cudaIpcMemHandle_t &mem_handle =
        _shared_data->mem_handle[sampler_ctx.device_id];
      CUDA_CALL(cudaIpcGetMemHandle(&mem_handle, (void*)shared_data));
    }

    _Barrier();

    // receive data from others
    for (int i = 0; i < num_worker; ++i) {
      if (i == sampler_id) {
        continue;
      }
      CHECK(part_data[i] == nullptr);
      auto ctx = _ctxes[i];
      cudaIpcMemHandle_t &mem_handle = _shared_data->mem_handle[ctx.device_id];
      void *ptr;
      CUDA_CALL(cudaIpcOpenMemHandle(
            &ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
      part_data[i] = Tensor::FromBlob(ptr, kI32, {part_size_vec[i]}, ctx,
          name + " in device:" + std::to_string(ctx.device_id));
    }

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
