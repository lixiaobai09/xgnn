#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>

#include "graph.h"
#include "timer.h"
#include <cub/cub.cuh>

constexpr int kNGPU = 2;
constexpr int NTIMES = 5;

// Functor type for selecting values with n % number == mod
struct ModEql
{
  int number;
  int mod;

  __host__ __device__ __forceinline__
  ModEql(int number, int mod) : number(number), mod(mod) {}

  // CUB_RUNTIME_FUNCTION __forceinline__
  // ModEql(int mod) : mod(mod) {}

  __host__ __device__ __forceinline__
  bool operator()(const int &n) const {
    return (n % number == mod);
  }
};

void LoadGraph(Graph* graph,
    std::string dataset_path = "/graph-learning/samgraph/papers100M") {
  std::unordered_map<std::string, size_t> meta;
  std::string meta_file_path = dataset_path + "/meta.txt";
  std::ifstream meta_file(meta_file_path);
  std::string line;
  while (std::getline(meta_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> kv{std::istream_iterator<std::string>{iss},
                                std::istream_iterator<std::string>{}};
    if (kv.size() < 2) {
      break;
    }

    meta[kv[0]] = std::stoul(kv[1]);
  }
  graph->num_node = static_cast<IdType>(meta["NUM_NODE"]);
  graph->num_edge = static_cast<IdType>(meta["NUM_EDGE"]);
  graph->num_train_set = static_cast<IdType>(meta["NUM_TRAIN_SET"]);

  // from mmap
  auto from_mmap = [](std::string filepath, IdType *len) -> void* {
    struct stat st;
    stat(filepath.c_str(), &st);
    size_t nbytes = st.st_size;

    int fd = open(filepath.c_str(), O_RDONLY, 0);
    void *data = mmap(NULL, nbytes, PROT_READ,
        MAP_SHARED | MAP_FILE | MAP_LOCKED, fd, 0);
    assert(data != (void *)-1);
    close(fd);

    *len = (nbytes / sizeof(IdType));
    return data;
  };

  IdType num_element;
  graph->indptr = static_cast<IdType*>(from_mmap(dataset_path + "/indptr.bin",
      &num_element));
  assert(num_element == (graph->num_node + 1));

  graph->indices = static_cast<IdType*>(from_mmap(dataset_path + "/indices.bin",
      &num_element));
  assert(num_element == (graph->num_edge));

  graph->train_set = static_cast<IdType*>(
      from_mmap(dataset_path + "/train_set.bin", &num_element));
  assert(num_element == (graph->num_train_set));
}

template<typename T>
T* ToGPU(T* data, IdType len) {
  T* ret_data;
  CUDA_CALL(cudaMalloc((void**)&ret_data, len * sizeof(T)));
  CUDA_CALL(cudaMemcpy(ret_data, data, len * sizeof(T), cudaMemcpyDefault));
  return ret_data;
}
template<typename T>
T* ToCPU(T* data, IdType len) {
  T* ret_data = static_cast<T*>(malloc(len * sizeof(T)));
  std::memcpy(ret_data, data, len * sizeof(T));
  return ret_data;
}

Graph CopyGraphToDevice(Graph* host_graph) {
  Graph ret_graph;
  ret_graph.indptr = ToGPU(host_graph->indptr, host_graph->num_node + 1);
  ret_graph.indices = ToGPU(host_graph->indices, host_graph->num_edge);
  ret_graph.train_set = ToGPU(host_graph->train_set, host_graph->num_train_set);
  ret_graph.num_node = host_graph->num_node;
  ret_graph.num_edge = host_graph->num_edge;
  ret_graph.num_train_set = host_graph->num_train_set;
  return ret_graph;
}

namespace {

__global__ void init_random_states(curandState *states, size_t num,
                                   unsigned long seed) {
  size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadId < num) {
    /** Using different seed & constant sequence 0 can reduce memory 
      * consumption by 800M
      * https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
      */
    curand_init(seed+threadId, 0, 0, &states[threadId]);
  }
}

template <size_t WARP_SIZE, size_t BLOCK_WARP, size_t TILE_SIZE,
          typename GraphType>
__global__ void sample_khop0(GraphType graph,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  assert(WARP_SIZE == blockDim.x);
  assert(BLOCK_WARP == blockDim.y);
  size_t index = TILE_SIZE * blockIdx.x + threadIdx.y;
  const size_t last_index = min(TILE_SIZE * (blockIdx.x + 1), num_input);

  size_t i =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
  // i is out of bound in num_random_states, so use a new curand
  curandState local_state;
  curand_init(i, 0, 0, &local_state);

  while (index < last_index) {
    const IdType rid = input[index];
    const IdType *edges = graph[rid];
    const IdType len = graph.NumEdge(rid);
    // const IdType off = indptr[rid];
    // const IdType len = indptr[rid + 1] - indptr[rid];

    if (len <= fanout) {
      size_t j = threadIdx.x;
      for (; j < len; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = edges[j];
      }
      __syncwarp();
      for (; j < fanout; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = kEmptyKey;
        tmp_dst[index * fanout + j] = kEmptyKey;
      }
    } else {
      size_t j = threadIdx.x;
      for (; j < fanout; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = edges[j];
      }
      __syncwarp();
      for (; j < len; j += WARP_SIZE) {
        size_t k = curand(&local_state) % (j + 1);
        if (k < fanout) {
          atomicExch(tmp_dst + index * fanout + k, edges[j]);
        }
      }
    }
    index += BLOCK_WARP;
  }
}

constexpr int HASH_SHIFT = 7;
constexpr int HASH_MASK  = ((1 << HASH_SHIFT) - 1);
constexpr int HASHTABLE_SIZE = (1 << HASH_SHIFT);
constexpr IdType HASH_EMPTY = 0xffffffff;

struct DeviceSet {
  IdType hashtable[HASHTABLE_SIZE];
  IdType count;
};

__forceinline__ __device__ void _SetInsert(DeviceSet *d_set,
    IdType *items, const IdType key, const IdType num_limit) {
  IdType pos = (key & HASH_MASK);
  IdType offset = 1;
  while(d_set->count < num_limit) {
    IdType old_key = atomicCAS( &(d_set->hashtable[pos]), HASH_EMPTY, key);
    if (old_key == HASH_EMPTY) {
      IdType old_count = atomicAdd(&(d_set->count), 1);
      if (old_count >= num_limit) {
        atomicSub(&(d_set->count), 1);
        d_set->hashtable[pos] = HASH_EMPTY;
      } else {
        items[old_count] = pos;
      }
      break;
    }
    if (old_key == key) {
      break;
    }
    pos = ((pos + offset) & HASH_MASK);
    assert(offset < HASHTABLE_SIZE);
    ++offset;
  }
}

template <size_t GROUP_SIZE, size_t BLOCK_WARP,size_t TILE_SIZE,
         typename GraphType>
__global__ void sample_khop3(GraphType graph,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  assert(GROUP_SIZE == blockDim.x);
  assert(BLOCK_WARP == blockDim.y);
  assert(fanout < HASHTABLE_SIZE);
  IdType index = TILE_SIZE * blockIdx.x + threadIdx.y;
  const IdType last_index = TILE_SIZE * (blockIdx.x + 1);
  IdType i =  blockIdx.x * blockDim.y + threadIdx.y;
  assert(i < num_random_states);

  __shared__ DeviceSet shared_set[BLOCK_WARP];
  // use shared memory to let warp threads share the same random_state
  __shared__ curandState shared_state[BLOCK_WARP];
  shared_state[threadIdx.y] = random_states[i];

  auto &d_set = shared_set[threadIdx.y];
  curandState &local_state = shared_state[threadIdx.y];

  for (int idx = threadIdx.x; idx < HASHTABLE_SIZE; idx += GROUP_SIZE) {
    d_set.hashtable[idx] = HASH_EMPTY;
  }
  if (threadIdx.x == 0) {
    d_set.count = 0;
  }

  for (; index < last_index; index += BLOCK_WARP) {
    if (index >= num_input) {
      continue;
    }
    const IdType rid = input[index];
    const IdType *edges = graph[rid];
    const IdType len = graph.NumEdge(rid);

    if (len <= fanout) {
      IdType j = threadIdx.x;
      for (; j < len; j += GROUP_SIZE) {
        tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = edges[j];
      }

      for (; j < fanout; j += GROUP_SIZE) {
        tmp_src[index * fanout + j] = kEmptyKey;
        tmp_dst[index * fanout + j] = kEmptyKey;
      }
    } else {
      IdType *mark_pos = (tmp_src + index * fanout);
      while (d_set.count < fanout) {
        IdType rand = (curand(&local_state) % len);
        _SetInsert(&d_set, mark_pos, rand, fanout);
      }
      __syncwarp();
      assert(d_set.count == fanout);
      for (IdType j = threadIdx.x; j < fanout; j += GROUP_SIZE) {
        IdType val = d_set.hashtable[mark_pos[j]];
        // reset the hashtable values
        d_set.hashtable[mark_pos[j]] = HASH_EMPTY;
        tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = edges[val];
      }
      if (threadIdx.x == 0) {
        d_set.count = 0;
      }
    }
  }
  random_states[i] = local_state;
}

}; // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <batch_size> <fanout>" << std::endl;
    exit(-1);
  }
  { // open p2p
    int flag1, flag2;
    CUDA_CALL(cudaDeviceCanAccessPeer(&flag1, 0, 1));
    CUDA_CALL(cudaDeviceCanAccessPeer(&flag2, 1, 0));
    if ((!flag1) || (!flag2)) {
      std::cerr << "can not open p2p" << std::endl;
      exit(-1);
    }
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaDeviceEnablePeerAccess(1, 0));
    CUDA_CALL(cudaSetDevice(1));
    CUDA_CALL(cudaDeviceEnablePeerAccess(0, 0));
  }

  Graph host_global_graph;
  LoadGraph(&host_global_graph);
  PartitionGraph part_graph(&host_global_graph, kNGPU);

  CUDA_CALL(cudaSetDevice(0));
  IdType batch_size = std::stoi(argv[1]);
  IdType *host_trainset =
    ToCPU(host_global_graph.train_set, host_global_graph.num_train_set);
  std::random_device rd;
  std::mt19937 g(rd());
  IdType fanout = std::stoi(argv[2]);

  IdType *tmp_src, *tmp_dst;
  CUDA_CALL(cudaMalloc((void**)&tmp_src, batch_size * fanout * sizeof(IdType)));
  CUDA_CALL(cudaMalloc((void**)&tmp_dst, batch_size * fanout * sizeof(IdType)));

  curandState *curand_states;
  CUDA_CALL(cudaMalloc((void**)&curand_states,
        batch_size * sizeof(curandState)));
  long seed = time(NULL);
  { // init curand
    const int BLOCK_SIZE = 128;
    const dim3 block_t(BLOCK_SIZE);
    const dim3 grid_t((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    init_random_states<<<grid_t, block_t>>>(curand_states, batch_size, seed);
  }

  cudaStream_t compute_stream;
  CUDA_CALL(cudaStreamCreate(&compute_stream));
  for (int i = 0; i < NTIMES; i++){ // call sampling kernel function
    std::shuffle(host_trainset,
        host_trainset + host_global_graph.num_train_set, g);
    IdType *input = ToGPU(host_trainset, host_global_graph.num_train_set);
    const int GROUP_SIZE = 16;
    const int BLOCK_WARP = 128 / GROUP_SIZE;
    const int TILE_SIZE = BLOCK_WARP * 16;
    const dim3 block_t(GROUP_SIZE, BLOCK_WARP);
    const dim3 grid_t((batch_size + TILE_SIZE - 1) / TILE_SIZE);
    // wait to stop others computing
    CUDA_CALL(cudaDeviceSynchronize());
    Timer t;
    for (IdType offset = 0; offset < host_global_graph.num_train_set;
        offset += batch_size) {
      IdType num_input = std::min(batch_size,
          host_global_graph.num_train_set - offset);
      sample_khop3<GROUP_SIZE, BLOCK_WARP, TILE_SIZE, DeviceDistGraph>
        <<<grid_t, block_t, 0, compute_stream>>> (
            part_graph.DeviceHandle(), input + offset, num_input, fanout,
            tmp_src, tmp_dst,
            curand_states, batch_size);
      CUDA_CALL(cudaStreamSynchronize(compute_stream));
    }
    std::cout << "epoch [" << i << "] sampling time: "
      << t.Passed() << std::endl;
    CUDA_CALL(cudaFree(input));
  }
  CUDA_CALL(cudaFree(curand_states));
  // release data
  CUDA_CALL(cudaStreamDestroy(compute_stream));
  { // close p2p
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaDeviceDisablePeerAccess(1));
    CUDA_CALL(cudaSetDevice(1));
    CUDA_CALL(cudaDeviceDisablePeerAccess(0));
  }

  { // for RPC test
    // init gpu0 data
    CUDA_CALL(cudaSetDevice(0));
    int *tmp_num_input;
    CUDA_CALL(cudaHostAlloc((void**)&tmp_num_input,
          sizeof(int), cudaHostAllocDefault));
    // init each gpu data
    curandState *rand_list[kNGPU];
    cudaStream_t stream[kNGPU];
    IdType *tmp_src_list[kNGPU];
    IdType *tmp_dst_list[kNGPU];
    IdType *tmp_input[kNGPU];
    for (int dev_id = 0; dev_id < kNGPU; ++dev_id) {
      CUDA_CALL(cudaSetDevice(dev_id));
      CUDA_CALL(cudaMalloc((void**)&rand_list[dev_id],
            batch_size * sizeof(curandState)));
      CUDA_CALL(cudaMalloc((void**)&tmp_src_list[dev_id],
            batch_size * fanout * sizeof(IdType)));
      CUDA_CALL(cudaMalloc((void**)&tmp_dst_list[dev_id],
            batch_size * fanout * sizeof(IdType)));
      CUDA_CALL(cudaMalloc((void**)&tmp_input[dev_id],
            batch_size * sizeof(IdType)));
      CUDA_CALL(cudaStreamCreate(&stream[dev_id]));
      const int BLOCK_SIZE = 128;
      const dim3 block_t(BLOCK_SIZE);
      const dim3 grid_t((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
      init_random_states<<<grid_t, block_t, 0, stream[dev_id]>>>(
          rand_list[dev_id], batch_size, seed);
    }
    // rpc test
    for (int i = 0; i < NTIMES; i++){ // call sampling kernel function
      // shuffle the train_set
      CUDA_CALL(cudaSetDevice(0));
      std::shuffle(host_trainset,
          host_trainset + host_global_graph.num_train_set, g);
      IdType *input = ToGPU(host_trainset, host_global_graph.num_train_set);
      // used for temp buffer data
      IdType *tmp_buffer_dev0;
      CUDA_CALL(cudaMalloc((void**)&tmp_buffer_dev0,
            batch_size * sizeof(IdType)));
      // for cub select
      // Determine temporary device storage requirements
      void     *d_temp_storage = NULL;
      size_t   temp_storage_bytes = 0;
      cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
          input, tmp_buffer_dev0, tmp_num_input,
          batch_size, ModEql(kNGPU, 0));
      // Allocate temporary storage
      CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

      // call rpc
      const int GROUP_SIZE = 16;
      const int BLOCK_WARP = 128 / GROUP_SIZE;
      const int TILE_SIZE = BLOCK_WARP * 16;
      const dim3 block_t(GROUP_SIZE, BLOCK_WARP);
      // wait to stop others computing
      for (int dev_id = 0; dev_id < kNGPU; ++dev_id) {
        CUDA_CALL(cudaSetDevice(dev_id));
        CUDA_CALL(cudaDeviceSynchronize());
      }
      // count the run time
      double total_time = 0;
      double select_time = 0, sample_time = 0, transform_time = 0;
      Timer t;
      for (IdType offset = 0; offset < host_global_graph.num_train_set;
          offset += batch_size) {
        IdType num_input = std::min(batch_size,
            host_global_graph.num_train_set - offset);
        for (int dev_id = 0; dev_id < kNGPU; ++dev_id) {
          Timer t1;
          { // pick nodes
            ModEql select_op(kNGPU, dev_id);
            CUDA_CALL(cudaSetDevice(0));
            // Run selection
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                input, tmp_buffer_dev0, tmp_num_input,
                // num_input, select_op, stream[0]);
                num_input, select_op);
            CUDA_CALL(cudaStreamSynchronize(0));
          }
          // std::cout << "for dev_id = " << dev_id << ", jktmp_num_input = "
          //   << *tmp_num_input << std::endl;
          CUDA_CALL(cudaSetDevice(dev_id));
          const dim3 grid_t((*tmp_num_input + TILE_SIZE - 1) / TILE_SIZE);
          CUDA_CALL(cudaMemcpyAsync(tmp_input[dev_id], tmp_buffer_dev0,
                *tmp_num_input * sizeof(IdType), cudaMemcpyDefault,
                stream[dev_id]));
          CUDA_CALL(cudaStreamSynchronize(stream[dev_id]));
          select_time += t1.Passed();
          Timer t2;
          sample_khop3<GROUP_SIZE, BLOCK_WARP, TILE_SIZE, DeviceNormalGraph>
            <<<grid_t, block_t, 0, stream[dev_id]>>> (
                part_graph.NormalGraph(dev_id),
                tmp_input[dev_id], *tmp_num_input, fanout,
                tmp_src_list[dev_id], tmp_dst_list[dev_id],
                rand_list[dev_id], batch_size);
          CUDA_CALL(cudaStreamSynchronize(stream[dev_id]));
          sample_time += t2.Passed();
          Timer t3;
          if (dev_id != 0) { // gpu 0 don not need to transform result
            CUDA_CALL(cudaMemcpy(tmp_src, tmp_src_list[dev_id],
                  *tmp_num_input * sizeof(IdType), cudaMemcpyDefault));
            CUDA_CALL(cudaMemcpy(tmp_dst, tmp_dst_list[dev_id],
                  *tmp_num_input * sizeof(IdType), cudaMemcpyDefault));
          }
          transform_time += t3.Passed();
        }
      }
      total_time = t.Passed();
      std::cout << "epoch [" << i << "] sampling time: "
        << total_time
        << ", select_time: " << select_time
        << ", sample_time: " << sample_time
        << ", transform_time: " << transform_time << std::endl;
      CUDA_CALL(cudaFree(d_temp_storage));
      CUDA_CALL(cudaFree(input));
      CUDA_CALL(cudaFree(tmp_buffer_dev0));
    }
    // release temp data
    for (int dev_id = 0; dev_id < kNGPU; ++dev_id) {
      CUDA_CALL(cudaSetDevice(dev_id));
      CUDA_CALL(cudaFree(rand_list[dev_id]));
      CUDA_CALL(cudaFree(tmp_src_list[dev_id]));
      CUDA_CALL(cudaFree(tmp_dst_list[dev_id]));
      CUDA_CALL(cudaFree(tmp_input[dev_id]));
      CUDA_CALL(cudaStreamDestroy(stream[dev_id]));
    }
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaFreeHost(tmp_num_input));
  }

  // Graph device_graph = CopyGraphToDevice(&host_global_graph);

  free(host_trainset);
  CUDA_CALL(cudaFree(tmp_src));
  CUDA_CALL(cudaFree(tmp_dst));
  return 0;
}
