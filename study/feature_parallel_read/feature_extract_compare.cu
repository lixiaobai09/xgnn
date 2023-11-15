#include <iostream>
#include <cassert>
#include <chrono>
constexpr int kMax = 400000;
constexpr int kFeatDim = 256;
constexpr int kNTimes = 1;
using DataType = float;

#define CUDA_CALL(func)                             \
 {                                                  \
    cudaError_t err = func;                         \
    if(err != cudaSuccess) {                        \
        std::cerr << __FILE__ << ":" << __LINE__    \
             << " " << #func << " "                 \
             << cudaGetErrorString(err)             \
             << " errnum " << err;                  \
        exit(EXIT_FAILURE);                         \
    }                                               \
 }

template <typename T>
inline T RoundUpDiv(T target, T unit) {
  return (target + unit - 1) / unit;
}

class Timer {
 public:
  Timer(std::chrono::time_point<std::chrono::steady_clock> tp =
            std::chrono::steady_clock::now())
      : _start_time(tp) {}

  template <typename T>
  bool Timeout(double count) const {
    return Passed<T>() >= count;
  }

  double Passed() const { return Passed<std::chrono::duration<double>>(); }

  double PassedSec() const { return Passed<std::chrono::seconds>(); }

  double PassedMicro() const { return Passed<std::chrono::microseconds>(); }

  double PassedNano() const { return Passed<std::chrono::nanoseconds>(); }

  template <typename T>
  double Passed() const {
    return Passed<T>(std::chrono::steady_clock::now());
  }

  template <typename T>
  double Passed(std::chrono::time_point<std::chrono::steady_clock> tp) const {
    const auto elapsed = std::chrono::duration_cast<T>(tp - _start_time);
    return elapsed.count();
  }

  uint64_t TimePointMicro() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
                   _start_time.time_since_epoch()).count();
  }

  void Reset() { _start_time = std::chrono::steady_clock::now(); }

 private:
  std::chrono::time_point<std::chrono::steady_clock> _start_time;
};

namespace {

template<typename DataType>
__global__ void extract_cache_from_one(DataType *output, const size_t num_node,
    const size_t feat_dim, DataType *input) {
  size_t i = threadIdx.y + blockIdx.x * blockDim.y;
  if (i >= num_node) return;
  assert(i < num_node);
  size_t col = threadIdx.x;
  while (col < feat_dim) {
    output[i * feat_dim + col] = input[i * feat_dim + col];
    col += blockDim.x;
  }
}

template<typename DataType>
__global__ void extract_cache_from_two(DataType *output, const size_t num_node,
    const size_t feat_dim, DataType *input1, DataType *input2) {
  size_t i = threadIdx.y + blockIdx.x * blockDim.y;
  if (i >= num_node) return;
  size_t n_id = i / 2;
  assert(i < num_node);
  size_t col = threadIdx.x;
  while (col < feat_dim) {
    output[n_id * feat_dim + col] = (i%2?
        input1[n_id * feat_dim + col] : input2[n_id * feat_dim + col]);
    col += blockDim.x;
  }
}

}; // namespace

int main(int argc, char** argv) {
  cudaStream_t compute_stream;
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaStreamCreate(&compute_stream));
  {// open p2p
    int flag1, flag2;
    CUDA_CALL(cudaDeviceCanAccessPeer(&flag1, 0, 1));
    CUDA_CALL(cudaDeviceCanAccessPeer(&flag2, 0, 2));
    if ((!flag1) || (!flag2)) {
      std::cerr << "can not open p2p" << std::endl;
      exit(-1);
    }
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaDeviceEnablePeerAccess(1, 0));
    CUDA_CALL(cudaDeviceEnablePeerAccess(2, 0));
  }
  DataType *h_feature_ptr, *d1_feature_ptr, *d2_feature_ptr, *d_output;
  CUDA_CALL(cudaMallocHost((void**)&h_feature_ptr,
        kMax * kFeatDim * sizeof(DataType)));
  CUDA_CALL(cudaSetDevice(1));
  CUDA_CALL(cudaMalloc((void**)&d1_feature_ptr,
        kMax * kFeatDim * sizeof(DataType)));
  CUDA_CALL(cudaSetDevice(2));
  CUDA_CALL(cudaMalloc((void**)&d2_feature_ptr,
        kMax * kFeatDim * sizeof(DataType)));
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaMalloc((void**)&d_output,
        kMax * kFeatDim * sizeof(DataType)));

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) / 2 > kFeatDim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(kMax, static_cast<int>(block.y)));

  // warm up
  extract_cache_from_one<DataType><<<grid, block, 0, compute_stream>>>(
      d_output, kMax, kFeatDim, h_feature_ptr);
  CUDA_CALL(cudaStreamSynchronize(compute_stream));

  Timer t1;
  for (int i = 0; i < kNTimes; ++i) {
    extract_cache_from_one<DataType><<<grid, block, 0, compute_stream>>>(
        d_output, kMax, kFeatDim, h_feature_ptr);
    CUDA_CALL(cudaStreamSynchronize(compute_stream));
  }
  std::cout << "extracting through PCIe time cost: " << t1.Passed() << " sec."
    << std::endl;

  t1.Reset();
  for (int i = 0; i < kNTimes; ++i) {
    extract_cache_from_one<DataType><<<grid, block, 0, compute_stream>>>(
        d_output, kMax, kFeatDim, d1_feature_ptr);
    CUDA_CALL(cudaStreamSynchronize(compute_stream));
  }
  std::cout << "extracting through NVLink(G1) time cost: " << t1.Passed() << " sec."
    << std::endl;

  t1.Reset();
  for (int i = 0; i < kNTimes; ++i) {
    extract_cache_from_one<DataType><<<grid, block, 0, compute_stream>>>(
        d_output, kMax, kFeatDim, d2_feature_ptr);
    CUDA_CALL(cudaStreamSynchronize(compute_stream));
  }
  std::cout << "extracting through NVLink(G2) time cost: " << t1.Passed() << " sec."
    << std::endl;

  t1.Reset();
  for (int i = 0; i < kNTimes; ++i) {
    extract_cache_from_two<DataType><<<grid, block, 0, compute_stream>>>(
        d_output, kMax, kFeatDim, h_feature_ptr, d1_feature_ptr);
    CUDA_CALL(cudaStreamSynchronize(compute_stream));
  }
  std::cout << "extracting through PCIe/NVLink(G1) parallelly time cost: " << t1.Passed() << " sec."
    << std::endl;

  t1.Reset();
  for (int i = 0; i < kNTimes; ++i) {
    extract_cache_from_two<DataType><<<grid, block, 0, compute_stream>>>(
        d_output, kMax, kFeatDim, d1_feature_ptr, d2_feature_ptr);
    CUDA_CALL(cudaStreamSynchronize(compute_stream));
  }
  std::cout << "extracting through NVLink(G1/G2) parallelly time cost: " << t1.Passed() << " sec."
    << std::endl;

  // release data
  CUDA_CALL(cudaStreamDestroy(compute_stream));
  CUDA_CALL(cudaFreeHost(h_feature_ptr));
  CUDA_CALL(cudaFree(d1_feature_ptr));
  CUDA_CALL(cudaFree(d2_feature_ptr));
  { // close p2p
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaDeviceDisablePeerAccess(1));
    CUDA_CALL(cudaDeviceDisablePeerAccess(2));
  }
  return 0;
}
