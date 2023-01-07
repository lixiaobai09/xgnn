#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <cassert>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <numeric>

using IdType = unsigned int;

constexpr int kNumSampler  = 4;

struct StepData {
  IdType num_samples;
  std::vector<IdType> samples;
};
using SamplerSteps = std::vector<StepData>;

void* DataFromFile(const std::string &path, size_t &nbytes) {
  struct stat st;
  stat(path.c_str(), &st);
  nbytes = st.st_size;
  int fd = open(path.c_str(), O_RDONLY, 0);
  void *ret_data = mmap(NULL, nbytes, PROT_READ,
      MAP_SHARED | MAP_FILE | MAP_LOCKED, fd, 0);
  assert(ret_data != reinterpret_cast<void*>(-1));
  close(fd);
  return ret_data;
}

template <typename T>
T GetMax(const std::vector<T> &vec) {
  T ret = 0;
  for (const auto &item: vec) {
    if (item > ret) {
      ret = item;
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " <sample results directory>"
      << " <cache ordered nodes>"
      << " <cache percentage>"
      << " <feature dim>" << std::endl;
    exit(-1);
  }
  // samples result files directory
  std::string dir_path = argv[1];
  // cache nodes file, ordered by cache degree
  std::string cache_ordered_nodes_file = argv[2];
  // cache percentage
  double cache_percentage = std::stod(argv[3]);
  // feature dimension
  size_t feat_dim = std::stoul(argv[4]);
  feat_dim *= 4;

  SamplerSteps sampler_data[kNumSampler];

  for (int sampler_id = 0; sampler_id < kNumSampler; ++sampler_id) {
    std::string sampler_file = (dir_path + "/sample_output_nodes_" +
        std::to_string(sampler_id) + ".bin");
    size_t nbytes;
    auto data = static_cast<IdType*>(DataFromFile(sampler_file, nbytes));

    IdType num_steps = 0;
    IdType offset = 0;
    IdType num_element = (nbytes / sizeof(IdType));
    while(offset < num_element) {
      offset += (data[offset] + 1);
      num_steps += 1;
    }
    assert(offset == num_element);

    SamplerSteps sampler_steps(num_steps);
    offset = 0;
    IdType step_id = 0;
    while(offset < num_element) {
      IdType num_samples = data[offset];
      IdType *data_start = (data + offset + 1);
      std::vector<IdType> samples(data_start, data_start + num_samples);
      StepData step_data{num_samples, std::move(samples)};
      sampler_steps[step_id] = std::move(step_data);

      offset += (data[offset] + 1);
      step_id += 1;
    }
    sampler_data[sampler_id] = std::move(sampler_steps);


    munmap(static_cast<void*>(data), nbytes);
  }

  for (int i = 0; i < kNumSampler; ++i) {
    const SamplerSteps &sampler_steps = sampler_data[i];
    for (int j = 0; j < sampler_steps.size(); ++j) {
      IdType num_samples = sampler_steps[j].num_samples;
      /*
      std::cout << "sample_id, step_id, num_samples: "
        << i << ", "
        << j << ", "
        << num_samples << std::endl;
      */
    }
  }

  // get cache nodes
  size_t nbytes;
  auto data = static_cast<IdType*>(DataFromFile(cache_ordered_nodes_file, nbytes));
  IdType num_nodes = (nbytes / sizeof(IdType));
  std::vector<IdType> cache_ordered_nodes(data, data + num_nodes);
  munmap(static_cast<void*>(data), nbytes);

  double local_speed = static_cast<double>(700.0)*1024*1024*1024;
  double nvlink_speed = static_cast<double>(25.0)*1024*1024*1024;
  double pcie_speed = static_cast<double>(10.0)*1024*1024*1024;
  // set local_cache_map
  std::vector<int> local_cache_map(num_nodes, 2); // 2 means in CPU memory
  IdType segment_size = static_cast<IdType>(cache_percentage * num_nodes);
  for (IdType i = 0; i < segment_size; ++i) {
    local_cache_map[cache_ordered_nodes[i]] = 0; // 0 means in local GPU
  }

  std::vector<int> universal_cache_map(num_nodes, kNumSampler); // default in CPU
  for (IdType i = 0; i < kNumSampler; ++i) {
    IdType start_off = (i * segment_size);
    IdType end_off = std::min(start_off + segment_size, num_nodes);
    for (IdType j = start_off; j < end_off; ++j) {
      universal_cache_map[cache_ordered_nodes[j]] = i; // in which device
    }
  }

  std::cout << "sampler_id\tcache_type\tlocal\tnvlink\tcpu" << std::endl;
  for (int i = 0; i < kNumSampler; ++i) {
    std::vector<size_t> local_cache_count(3, 0);
    std::vector<size_t> universal_cache_count(3, 0);
    size_t total_sample_nodes = 0;

    const SamplerSteps &sampler_steps = sampler_data[i];
    for (int j = 0; j < sampler_steps.size(); ++j) {
      std::vector<size_t> nvlink_access_count(kNumSampler, 0);
      IdType num_samples = sampler_steps[j].num_samples;
      total_sample_nodes += num_samples;
      for (const auto &node_id : sampler_steps[j].samples) {
        local_cache_count[local_cache_map[node_id]] += 1;
        IdType universal_cache_type = 2;
        IdType map_id = universal_cache_map[node_id];
        if (map_id == i) {
          universal_cache_count[0] += 1;
        }
        else if (map_id >= 0 && map_id < kNumSampler) {
          nvlink_access_count[map_id] += 1;
        }
        else if (map_id == kNumSampler){
          universal_cache_count[2] += 1;
        } else {
          assert(0);
        }
      }
      universal_cache_count[1] += GetMax(nvlink_access_count);
    }

    // check
    assert(std::accumulate(
          local_cache_count.begin(), local_cache_count.end(), 0)
        == total_sample_nodes);

    /*
    std::cout << "local cache hit, total_sample_nodes: "
      << std::fixed << std::setprecision(4)
      << 1.0 * local_cache_count[0] / total_sample_nodes << ", "
      << total_sample_nodes << std::endl;

    std::cout << "sample_id " << i << ": " << std::endl;
    std::cout << std::setw(12) << "cache type"
      << std::setw(12) << "local"
      << std::setw(12) << "nvlink"
      << std::setw(12) << "cpu" << std::endl;
    std::cout << std::setw(12) << "same"
      << std::setw(12) << (double)1.0 * local_cache_count[0] * feat_dim / local_speed
      << std::setw(12) << (double)1.0 * local_cache_count[1] * feat_dim / nvlink_speed
      << std::setw(12) << (double)1.0 * local_cache_count[2] * feat_dim / pcie_speed << std::endl;
    std::cout << std::setw(12) <<  "different"
      << std::setw(12) << (double)1.0 * universal_cache_count[0] * feat_dim / local_speed
      << std::setw(12) << (double)1.0 * universal_cache_count[1] * feat_dim / nvlink_speed
      << std::setw(12) << (double)1.0 * universal_cache_count[2] * feat_dim / pcie_speed << std::endl;
    */
    std::cout << i << "\tREP" << std::fixed << std::setprecision(3)
      << "\t" << (double)1.0 * local_cache_count[0] * feat_dim / local_speed
      << "\t" << (double)1.0 * local_cache_count[1] * feat_dim / nvlink_speed
      << "\t" << (double)1.0 * local_cache_count[2] * feat_dim / pcie_speed << std::endl;
    std::cout << i <<  "\tPAR"
      << "\t" << (double)1.0 * universal_cache_count[0] * feat_dim / local_speed
      << "\t" << (double)1.0 * universal_cache_count[1] * feat_dim / nvlink_speed
      << "\t" << (double)1.0 * universal_cache_count[2] * feat_dim / pcie_speed << std::endl;
  }

  return 0;
}
