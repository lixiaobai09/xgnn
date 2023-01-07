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

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " <sample results file>"
      << " <cache ordered nodes file>"
      << " <dataset>"
      << " <cache policy>" << std::endl;
    exit(-1);
  }
  // samples result files directory
  std::string samples_file_path = argv[1];
  // cache nodes file, ordered by cache degree
  std::string cache_ordered_nodes_file = argv[2];
  std::string dataset = argv[3];
  std::string cache_policy = argv[4];

  SamplerSteps sampler_data;

  {
    size_t nbytes;
    auto data = static_cast<IdType*>(DataFromFile(samples_file_path, nbytes));

    IdType num_steps = 0;
    IdType offset = 0;
    IdType num_element = (nbytes / sizeof(IdType));
    while(offset < num_element) {
      offset += (data[offset] + 1);
      num_steps += 1;
    }
    assert(offset == num_element);

    sampler_data.resize(num_steps);
    offset = 0;
    IdType step_id = 0;
    while(offset < num_element) {
      IdType num_samples = data[offset];
      IdType *data_start = (data + offset + 1);
      std::vector<IdType> samples(data_start, data_start + num_samples);
      StepData step_data{num_samples, std::move(samples)};
      sampler_data[step_id] = std::move(step_data);

      offset += (data[offset] + 1);
      step_id += 1;
    }

    munmap(static_cast<void*>(data), nbytes);
  }

  // get cache nodes
  size_t nbytes;
  auto data = static_cast<IdType*>(DataFromFile(cache_ordered_nodes_file, nbytes));
  IdType num_nodes = (nbytes / sizeof(IdType));
  std::vector<IdType> cache_ordered_nodes(data, data + num_nodes);
  munmap(static_cast<void*>(data), nbytes);

  std::vector<IdType> node_access_count(num_nodes, 0);
  size_t total_num_samples = 0;
  for (const auto &step_data : sampler_data) {
    IdType num_samples = step_data.num_samples;
    total_num_samples += num_samples;
    for (const auto &node : step_data.samples) {
      node_access_count[node] += 1;
    }
  }

  std::vector<double> hit_rate(101, -1);
  IdType len = hit_rate.size();
  size_t cache_access_cnt = 0;
  IdType hit_rate_pos = 0;
  hit_rate[hit_rate_pos++] = 0;
  for (IdType i = 0; i < num_nodes; ++i) {
    cache_access_cnt += node_access_count[cache_ordered_nodes[i]];
    if (i >= (1ll * (num_nodes - 1) * hit_rate_pos / (len - 1))) {
      hit_rate[hit_rate_pos++] =
        static_cast<double>(1.0) * cache_access_cnt / total_num_samples;

    }
  }
  std::cout << "dataset\tcache_policy\tcache_ratio\thit_rate" << std::endl;
  for (IdType i = 0; i < len; ++i) {
    std::cout << dataset << "\t"
      << cache_policy << "\t"
      << static_cast<double>(i) / (len - 1) << "\t"
      << hit_rate[i] << std::endl;
  }

  return 0;
}
