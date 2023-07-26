#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <limits>
#include <tuple>

#include "timer.h"

// #define DEBUG

int check_max_parts = 2;

class Solver {
 public:
  Solver() : _num_ctx(0) {};
  void Solve(int n_gpu, const std::vector<std::vector<double>> &matrix);

 private:
  std::vector<std::vector<double>> _bandwidth_matrix;
  int _num_ctx;
};

template<typename T>
std::vector<T> operator- (const std::set<T> &a, const std::multiset<T> &b) {
  std::vector<T> ret;
  for (auto i : a) {
    if (!b.count(i)) {
      ret.emplace_back(i);
    }
  }
  return std::move(ret);
};

namespace {

void search_access_config(int gpu,
    int part_id,
    int n_part,
    std::vector<int> &access_config,
    std::vector<int> &result,
    double &min_bandwidth,
    const std::vector<std::set<int>> &part_gpu_map,
    const std::vector<std::vector<double>> &bandwidth_matrix) {
  if (part_id == n_part) {
    double max_bandwidth = 0.0;
    const std::vector<double> &bandwidth_list = bandwidth_matrix[gpu];
    int n_gpu = bandwidth_list.size();
    std::vector<int> access_cnt(n_gpu, 0);
    for (auto gpu_i : access_config) {
      access_cnt[gpu_i] += 1;
    }
    for (int i = 0; i < n_gpu; ++i) {
      if (access_cnt[i]
          && (
            static_cast<double>(access_cnt[i]) / bandwidth_list[i]
            > max_bandwidth
          )) {
        max_bandwidth = static_cast<double>(access_cnt[i]) / bandwidth_list[i];
      }
    }
    if (max_bandwidth < min_bandwidth) {
      min_bandwidth = max_bandwidth;
      result = access_config;
    }
    return;
  }
  for (auto gpu_j : part_gpu_map[part_id]) {
    access_config.emplace_back(gpu_j);
    search_access_config(gpu, part_id + 1, n_part,
        access_config, result, min_bandwidth,
        part_gpu_map, bandwidth_matrix);
    access_config.pop_back();
  }
}

using ResultType = std::vector<std::tuple<std::set<int>, std::vector<int>>>;

void solver_recursive(int current_gpu,
    int n_gpu,
    int access_current_id,
    std::vector<int> can_not_access_parts,
    std::vector<std::set<int>> &store_parts,
    std::vector<std::multiset<int>> &can_access_parts,
    ResultType &result,
    double &min_max_bandwidth,
    const std::set<int> &parts_universal_set,
    const std::vector<std::set<int>> &neighbor_adjacency,
    const std::vector<std::vector<double>> &bandwidth_matrix) {

  // stop condition
  if (current_gpu == n_gpu) {
    std::vector<std::vector<int>> access_config_list;
    double max_bandwidth = 0.0;
    for (int gpu = 0; gpu < store_parts.size(); ++gpu) {
      // std::cout << "gpu: " << gpu << std::endl;
      std::vector<std::set<int>> part_gpu_map(n_gpu);
      for (auto neighbor : neighbor_adjacency[gpu]) {
        for (auto store_part : store_parts[neighbor]) {
          part_gpu_map[store_part].insert(neighbor);
        }
      }
      std::vector<int> access_config;
      std::vector<int> result;
      double min_bandwidth = std::numeric_limits<double>::max();
      search_access_config(gpu, 0, n_gpu, access_config, result,
          min_bandwidth, part_gpu_map, bandwidth_matrix);
      if (min_bandwidth > max_bandwidth) {
        max_bandwidth = min_bandwidth;
      }
      access_config_list.emplace_back(result);
      // for (auto gpu_j : result) { std::cout << gpu_j << " "; }
      // std::cout << std::endl;
    }
    if (max_bandwidth < min_max_bandwidth) {
      min_max_bandwidth = max_bandwidth;
      result.clear();
      for (int i = 0; i < n_gpu; ++i) {
        result.emplace_back(store_parts[i], access_config_list[i]);
      }
    }
    return;
  }
  if (can_not_access_parts.size() == 0) {
    // get can not access parts for GPU i
    can_not_access_parts =
      (parts_universal_set - can_access_parts[current_gpu]);
    access_current_id = 0;
  }
  if (access_current_id < can_not_access_parts.size()) {
    int need_part = can_not_access_parts[access_current_id];
    // id, stored_parts_size, need_score, if_same_part_in_neighbors
    std::vector<std::tuple<int, int, int, int>> tmp_vec;
    for (auto j : neighbor_adjacency[current_gpu]) {
      int need_score = 0;
      for (auto k : neighbor_adjacency[j]) {
        if (!can_access_parts[k].count(need_part)) {
          ++need_score;
        }
      }
      tmp_vec.emplace_back(j, store_parts[j].size(), need_score,
          // XXX: if need this?
          (can_access_parts[j].count(need_part) == 0? 0 : 1));
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
          return std::get<0>(x) < std::get<0>(y);
        });
    int last = (tmp_vec.size() - 1);
    auto cmp_equal = [](const std::tuple<int, int, int, int> &x,
        const std::tuple<int, int, int, int> &y) {
      return (std::get<1>(x) == std::get<1>(y)
          && std::get<2>(x) == std::get<2>(y)
          && std::get<3>(x) == std::get<3>(y));
    };
    while (!cmp_equal(tmp_vec[0], tmp_vec[last])) {
      tmp_vec.pop_back();
      --last;
    }
    for (auto item : tmp_vec) {
      int store_gpu = std::get<0>(item);
      store_parts[store_gpu].insert(need_part);
      for (auto neighbor : neighbor_adjacency[store_gpu]) {
        can_access_parts[neighbor].insert(need_part);
      }
      solver_recursive(current_gpu, n_gpu,
          access_current_id + 1, can_not_access_parts,
          store_parts, can_access_parts,
          result, min_max_bandwidth,
          parts_universal_set,
          neighbor_adjacency, bandwidth_matrix);
      // recover
      store_parts[store_gpu].erase(store_parts[store_gpu].find(need_part));
      for (auto neighbor : neighbor_adjacency[store_gpu]) {
        can_access_parts[neighbor].erase(
            can_access_parts[neighbor].find(need_part));
      }
    }
  } else {
    can_not_access_parts.clear();
    solver_recursive(current_gpu + 1, n_gpu, 0, can_not_access_parts,
        store_parts, can_access_parts,
        result, min_max_bandwidth,
        parts_universal_set, neighbor_adjacency,
        bandwidth_matrix);
  }

}

} // namespace

void Solver::Solve(int n_gpu,
    const std::vector<std::vector<double>> &matrix_input) {
  _num_ctx = n_gpu;
  _bandwidth_matrix = matrix_input;

  std::vector<std::vector<int>> access_count(
      _num_ctx, std::vector<int>(_num_ctx, 0));
  std::vector<std::set<int>> store_parts(_num_ctx);

  std::vector<std::multiset<int>> can_access_parts(_num_ctx);
  // from bandwith matrix
  std::vector<std::set<int>> neighbor_adjacency(_num_ctx);
  std::set<int> parts_universal_set;
  for (int i = 0; i < _num_ctx; ++i) {
    parts_universal_set.insert(i);
    store_parts[i].insert(i);
    for (int j = 0; j < _num_ctx; ++j) {
      if (_bandwidth_matrix[i][j] != 0.0) {
        can_access_parts[i].insert(j);
        neighbor_adjacency[i].insert(j);
      }
    }
  }
  ResultType result;
  double max_avg_bandwidth = std::numeric_limits<double>::max();
  solver_recursive(0, n_gpu, 0, {},
      store_parts, can_access_parts,
      result, max_avg_bandwidth,
      parts_universal_set, neighbor_adjacency,
      _bandwidth_matrix);
  for (int i = 0; i < n_gpu; ++i) {
    std::cout << "gpu " << i << ": [ ";
    for (auto part_id : std::get<0>(result[i])) {
      std::cout << part_id << " ";
    }
    std::cout << "], [ ";
    for (auto gpu_id : std::get<1>(result[i])) {
      std::cout << gpu_id << " ";
    }
    std::cout << "]" << std::endl;
  }



}

void PrintBandwithMatrix(
    const std::vector<std::vector<double>> &bandwidth_matrix) {
  int num_node = bandwidth_matrix.size();

  std::cout << std::setw(6) << std::fixed << std::setprecision(2)
    << 0 << " ";
  for (int i = 0; i < num_node; ++i) {
    std::cout << std::setw(6) << std::fixed << std::setprecision(2)
      << i << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < num_node; ++i) {
    std::cout << std::setw(6) << std::fixed << std::setprecision(2)
      << i << " ";
    for (int j = 0; j < num_node; ++j) {
      std::cout << std::setw(6) << std::fixed << std::setprecision(2)
        << bandwidth_matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }

}

std::vector<std::vector<double>> ReadDataFromFile(std::string path) {
  int num_node, num_edge;
  int a, b;
  double weight;
  std::vector<std::vector<double>> bandwidth_matrix;

  std::ifstream ifile(path);
  ifile >> num_node >> num_edge;

  bandwidth_matrix.resize(num_node, std::vector<double>(num_node, 0.0));

  while(num_edge--) {
    ifile >> a >> b >> weight;
    bandwidth_matrix[a][b] = weight;
    if (a != b) {
      bandwidth_matrix[b][a] = weight;
    }
  }
  PrintBandwithMatrix(bandwidth_matrix);
  return std::move(bandwidth_matrix);
}

int main(int argc, char** argv) {
  std::string input_file = argv[1];
  // max part size in each GPU
  check_max_parts = std::stoi(argv[2]);

  auto bandwidth_matrix = ReadDataFromFile(input_file);
  int n_gpu = bandwidth_matrix.size();
  // 0 1 2 6 5 4 3 7 a example link not be used for 8-GPU
  std::vector<int> map_vec(n_gpu);
  for (int i = 0; i < n_gpu; ++i) {
    map_vec[i] = i;
  }
  // map_vec = {0, 1, 3, 5, 4, 2};
  // map_vec = {2, 4, 0, 3, 5, 1};

  Timer t;
  Solver solver;
  solver.Solve(n_gpu, bandwidth_matrix);
  double solve_time = t.PassedMicro();
  std::cout << "time cost: " << solve_time << " us." << std::endl;

  /*
#ifdef DEBUG
  int t = 10;
#endif

  do {
    std::vector<std::vector<double>> maped_bandwidth_matrix(n_gpu,
        std::vector<double>(n_gpu, 0.0));
    for (int i = 0; i < n_gpu; ++i) {
      for (int j = 0; j < n_gpu; ++j) {
        maped_bandwidth_matrix[map_vec[i]][map_vec[j]] =
          bandwidth_matrix[i][j];
      }
    }
    solver.Solve(n_gpu, maped_bandwidth_matrix);

#ifdef DEBUG
    std::cout << "---------------" << std::endl;
    for (int i = 0; i < n_gpu; ++i) std::cout << map_vec[i] << " ";
    std::cout << std::endl;
    PrintBandwithMatrix(maped_bandwidth_matrix);
    if (!(t--)) break;
#endif

  } while(std::next_permutation(map_vec.begin(), map_vec.end()));
  */

  return 0;
}
