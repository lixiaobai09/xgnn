#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <tuple>

#define DEBUG

class Solver {
 public:
  Solver() : _num_ctx(0) {};
  void Solve(int n_gpu, const std::vector<std::vector<double>> &matrix);

 private:
  std::vector<std::vector<double>> _bandwidth_matrix;
  int _num_ctx;
};

template<typename T>
std::set<T> operator- (const std::set<T> &a, const std::set<T> &b) {
  std::set<T> ret;
  for (auto i : a) {
    if (!b.count(i)) {
      ret.insert(i);
    }
  }
  return std::move(ret);
};

void Solver::Solve(int n_gpu,
    const std::vector<std::vector<double>> &matrix_input) {
  _num_ctx = n_gpu;
  _bandwidth_matrix = matrix_input;

  std::vector<std::vector<int>> access_count(
      _num_ctx, std::vector<int>(_num_ctx, 0));
  std::vector<std::vector<int>> access_part_ctx(
      _num_ctx, std::vector<int>(_num_ctx, -1));
  std::vector<std::set<int>> store_parts(_num_ctx);

  std::vector<std::set<int>> can_access_parts(_num_ctx);
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
  // iterator for each GPU ctx
  for (int i = 0; i < _num_ctx; ++i) {
    // get can not access parts for GPU i
    auto can_not_access_parts = (parts_universal_set - can_access_parts[i]);
    for (auto need_part : can_not_access_parts) {
      // id, stored_parts_size, need_score, if_same_part_in_neighbors, bandwith
      std::vector<std::tuple<int, int, int, int, double>> tmp_vec;
      const std::vector<double> &neighbor_bandwidth = _bandwidth_matrix[i];
      // iterate GPU_i neighbors
      for(auto j : neighbor_adjacency[i]) {
        int need_score = 0;
        for (auto k : neighbor_adjacency[j]) {
          if(!can_access_parts[k].count(j)) {
            ++need_score;
          }
        }
        tmp_vec.emplace_back(j, store_parts[j].size(), need_score,
            can_access_parts[j].count(need_part),
            _bandwidth_matrix[i][j] / (access_count[i][j] + 1));
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
            // bandwith
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
    assert(can_access_parts[i].size() == _num_ctx);
    for (int j = 0; j < _num_ctx; ++j) {
      int which_gpu;
      double max_bandwith = 0.0;
      for(auto neighbor : neighbor_adjacency[i]) {
        if (store_parts[neighbor].count(j)) {
          double tmp_bandwith =
            _bandwidth_matrix[i][neighbor] / (access_count[i][neighbor] + 1);
          if (tmp_bandwith > max_bandwith) {
            max_bandwith = tmp_bandwith;
            which_gpu = neighbor;
          }
        }
      }
      access_part_ctx[i][j] = which_gpu;
      access_count[i][j] += 1;
    }
  }

#ifdef DEBUG
  for (int i = 0; i < _num_ctx; ++i) {
    assert(can_access_parts[i].size() == _num_ctx);
    std::cout << i << ", [ ";
    for (auto j : store_parts[i]) {
      std::cout << j << " ";
    }
    std::cout << "], [ ";
    for (auto j : access_part_ctx[i]) {
      std::cout << j << " ";
    }
    std::cout << "]" << std::endl;
  }

  std::cout << "---- access matrix ----" << std::endl;
  for (int i = 0; i < _num_ctx; ++i) {
    for (int j = 0; j < _num_ctx; ++j) {
      std::cout << access_count[i][j] << " ";
    }
    std::cout << std::endl;
  }
#endif

  for (int i = 0; i < _num_ctx; ++i) {
    for (int j = 0; j < _num_ctx; ++j) {
      auto group_ctx_id = access_part_ctx[i][j];
      if (!store_parts[group_ctx_id].count(j)) {
        std::cout << "\033[31mERROR\033[0m " << i << " can not access part "
          << j << " in GPU " << group_ctx_id << std::endl;
      }
      if (access_count[i][j] != 1) {
        std::cout << "\033[31mWARN\033[0m access imbalance" << std::endl;
      }
    }
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
  auto bandwidth_matrix = ReadDataFromFile("./input.txt");
  int n_gpu = bandwidth_matrix.size();
  std::vector<int> map_vec(n_gpu);
  for (int i = 0; i < n_gpu; ++i) {
    map_vec[i] = i;
  }

  Solver solver;

#ifdef DEBUG
  int t = 5;
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

  return 0;
}
