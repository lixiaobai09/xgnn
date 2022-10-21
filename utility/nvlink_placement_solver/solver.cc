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
  Solver(int n);
  void ReadData(std::string path);
  void Solve();

 private:
  std::vector<std::vector<double>> _bandwidth_matrix;
  int _num_ctx;
};

Solver::Solver(int n) {
  _num_ctx = n;
  _bandwidth_matrix.resize(n, std::vector<double>(n, 0.0));
}

void Solver::ReadData(std::string path) {
  std::ifstream ifile(path);
  int num_node, num_edge;
  int a, b;
  double weight;
  ifile >> num_node >> num_edge;
  assert(num_node == _num_ctx);
  while(num_edge--) {
    ifile >> a >> b >> weight;
    _bandwidth_matrix[a][b] = weight;
    if (a != b) {
      _bandwidth_matrix[b][a] = weight;
    }
  }
#ifdef DEBUG
  std::cout << std::setw(6) << std::fixed << std::setprecision(2)
    << 0 << " ";
  for (int i = 0; i < _num_ctx; ++i) {
    std::cout << std::setw(6) << std::fixed << std::setprecision(2)
      << i << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < _num_ctx; ++i) {
    std::cout << std::setw(6) << std::fixed << std::setprecision(2)
      << i << " ";
    for (int j = 0; j < _num_ctx; ++j) {
      std::cout << std::setw(6) << std::fixed << std::setprecision(2)
        << _bandwidth_matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
#endif
}

template<typename T>
std::set<T> operator-(const std::set<T> &a, const std::set<T> &b) {
  std::set<T> ret;
  for (auto i : a) {
    if (!b.count(i)) {
      ret.insert(i);
    }
  }
  return std::move(ret);
};

void Solver::Solve() {
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

  for (int i = 0; i < _num_ctx; ++i) {
    for (int j = 0; j < _num_ctx; ++j) {
      auto group_ctx_id = access_part_ctx[i][j];
      if (!store_parts[group_ctx_id].count(j)) {
        std::cout << "\033[31mERROR\033[0m " << i << " can not access part "
          << j << " in GPU " << group_ctx_id << std::endl;
      }
    }
  }

#endif
}

int main(int argc, char** argv) {
  int n = std::atoi(argv[1]);
  Solver solver(n);
  solver.ReadData("./input.txt");
  solver.Solve();
  return 0;
}
