#ifndef ICS22_SONG_DIST_FEATURE_H
#define ICS22_SONG_DIST_FEATURE_H

#include "dist_graph.h"
#include <cassert>

namespace samgraph {
namespace common {
namespace cuda {


class DeviceICS22SongDistFeature {
 public:
  DeviceICS22SongDistFeature() = default;
  DeviceICS22SongDistFeature(void **part_feature_data,
      IdType *device_map,
      IdType *new_idx_map,
      IdType num_cache_node, IdType dim)
    : _part_feature_data(part_feature_data),
      _device_map(device_map),
      _new_idx_map(new_idx_map),
      _num_cache_node(num_cache_node),
      _dim(dim) {};

  template<typename T>
  inline __device__ const T& Get(IdType node_id, IdType col) {
    assert(node_id < _num_cache_node);
    assert(col < _dim);
    IdType part_id, real_id;
    _GetRealPartId(node_id, &part_id, &real_id);
    auto part_ptr = static_cast<T*>(_part_feature_data[part_id]);
    return part_ptr[real_id * _dim + col];
  }

 private:
  inline __device__ void _GetRealPartId(IdType node_id,
    IdType *part_id, IdType *real_id) {
    *part_id = _device_map[node_id];
    *real_id = _new_idx_map[node_id];
  }
  void **_part_feature_data;
  IdType *_device_map;
  IdType *_new_idx_map;
  IdType _num_cache_node;
  IdType _dim;
};

class ICS22SongDistGraph : public DistGraph {
 public:
  void FeatureLoad(int trainer_id, Context trainer_ctx,
      const IdType *cache_rank_node, const IdType num_cache_node,
      DataType dtype, size_t dim,
      const void* cpu_src_feature_data,
      StreamHandle stream = nullptr) override;
  DeviceICS22SongDistFeature DeviceFeatureHandle() const;
  static void Create(std::vector<Context> ctxes, IdType clique_size,
      Dataset *dataset,
      const double alpha,
      IdType num_feature_cached_node);
  static void Release(DistGraph *dist_graph);
  ~ICS22SongDistGraph() { LOG(ERROR) << "Do not call function in here"; };

 private:
  ICS22SongDistGraph() = delete;
  ICS22SongDistGraph(const ICS22SongDistGraph &graph) = delete;
  ICS22SongDistGraph& operator=(const ICS22SongDistGraph& graph) = delete;
  ICS22SongDistGraph(std::vector<Context> ctxes, IdType clique_size,
      Dataset *dataset,
      const double alpha,
      IdType num_feature_cached_node);

  std::vector<Context> _ctxes;
  std::vector<TensorPtr> _h_device_map_vec;
  std::vector<TensorPtr> _h_new_idx_map_vec;
  std::vector<TensorPtr> _h_device_cached_nodes_vec;

  IdType *_d_device_map;
  IdType *_d_new_idx_map;
};

} // cuda
} // common
} // namespace samgraph

#endif // ICS22_SONG_DIST_FEATURE_H
