#pragma once

#include <vector>
#include <unordered_map>
#include "../../extra/cloud/cloud.h"

class SampledData {
public:
  int count;
  PointXYZ point;

  SampledData() {
    count = 0;
    point = PointXYZ();
  }

  void update(const PointXYZ& p) {
    count += 1;
    point += p;
  }
};

class SampledInst {
public:
  int count;
  std::vector<int> inst_ids;

  SampledInst() {
    count = 0;
  }

  void update(const int inst_id) {
    count += 1;
    inst_ids.push_back(inst_id);
  }
};

void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  std::vector<int>& inst,
  std::vector<int>& s_inst,
  float voxel_size
);

void grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  std::vector<int>& o_insts,
  std::vector<int>& s_insts,
  std::vector<long>& o_lengths,
  std::vector<long>& s_lengths,
  float voxel_size
);

