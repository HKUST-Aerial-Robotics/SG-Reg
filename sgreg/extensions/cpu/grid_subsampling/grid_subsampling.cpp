#include <cstring>
#include "grid_subsampling.h"
#include "grid_subsampling_cpu.h"

std::vector<at::Tensor> grid_subsampling(
  at::Tensor points,
  at::Tensor insts,
  at::Tensor lengths,
  float voxel_size
) {
  CHECK_CPU(points);
  CHECK_CPU(lengths);
  CHECK_IS_FLOAT(points);
  CHECK_IS_LONG(lengths);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(lengths);

  CHECK_CPU(insts);
  CHECK_IS_INT(insts);
  CHECK_CONTIGUOUS(insts);

  std::size_t batch_size = lengths.size(0);
  std::size_t total_points = points.size(0);

  std::vector<PointXYZ> vec_points = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_points;

  std::vector<long> vec_lengths = std::vector<long>(
    lengths.data_ptr<long>(),
    lengths.data_ptr<long>() + batch_size
  );
  std::vector<long> vec_s_lengths;

  std::vector<int> vec_insts = std::vector<int>(
    reinterpret_cast<int*>(insts.data_ptr<int>()),
    reinterpret_cast<int*>(insts.data_ptr<int>()) + total_points
  );
  std::vector<int> vec_s_insts;

  grid_subsampling_cpu(
    vec_points,
    vec_s_points,
    vec_insts,
    vec_s_insts,
    vec_lengths,
    vec_s_lengths,
    voxel_size
  );

  std::size_t total_s_points = vec_s_points.size();
  at::Tensor s_points = torch::zeros(
    {total_s_points, 3},
    at::device(points.device()).dtype(at::ScalarType::Float)
  );
  at::Tensor s_lengths = torch::zeros(
    {batch_size},
    at::device(lengths.device()).dtype(at::ScalarType::Long)
  );

  at::Tensor s_insts = torch::zeros(
    {total_s_points, 1},
    at::device(insts.device()).dtype(at::ScalarType::Int)
  );

  std::memcpy(
    s_points.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_points.data()),
    sizeof(float) * total_s_points * 3
  );
  std::memcpy(
    s_lengths.data_ptr<long>(),
    vec_s_lengths.data(),
    sizeof(long) * batch_size
  );

  std::memcpy(
    s_insts.data_ptr<int>(),
    reinterpret_cast<int*>(vec_s_insts.data()),
    sizeof(int) * total_s_points * 1
  );


  return {s_points, s_insts, s_lengths};
}
