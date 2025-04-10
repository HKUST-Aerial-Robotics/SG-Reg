# Geotransformer with grid subsampling for point clouds with semantic labels
## Grid Subsampling （Modified with semantic labels）

This code implements the grid subsampling algorithm for point clouds. It performs grid subsampling on each batch of points and returns the subsampled points, subsampled instance labels, and lengths of the subsampled point cloud batches.

It provides a function `grid_subsampling_cpu` that takes in a set of 3D points, **their corresponding instance labels**, and lengths of point cloud batches. 

## Radius-based Neighbor Search
The main function, `radius_neighbors_cpu`, performs a batched search on two sets of 3D points, storing the indices of the neighbors found within a given radius for each point in the first set.


