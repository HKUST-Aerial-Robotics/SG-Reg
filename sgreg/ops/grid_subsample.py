import importlib


ext_module = importlib.import_module('ext')


def grid_subsample(points, insts, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    s_points, s_insts, s_lengths = ext_module.grid_subsampling(points, insts, lengths, voxel_size)
    return s_points, s_insts, s_lengths
