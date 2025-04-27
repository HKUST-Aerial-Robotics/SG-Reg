''' This is a module adopted from SG-PGM:
    git@github.com:dfki-av/sg-pgm.git
    It maintains the instance label while downsampling the point cloud.
'''

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='kpconv_extension',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='.ext',
            sources=[
                'sgreg/extensions/extra/cloud/cloud.cpp',
                'sgreg/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'sgreg/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'sgreg/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'sgreg/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'sgreg/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
