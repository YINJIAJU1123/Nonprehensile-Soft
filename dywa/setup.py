#!/usr/bin/env python3

import os
from pathlib import Path
from setuptools import setup
# from cmake_build_extension import (CMakeExtension, BuildExtension)
from torch.utils.cpp_extension import (BuildExtension, CUDAExtension)


def find_eigen_include() -> str:
    candidates = [
        os.environ.get('EIGEN3_INCLUDE_DIR'),
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        '/tmp/eigen-3.4.0',
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / 'Eigen' / 'Core').is_file():
            return candidate
    raise RuntimeError(
        'Eigen3 headers not found. Install libeigen3-dev or set '
        'EIGEN3_INCLUDE_DIR to the directory containing Eigen/Core.'
    )


EIGEN_INCLUDE = find_eigen_include()

# NOTE: we'll override `ext_modules` and `cmdclass`
# later when we implement pybind11 cxx sub-modules.
# for eigen:
# wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip eigen-3.4.0.zip && rm -f eigen-3.4.0.zip
ext_modules = [
    CUDAExtension('cxx.ur5_kin_cuda', [
        'c_src/ur5_kin_cuda.cpp',
        'c_src/ur5_kin_cuda_kernel.cu',
    ]),
    CUDAExtension('cxx.franka_kin_cuda', [
        'c_src/franka_kin_cuda.cpp',
        'c_src/franka_kin_cuda_kernel.cu',
    ],
        include_dirs=[EIGEN_INCLUDE],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O3']}
    )
]
cmdclass = {
    'build_ext': BuildExtension
}


if __name__ == '__main__':
    setup(name='dywa',
          version='0.1.0',
          ext_modules=ext_modules,
          cmdclass=cmdclass,
          scripts=[]
          )
