# use the command below to install:
# pip install -e . --no-build-isolation -v

import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

# self include and source
sources = [
    "src/ext_nccl_backend.cpp",
    "src/ext_nccl_comm.cpp",
    "src/ext_flight_recorder.cpp",
    "src/group_collective.cu",
]

include_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/include/",
    "/home/littsk/kato/workspace/attn-library/magiattn/magi_attention/csrc/cutlass/include/,"
]

# torch include and source
# torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
# torch_include_path = os.path.join(os.path.dirname(torch.__file__), 'include')
# include_dirs.append(torch_include_path)

# extra_link_args = ['-Wl,-rpath,' + torch_lib_path]
# library_dirs = [torch_lib_path]
# libraries = ['torch_python', 'torch', 'torch_cuda', 'c10', 'c10_cuda']

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="ext_nccl_backend",
        sources=sources,
        include_dirs=include_dirs,
        # library_dirs=library_dirs,
        # libraries=libraries,
        # extra_link_args=extra_link_args,
        extra_compile_args={
            # 'cxx': ['-std=c++17', '-fPIC'],
            'nvcc': ['-lineinfo']
        }
    )
else:
    module = cpp_extension.CppExtension(
        name="ext_nccl_backend",
        sources=sources,
        include_dirs=include_dirs,
        # libraries=libraries,
        # extra_link_args=extra_link_args,
        # extra_compile_args=['-std=c++17', '-fPIC']
    )
    
# pyi
package_data = {
    '': ['*.pyi'],
}


setup(
    name="ext_nccl_backend",
    version="0.0.1",
    
    package_data=package_data,
    include_package_data=True,
    python_requires='>=3.10',
    
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
)
