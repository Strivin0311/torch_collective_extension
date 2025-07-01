# use the command below to install:
# pip install -e . --no-build-isolation -v

import os
import glob
import torch
from setuptools import setup
from torch.utils import cpp_extension

csrc_root = "csrc/comm"

# self include and source
sources = []
sources.extend(glob.glob(f"{csrc_root}/*.cpp"))
sources.extend(glob.glob(f"{csrc_root}/*.cu"))

include_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/{csrc_root}/",
    "/home/littsk/kato/workspace/attn-library/magiattn/magi_attention/csrc/cutlass/include/", # NOTE: replace to your own cutlass path
]

# torch include and source
# torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
# torch_include_path = os.path.join(os.path.dirname(torch.__file__), 'include')
# include_dirs.append(torch_include_path)

# extra_link_args = ['-Wl,-rpath,' + torch_lib_path]
# library_dirs = [torch_lib_path]
# libraries = ['torch_python', 'torch', 'torch_cuda', 'c10', 'c10_cuda']

assert torch.cuda.is_available(), "MagiNCCL backend requires CUDA"

module = cpp_extension.CUDAExtension(
    name="magi_nccl",
    sources=sources,
    include_dirs=include_dirs,
    # library_dirs=library_dirs,
    # libraries=libraries,
    # extra_link_args=extra_link_args,
    extra_compile_args={
        'nvcc': [
            '-O3',
            '-lineinfo', 
            "-Xptxas", 
            "-v", 
            "-gencode", "arch=compute_90,code=sm_90",
        ]
    }
)
    
# pyi
package_data = {
    '': ['*.pyi'],
}

# set up
setup(
    name="magi_nccl",
    version="0.0.1",
    
    package_data=package_data,
    include_package_data=True,
    python_requires='>=3.10',
    
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
)
