import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["src/ext_nccl_backend.cpp"]
include_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/include/",
]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="ext_nccl_backend",
        sources=sources,
        include_dirs=include_dirs,
    )
else:
    module = cpp_extension.CppExtension(
        name="ext_nccl_backend",
        sources=sources,
        include_dirs=include_dirs,
    )

setup(
    name="ext_nccl_backend",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
