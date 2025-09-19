import os
from typing import Tuple

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:  # pragma: no cover - pybind11 provided via setup_requires at build time
    Pybind11Extension = None
    build_ext = None


def locate_cuda() -> Tuple[str, str]:
    cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    include_dir = os.path.join(cuda_path, "include")
    lib_dir_candidates = [
        os.path.join(cuda_path, "lib64"),
        os.path.join(cuda_path, "lib"),
    ]
    lib_dir = next((path for path in lib_dir_candidates if os.path.isdir(path)), None)
    if not os.path.isdir(include_dir) or lib_dir is None:
        raise FileNotFoundError(
            "CUDA toolkit not found; set CUDA_HOME or CUDA_PATH to your installation root"
        )
    return include_dir, lib_dir


if Pybind11Extension is not None:
    try:
        include_dir, lib_dir = locate_cuda()
        ext_modules.append(
            Pybind11Extension(
                "chroma2.runtime.queue._queue_ext",
                ["chroma2/runtime/queue/_queue_ext.cpp"],
                include_dirs=[include_dir],
                libraries=["cuda", "cudart", "nvrtc"],
                library_dirs=[lib_dir],
                extra_compile_args=["-std=c++17"],
            )
        )
        cmdclass["build_ext"] = build_ext
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Warning: CUDA queue extension disabled: {exc}")

setup(
    name="Chroma",
    version="0.5",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "chroma": ["cuda/*.cu", "cuda/*.h"],
    },
    scripts=[
        "bin/chroma-sim",
        "bin/chroma-cam",
        "bin/chroma-geo",
        "bin/chroma-bvh",
        "bin/chroma-server",
    ],
    setup_requires=["pybind11>=2.11"],
    install_requires=[
        "uncertainties",
        "pyzmq",
        "pycuda",
        "pytools==2022.1.2",
        "numpy>=1.6",
        "pygame",
        "nose",
        "sphinx",
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    # test_suite = 'nose.collector',
)
