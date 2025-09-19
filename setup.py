import os
import subprocess
from pathlib import Path
from typing import List, Tuple

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind_build_ext
except ImportError:  # pragma: no cover - pybind11 provided via setup_requires at build time
    Pybind11Extension = None
    pybind_build_ext = None


def locate_cuda() -> Tuple[str, str, str]:
    cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_path is None:
        raise FileNotFoundError(
            "CUDA toolkit not found; set CUDA_HOME or CUDA_PATH to your installation root"
        )
    include_dir = os.path.join(cuda_path, "include")
    lib_dir_candidates = [
        os.path.join(cuda_path, "lib64"),
        os.path.join(cuda_path, "lib"),
    ]
    lib_dir = next((path for path in lib_dir_candidates if os.path.isdir(path)), None)
    bin_dir = os.path.join(cuda_path, "bin")
    nvcc = os.path.join(bin_dir, "nvcc")
    if not os.path.isdir(include_dir) or lib_dir is None or not os.path.exists(nvcc):
        raise FileNotFoundError(
            "CUDA toolkit not found; set CUDA_HOME or CUDA_PATH to your installation root"
        )
    return include_dir, lib_dir, nvcc


if Pybind11Extension is not None:
    try:
        include_dir, lib_dir, nvcc_path = locate_cuda()

        class BuildExt(pybind_build_ext):
            def build_extensions(self) -> None:  # type: ignore[override]
                for ext in self.extensions:
                    cuda_sources = [src for src in ext.sources if src.endswith(".cu")]
                    ext.sources = [src for src in ext.sources if not src.endswith(".cu")]
                    if cuda_sources:
                        objects: List[str] = []
                        for source in cuda_sources:
                            objects.append(self.compile_cuda(source, nvcc_path, ext))
                        ext.extra_objects = getattr(ext, "extra_objects", []) + objects
                super().build_extensions()

            def compile_cuda(self, source: str, nvcc: str, ext) -> str:
                build_temp = Path(self.build_temp)
                build_temp.mkdir(parents=True, exist_ok=True)
                obj_path = build_temp / (Path(source).stem + ".o")
                arch_list = ["60", "70", "75", "80", "86"]
                arch_flags = []
                for arch in arch_list:
                    arch_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
                cmd = [
                    nvcc,
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "-c",
                    source,
                    "-o",
                    str(obj_path),
                ] + arch_flags
                include_dirs = list(set(ext.include_dirs + [include_dir]))
                for inc in include_dirs:
                    cmd.extend(["-I", inc])
                subprocess.check_call(cmd)
                return str(obj_path)

        ext_modules.append(
            Pybind11Extension(
                "chroma2.runtime.queue._queue_ext",
                [
                    "chroma2/runtime/queue/_queue_ext.cpp",
                    "chroma2/runtime/queue/device_queue.cu",
                ],
                include_dirs=[include_dir],
                libraries=["cudart"],
                library_dirs=[lib_dir],
                extra_compile_args=['-std=c++17'],
            )
        )
        cmdclass["build_ext"] = BuildExt
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
