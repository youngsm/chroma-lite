"""PyCUDA helpers for experimenting with DeviceQueue kernels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401 - ensures a context exists for prototype usage
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "pycuda is required to use the chroma2 queue prototype"  # noqa: B904
    ) from exc

MODULE_PATH = Path(__file__).with_name("device_queue.cu")
INCLUDE_DIR = MODULE_PATH.parent

QUEUE_DTYPE = np.dtype(
    [
        ("buffer", np.uint64),
        ("head", np.uint64),
        ("tail", np.uint64),
        ("capacity", np.uint32),
        ("mask", np.uint32),
    ]
)


@dataclass
class DeviceQueueHandles:
    module: SourceModule
    reset_kernel: cuda.Function
    push_kernel: cuda.Function
    pop_kernel: cuda.Function
    drain_kernel: cuda.Function


def _load_module(options: Tuple[str, ...] | None = None) -> DeviceQueueHandles:
    opts = ["-std=c++14"]
    if options:
        opts.extend(options)
    source = MODULE_PATH.read_text()
    module = SourceModule(
        source,
        options=opts,
        include_dirs=[str(INCLUDE_DIR)],
        no_extern_c=True,
    )
    return DeviceQueueHandles(
        module=module,
        reset_kernel=module.get_function("dq_reset"),
        push_kernel=module.get_function("dq_push_kernel"),
        pop_kernel=module.get_function("dq_pop_kernel"),
        drain_kernel=module.get_function("dq_drain"),
    )


class DeviceQueuePrototype:
    """Convenience wrapper that manages queue buffers and kernel launches."""

    def __init__(self, capacity: int, block_dim: int = 128, grid_dim: int = 4):
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power of two for mask arithmetic")
        self.capacity = capacity
        self.block_dim = block_dim
        self.grid_dim = grid_dim
        self.handles = _load_module()
        self.buffer = gpuarray.empty(capacity, dtype=np.uint32)
        self.head = gpuarray.zeros(1, dtype=np.uint32)
        self.tail = gpuarray.zeros(1, dtype=np.uint32)
        self._struct = np.zeros(1, dtype=QUEUE_DTYPE)
        self._struct["buffer"][0] = int(self.buffer.gpudata)
        self._struct["head"][0] = int(self.head.gpudata)
        self._struct["tail"][0] = int(self.tail.gpudata)
        self._struct["capacity"][0] = capacity
        self._struct["mask"][0] = capacity - 1
        self.reset()

    @property
    def struct(self) -> np.ndarray:
        return self._struct

    def reset(self) -> None:
        self.handles.reset_kernel(
            cuda.In(self._struct),
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )
        cuda.Context.synchronize()

    def push(self, values: np.ndarray) -> None:
        values_gpu = gpuarray.to_gpu(values.astype(np.uint32, copy=False))
        self.handles.push_kernel(
            cuda.In(self._struct),
            values_gpu.gpudata,
            np.uint32(values_gpu.size),
            block=(self.block_dim, 1, 1),
            grid=(self.grid_dim, 1, 1),
        )
        cuda.Context.synchronize()

    def pop(self, count: int) -> np.ndarray:
        out_gpu = gpuarray.empty(count, dtype=np.uint32)
        self.handles.pop_kernel(
            cuda.In(self._struct),
            out_gpu.gpudata,
            np.uint32(count),
            block=(self.block_dim, 1, 1),
            grid=(self.grid_dim, 1, 1),
        )
        cuda.Context.synchronize()
        return out_gpu.get()

    def drain(self) -> int:
        counter = gpuarray.zeros(1, dtype=np.uint32)
        self.handles.drain_kernel(
            cuda.In(self._struct),
            counter.gpudata,
            block=(1, 1, 1),
            grid=(1, 1, 1),
        )
        cuda.Context.synchronize()
        return int(counter.get()[0])


__all__ = ["DeviceQueuePrototype", "DeviceQueueHandles"]
