# Chroma2 Queue Prototype Notes

This iteration introduces GPU-resident ring buffers for photon scheduling, built as a pybind11 extension with statically compiled CUDA kernels. Key components:

- `chroma2/runtime/queue/device_queue.cuh` defines the queue layout and device-side helpers.
- `chroma2/runtime/queue/device_queue.cu` implements the kernels and thin host launch wrappers compiled with `nvcc`.
- `chroma2/runtime/queue/_queue_ext.cpp` provides the pybind11 binding and manages CUDA streams, buffers, and host-device transfers.
- `chroma2/runtime/queue/benchmarks.py` measures queue throughput using Python's high-resolution timers.
- `test/test_chroma2_queue.py` exercises FIFO ordering and batching semantics (skipped when a CUDA device or the extension is missing).

## Building the Extension

Ensure the CUDA toolkit (with `nvcc`) and pybind11 headers are installed, then rebuild the project:

```bash
python -m pip install -e .[dev]
```

The setup script discovers `CUDA_HOME` / `CUDA_PATH` (default `/usr/local/cuda`) and compiles `device_queue.cu` for a set of common architectures (`sm_60`, `sm_70`, `sm_75`, `sm_80`, `sm_86`). If discovery fails, the extension is skipped and importing `DeviceQueue` will raise with guidance.

## Running the Prototype

```bash
python - <<'PY'
from chroma2.runtime.queue.benchmarks import measure_queue_round_trip
result = measure_queue_round_trip(batch_size=1_000, iterations=50)
print(result)
print(f"Throughput: {result.photons_per_second/1e6:.2f} M items/s")
PY
```

Tune `batch_size`, `iterations`, and the queue's `launch_config` to explore occupancy and contention behaviour before wiring the queue into the persistent propagation pipeline.

## Persistent Kernel Harness

Create paired queues and invoke `run_persistent_kernel` to exercise a minimal worker loop that drains the active queue and routes items to finished or spawn queues:

```python
from chroma2.runtime.queue import DeviceQueue, run_persistent_kernel
import numpy as np

active = DeviceQueue(capacity=1 << 20)
spawn = DeviceQueue(capacity=1 << 20)
finished = DeviceQueue(capacity=1 << 20)

active.push(np.arange(100_000, dtype=np.uint32))
run_persistent_kernel(active, spawn, finished, max_iterations=200_000, idle_threshold=2048,
                      spawn_interval=5, payload_increment=1, block_dim=128, grid_dim=1)
print(finished.pop(100_000))
print(spawn.pop(spawn.capacity()))
```

`spawn_interval` controls how frequently secondaries are emitted (set to zero to disable), and `payload_increment` lets you verify that photons were touched by the kernel.

> NOTE: The persistent kernel currently uses synthetic physics to exercise queue plumbing; replace the spawn/finished logic with detector-aware propagation before production use.