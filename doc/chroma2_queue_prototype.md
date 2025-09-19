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
