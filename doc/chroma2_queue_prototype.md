# Chroma2 Queue Prototype Notes

This iteration introduces GPU-resident ring buffers for photon scheduling, built as a pybind11 extension that JIT-compiles CUDA kernels with NVRTC. Key components:

- `chroma2/runtime/queue/_queue_ext.cpp` exposes a `DeviceQueue` class backed by lock-free push/pop kernels compiled at runtime.
- `chroma2/runtime/queue/__init__.py` provides a friendly import surface with build-time hints when the extension is unavailable.
- `chroma2/runtime/queue/benchmarks.py` measures queue throughput using Python's high-resolution timers.
- `test/test_chroma2_queue.py` exercises FIFO ordering and batching semantics (skipped when a CUDA device or the extension is missing).

## Building the Extension

Ensure the CUDA toolkit and pybind11 headers are installed, then rebuild the project:

```bash
python -m pip install -e .[dev]
```

The setup script detects `CUDA_HOME` / `CUDA_PATH` (default `/usr/local/cuda`) for headers and libraries. If discovery fails, no extension is built and `DeviceQueue` imports will raise with guidance.

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
