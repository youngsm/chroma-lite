# Chroma2 Queue Prototype Notes

This prototype introduces GPU-resident ring buffers for photon scheduling. The initial drop adds:

- `chroma2/runtime/queue/device_queue.cuh` with lock-free push/pop helpers built on atomic CAS loops.
- CUDA kernels in `device_queue.cu` for resetting, bulk push/pop, and draining queues to benchmark contention.
- `prototype.py` PyCUDA helpers that allocate queue storage, compile kernels, and expose `push`, `pop`, and `drain` methods for experiments.
- `benchmarks.py` convenience routines that measure push/pop throughput using CUDA events.
- A pytest harness (`test/test_chroma2_queue.py`) that validates FIFO ordering for single- and multi-batch scenarios (skipped automatically when no GPU is present).

To try the benchmark locally run:

```bash
python - <<'PY'
from chroma2.runtime.queue.benchmarks import measure_queue_round_trip
result = measure_queue_round_trip(batch_size=1_000, iterations=50)
print(result)
print(f"Throughput: {result.photons_per_second/1e6:.2f} M items/s")
PY
```

Adjust `batch_size`, `iterations`, and grid/block dimensions in `DeviceQueuePrototype` to profile hotspot behaviour before integrating with the persistent propagation kernel.
