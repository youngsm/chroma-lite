"""Benchmark helpers for the chroma2 DeviceQueue prototype."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import DeviceQueue, device_count


def _ensure_extension() -> None:
    if not isinstance(DeviceQueue, type):
        raise ImportError(
            "chroma2.runtime.queue DeviceQueue extension is unavailable; build the pybind11 module first"
        )


@dataclass
class ThroughputResult:
    pushes: int
    pops: int
    elapsed_ms: float

    @property
    def photons_per_second(self) -> float:
        if self.elapsed_ms == 0:
            return float("inf")
        return (self.pushes + self.pops) / (self.elapsed_ms * 1e-3)


def measure_queue_round_trip(batch_size: int, iterations: int, capacity: int = 1 << 14) -> ThroughputResult:
    _ensure_extension()
    if device_count() == 0:
        raise RuntimeError("No CUDA devices available for queue benchmark")

    queue = DeviceQueue(capacity=capacity)
    payload = np.arange(batch_size, dtype=np.uint32)
    push_total = batch_size * iterations
    pop_total = push_total

    start = time.perf_counter()
    for _ in range(iterations):
        queue.push(payload)
        queue.pop(batch_size)
    queue.drain()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    return ThroughputResult(pushes=push_total, pops=pop_total, elapsed_ms=elapsed_ms)


__all__ = ["measure_queue_round_trip", "ThroughputResult"]
