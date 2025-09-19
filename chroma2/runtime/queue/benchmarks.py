"""Benchmark helpers for the chroma2 DeviceQueue prototype."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pycuda.driver as cuda

from .prototype import DeviceQueuePrototype


@contextlib.contextmanager
def cuda_events() -> Iterator[tuple[cuda.Event, cuda.Event]]:
    start = cuda.Event()
    stop = cuda.Event()
    try:
        yield start, stop
    finally:
        del start
        del stop


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
    queue = DeviceQueuePrototype(capacity=capacity)
    payload = np.arange(batch_size, dtype=np.uint32)
    push_total = batch_size * iterations
    pop_total = push_total
    with cuda_events() as (start, stop):
        start.record()
        for _ in range(iterations):
            queue.push(payload)
            queue.pop(batch_size)
        stop.record()
        stop.synchronize()
    elapsed_ms = start.time_till(stop)
    return ThroughputResult(pushes=push_total, pops=pop_total, elapsed_ms=elapsed_ms)


__all__ = ["measure_queue_round_trip", "ThroughputResult"]
