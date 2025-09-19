import numpy as np
import pytest

pycuda = pytest.importorskip("pycuda")
import pycuda.driver as cuda  # noqa: E402

from chroma2.runtime.queue.prototype import DeviceQueuePrototype  # noqa: E402


@pytest.mark.skipif(cuda.Device.count() == 0, reason="No CUDA devices available for queue prototype")
def test_device_queue_round_trip_preserves_order():
    queue = DeviceQueuePrototype(capacity=1024, block_dim=128, grid_dim=4)
    payload = np.arange(512, dtype=np.uint32)
    queue.push(payload)
    popped = queue.pop(payload.size)
    assert np.array_equal(popped, payload)


@pytest.mark.skipif(cuda.Device.count() == 0, reason="No CUDA devices available for queue prototype")
def test_device_queue_handles_multiple_batches():
    queue = DeviceQueuePrototype(capacity=2048, block_dim=64, grid_dim=8)
    first = np.arange(600, dtype=np.uint32)
    second = np.arange(400, dtype=np.uint32) + 10000
    queue.push(first)
    queue.push(second)
    out = queue.pop(first.size + second.size)
    expected = np.concatenate([first, second])
    assert np.array_equal(out, expected)
    assert queue.drain() == 0
