"""Expose queue prototype depending on whether the native extension is available."""
from __future__ import annotations

try:
    from ._queue_ext import DeviceQueue as DeviceQueue  # type: ignore[assignment]
    from ._queue_ext import device_count
    from ._queue_ext import run_persistent_kernel
except ImportError as exc:  # pragma: no cover - exercised when CUDA toolchain missing
    _queue_import_error = exc

    class DeviceQueue:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "chroma2.runtime.queue requires the CUDA toolchain and pybind11 extension; "
                "see doc/chroma2_queue_prototype.md for build instructions"
            ) from _queue_import_error

    def device_count() -> int:  # pragma: no cover - fallback path
        return 0

    def run_persistent_kernel(*args, **kwargs):  # pragma: no cover - fallback path
        raise ImportError(
            "chroma2.runtime.queue requires the CUDA toolchain and pybind11 extension; "
            "see doc/chroma2_queue_prototype.md for build instructions"
        ) from _queue_import_error

__all__ = ["DeviceQueue", "device_count", "run_persistent_kernel"]
