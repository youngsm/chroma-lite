"""Optional OptiX helpers for ChromA GPU code.

This module centralises all direct imports of the ``optix_raycaster`` extension so
callers can feature-detect OptiX support in one place.  Downstream users should
invoke :func:`ensure_initialized` before constructing any OptiX-backed objects and
handle :class:`OptixUnavailableError` when the dependency is missing.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OptixUnavailableError(RuntimeError):
    """Raised when OptiX support is not present or fails to initialise."""


try:
    import optix_raycaster as _optix_module
except Exception:  # pragma: no cover - handled via OptixUnavailableError
    _optix_module = None


_INITIALISED = False


def is_available() -> bool:
    """Return ``True`` when the OptiX extension can be imported and initialised."""

    global _INITIALISED

    if _optix_module is None:
        return False
    if _INITIALISED:
        return True
    try:
        _optix_module.initialize()
    except Exception:  # pragma: no cover - relies on GPU runtime
        return False
    _INITIALISED = True
    return True


def ensure_initialized() -> None:
    """Initialise the CUDA and OptiX runtime once.

    Raises:
        OptixUnavailableError: When the extension cannot be loaded or fails to
            initialise the driver/runtime.
    """

    global _INITIALISED

    if _optix_module is None:
        raise OptixUnavailableError(
            "optix_raycaster extension not available; ensure it is installed and on PYTHONPATH."
        )

    if not _INITIALISED:
        try:
            _optix_module.initialize()
        except Exception as exc:  # pragma: no cover - requires GPU runtime
            raise OptixUnavailableError(
                "OptiX initialisation failed; check CUDA driver and OptiX installation."
            ) from exc
        _INITIALISED = True
        logger.debug("OptiX runtime initialised")


def raycaster_class() -> type:
    """Return the underlying OptiX raycaster class after initialisation."""

    ensure_initialized()
    assert _optix_module is not None  # for type-checkers
    return _optix_module.Raycaster


def create_raycaster(vertices, triangles):
    """Convenience wrapper that instantiates a :class:`Raycaster` after initialisation."""

    return raycaster_class()(vertices, triangles)


__all__ = [
    "OptixUnavailableError",
    "is_available",
    "ensure_initialized",
    "raycaster_class",
    "create_raycaster",
]
