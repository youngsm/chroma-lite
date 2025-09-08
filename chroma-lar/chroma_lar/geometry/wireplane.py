"""
Analytic wire-plane helpers for Chroma-LAr.

This module attaches fast, analytic wire-plane descriptors to a Chroma
Geometry instance so the GPU can use a bespoke wire-cylinder intersection
instead of triangle meshes.

Notes:
- The surface and materials referenced should already be present in geom
  (i.e., appear in geom.unique_surfaces/materials) so GPUGeometry can map
  them to indices. The simplest way is to reuse the same Surface/Material
  objects you assign to other solids in the geometry.
  If that’s not possible, set `surface_index`, `material_inner_index`, and
  `material_outer_index` explicitly in each plane dict.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Any
import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    return (v / n).astype(np.float32)


def _validate_plane(p: Mapping[str, Any]) -> Mapping[str, Any]:
    required = ["origin", "u", "v", "pitch", "radius"]
    for k in required:
        if k not in p:
            raise KeyError(f"wireplane missing required field '{k}'")

    u = _unit(p["u"])  # type: ignore[index]
    v = _unit(p["v"])  # type: ignore[index]
    if abs(np.dot(u, v)) > 1e-3:
        raise ValueError("u and v must be perpendicular unit vectors")

    out = dict(p)
    out["u"], out["v"] = u, v
    # defaults for finite extents; large bounds approximate infinite
    out.setdefault("umin", -1e9)
    out.setdefault("umax", +1e9)
    out.setdefault("vmin", -1e9)
    out.setdefault("vmax", +1e9)
    out.setdefault("v0", 0.0)
    out.setdefault("color", 0x33FFFFFF)
    # Allow either *_index ints or object references
    return out


def attach_wireplanes(geom, planes: Iterable[Mapping[str, Any]]):
    """Attach analytic wire-plane descriptors to `geom`.

    - `geom`: chroma.geometry.Geometry instance.
    - `planes`: iterable of dicts with keys described in the module docstring.
    """
    validated = [_validate_plane(p) for p in planes]
    # Attach attribute dynamically; GPUGeometry checks for it
    geom.wireplanes = validated
    return geom

