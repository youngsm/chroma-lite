# OptiX Integration Plan

This checklist captures the work required to swap ChromA-lite's bespoke BVH traversal with OptiX-based ray queries. Tackle the steps in order; each milestone leaves the code runnable so we can validate physics along the way.

## 0. Prerequisites
- [x] Ensure `optix_raycaster` builds in your environment and the shared library is on the `PYTHONPATH` (or installed) for the Python interpreter used by `chroma-lite`.
- [x] Add a thin loader module (e.g. `chroma/gpu/optix.py`) that imports the extension, runs `optixInit()` once, and exposes helpers to create/destroy OptiX contexts.

## 1. Extend the Raycaster API
- [x] Update `optix_test/optix_raycaster/src/raycaster.cpp` so the closest-hit program writes both hit distance and triangle index. Use `optixGetPrimitiveIndex()` and expose the result through the payload.
- [x] Teach the any-hit program to skip triangles whose id matches a `last_hit` value passed in the launch params (mirror `intersect_mesh` semantics).
- [x] Return optional attributes (e.g. geometric normal or instance id) that downstream code might need, keeping the existing CUDA path feature-parity.

## 2. Add Batched Tracing
- [x] Introduce a `trace_many(origins, directions, last_hits, tmin, tmax)` method that launches OptiX with `launch_dim = N` and writes distances + triangle ids to device buffers.
- [x] Extend the Python binding to accept NumPy arrays or CuPy/PyCUDA device allocations and to expose both distance and primitive index arrays.
- [x] Unit-test the batch API against a simple mesh (e.g. the square in `python/simple_example.py`).

## 3. Hook Into `GPUGeometry`
- [x] After `self.vertices` and `self.triangles` are populated (`chroma/gpu/geometry.py:389-454`), instantiate a singleton OptiX raycaster with those arrays.
- [x] Cache the instance on `GPUGeometry` (e.g. `self.optix_raycaster`) so render kernels and the photon propagator reuse the same GAS.
- [x] Add graceful fallbacks when OptiX is unavailable (skip instantiation, drop to legacy BVH).

## 4. Restructure Photon Propagation
- [x] Refactor the propagation loop so the expensive “ray vs. mesh” query happens via the batched OptiX interface.
- [x] Expose GPU buffers for photon origin/direction/last-hit ids to the Python side before propagation begins.
- [x] Consume OptiX results inside the CUDA kernel (e.g. pass distances/triangle ids as kernel arguments rather than calling `intersect_mesh`).

## 5. Migrate Other Call Sites
- [x] Replace `distance_to_mesh` launches (`chroma/cuda/mesh.h:142-164`) with the new OptiX batch query.
- [x] Update rendering kernels (`chroma/cuda/hybrid_render.cu`, `render.cu`) to consume the OptiX data path or call a host-side helper that wraps it.
- [x] Keep the legacy path behind a feature flag until regression testing passes.

## 6. Retire the Legacy BVH
- [ ] Remove uploads of `geometry.bvh.nodes` and the CPU/GPU split logic from `GPUGeometry` once OptiX drives every intersection.
- [ ] Delete `intersect_mesh`, `intersect_node`, and related helpers from `chroma/cuda/mesh.h` after confirming no kernels reference them.
- [ ] Clean up profiling counters that referenced `intersect_mesh`.

## 7. Validation & Enhancements
- [ ] Write regression tests comparing OptiX results with the bespoke BVH on representative meshes.
- [ ] Extend the OptiX pipeline to return barycentric coordinates or normals if needed for downstream shading logic.
- [ ] Document runtime requirements (OptiX SDK location, compatible GPU drivers) in the project README.

Track progress by checking items off; feel free to expand sections with implementation notes or links to PRs as work lands.
