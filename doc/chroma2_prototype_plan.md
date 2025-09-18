# Chroma2 Prototype Work Plan

## Objective
Design and validate early prototypes for the GPU work queues and the Python-facing session API to de-risk the chroma2 runtime architecture.

## Prototype 1: GPU Work Queue Kernel Suite
- **Goals**: Implement lock-free ring buffers in CUDA, measure contention, and confirm compatibility with persistent kernels.
- **Tasks**:
  1. Build minimal queue structs (`active_photons`, `spawn_queue`) with device-side push/pop using cooperative groups.
  2. Create synthetic workload kernel that simulates photon lifecycle events to stress-test queue operations.
  3. Integrate unit tests via `pytest` + `pycuda` harness to validate correctness under variable batch sizes.
  4. Profile on Ampere (e.g., A100) with Nsight Compute to capture warp occupancy, atomics throughput, and shared memory pressure.
- **Deliverables**: CUDA source under `chroma2/runtime/queue/`, benchmark script, profiler reports, and pass/fail gating tests.

## Prototype 2: Session & Event API Skeleton
- **Goals**: Expose a Python API that manages runtime initialization, event submission, and async result retrieval.
- **Tasks**:
  1. Draft `Session` and `EventDescriptor` classes with async futures/promises, backed by dummy device memory pools.
  2. Mock persistent kernel interactions using CPU fallbacks so API can be exercised before GPU integration lands.
  3. Provide example notebook or script demonstrating submission of multiple events with overlapping lifetimes.
  4. Collect feedback from existing Chroma users to refine ergonomics and data-model compatibility.
- **Deliverables**: `chroma2/api/session.py`, integration tests, sample usage, design notes capturing feedback.

## Cross-Cutting Considerations
- Establish deterministic logging/tracing hooks to reuse in full pipeline.
- Set up CI jobs (GPU-enabled if available) to run prototype tests and surface performance regressions early.
- Document learnings in `doc/chroma2_pipeline_design.md` revisions before graduating prototypes into production.
