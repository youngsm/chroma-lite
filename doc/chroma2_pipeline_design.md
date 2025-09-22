# Chroma2 Propagation Pipeline Design

## Goals
- Preserve legacy photon physics while reorganizing the runtime around persistent GPU execution.
- Achieve near-continuous device occupancy by streaming photon workloads and overlapping host interaction.
- Provide modular components so future detector geometries or materials can plug into the same pipeline without rewriting kernels.

## High-Level Architecture
```
Python API -> Event Builder -> Pipeline Orchestrator -> Runtime Services
                               |                    |
                               v                    v
                       GPU Work Queues <--> Persistent Kernels
```
- **API Layer** supplies Python bindings and validation, normalizing inputs into `PhotonBatch` descriptors that alias device buffers.
- **Pipeline Orchestrator** coordinates queue populations, launches graph instances, and ties together per-stage kernels.
- **Runtime Services** own device memory pools, CUDA streams, and profiling hooks shared across modules.
- **Persistent Kernels** fetch photon work units, execute physics steps, and report completion without tearing down the grid.

## Module Breakdown

### Pipeline Orchestrator (`chroma2/pipeline`)
- Maintains a DAG of stages: `initialize`, `propagate`, `postprocess`, `compact`, `emit_secondary`.
- Uses CUDA Graphs to instantiate the steady-state flow; graph nodes reference runtime-managed streams and buffers.
- Exposes async entrypoints: `submit_event(EventDescriptor)` returns a future resolved when photon propagation completes.
- Integrates host-to-device (H2D) staging: for each submitted event, a producer fiber pushes photon state into ingestion queues while consumer kernels drain them.

### Runtime Services (`chroma2/runtime`)
- Provides memory arenas: `DeviceArena` for long-lived assets (geometry, materials, RNG seeds) and `StagingArena` for transient photon buffers.
- Manages stream pools: dedicated streams for copies, initialization kernels, propagation kernels, and reduction/postprocessing.
- Hosts metrics collectors (CUDA events, Nsight ranges) and exposes an instrumentation bus consumed by logging or dashboards.
- Supplies synchronization primitives: CUDA fences and lightweight host futures to coordinate overlapping events.
- Publishes stateless RNG utilities so kernels can derive reproducible random sequences without per-photon state.

### Work Queues & Scheduling (`chroma2/runtime/queue`)
- Implements lock-free ring buffers in global memory for `active_photons`, `spawn_queue`, and `finished_photons`.
- Queue descriptors contain head/tail pointers cached per-block in shared memory; cooperative groups synchronize pushes/pops.
- `active_photons` holds structs-of-arrays indices referencing photon state pages; persistent kernel warps claim chunks of e.g. 128 photons.
- `spawn_queue` accumulates secondary photons (reemission, wavelength-shift); compaction kernel periodically moves these into `active_photons` when occupancy drops below a threshold.
- `finished_photons` accumulates indices ready for host consumption; a dedicated stream copies results back using `cudaMemcpyAsync` while computation continues.

### Persistent Propagation Kernel (`chroma2/physics/propagate.cu`)
- Launches once per device with enough blocks to fully utilize SMs; uses a device-side work loop:
  1. Pop photon chunk from `active_photons` queue.
  2. Execute physics stepper (surface interactions, scattering, absorption) up to a fixed micro-iteration budget.
  3. If photon survives, push back into `active_photons`; if it emits offspring, enqueue into `spawn_queue`; if terminated, append to `finished_photons`.
- Employs warp specialization: dedicated warps handle geometry intersection (BVH traversal), others handle material response, reducing divergence.
- Uses CUDA cooperative groups for synchronization during queue updates and shared-memory staging of BVH nodes.

### Data Residency & Layout (`chroma2/runtime/state.py`)
- Photon state stored in SoA pages: positions, directions, wavelengths, times-of-flight, RNG seeds, flags; padded to 128B boundaries for coalescing.
- Isotropic sources mark an `ISOTROPIC_INIT` flag (and optional seeds) so directions can be generated on-device, trimming staging traffic for random emitters.
- Stateless RNG counters live in compact per-photon metadata pages, enabling reconstruction of random draws during requeue without loading global RNG state.
- Geometry acceleration structures (BVH, material tables) stay in device memory across events; host updates use diff uploads when needed.
- Host-visible pinned buffers mirror `finished_photons` segments, enabling overlapped copies and zero-copy inspection for small batches.

### Host Integration (`chroma2/api/session.py`)
- Provides context-managed sessions: `with chroma2.Session(device_id) as session:` loads geometry, seeds RNGs, and warms persistent kernels.
- Users submit `EventDescriptor` objects referencing detector configuration and initial photon distributions; descriptors may carry callbacks for streaming results.
- Session exposes telemetry (per-stage timing, queue depths) to guide tuning and detect backpressure early.

## Execution Flow
1. **Session startup** loads geometry/material tables, allocates arenas, launches persistent kernel, and records CUDA graph handles.
2. **Event submission** pushes photon descriptors to a host-side bounded queue; ingestion fibers pack data into staging buffers.
3. **H2D transfer** streams photon state into device staging buffers; the pipeline orchestrator enqueues initialization kernels in a copy stream.
4. **Propagation** persistent kernel drains `active_photons`, executing micro-iterations until queues empty; compaction kernels rebalance from `spawn_queue`.
5. **Completion** when device queues fall below threshold and async copies finish, futures resolve with pointers to host results.
6. **Tear-down** flush metrics, optionally checkpoint RNG states, keep persistent kernels alive for the next event unless session closes.

## Open Questions
- How to expose deterministic replay modes while allowing out-of-order completion across events?
- Which heuristics control compaction cadence (time-based vs occupancy-based) for different detector geometries?
- What telemetry sampling rate balances insight with minimal device overhead?

## Immediate TODOs
1. Prototype lock-free queue kernels and verify warp-level contention behavior on Ampere hardware.
2. Draft API signatures for `Session`, `EventDescriptor`, and queue descriptors.
3. Build Nsight-based profiling harness to measure baseline occupancy, memory throughput, and queue latencies.
