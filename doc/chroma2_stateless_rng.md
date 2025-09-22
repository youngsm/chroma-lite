# Chroma2 Stateless RNG & Isotropic Source Design

## Goals
- Eliminate per-photon RNG state to reduce device memory pressure for 50–200M photon workloads.
- Allow isotropically generated photons to skip host-side direction uploads by reconstructing directions on the GPU.
- Preserve determinism and reproducibility across replays, queue reordering, and multi-device execution.

## Isotropic Source Flag
- Extend the photon metadata word with an `ISOTROPIC_INIT` bit. When set, the host omits the explicit direction vector.
- Each isotropic photon carries a 32-bit `direction_seed` generated from `(event_id, photon_index)`.
- On ingestion, persistent kernels compute the initial direction with `uniform_sphere(prng(seed, 0))` and cache the local result in registers while processing the chunk.
- For photons that scatter and requeue, the kernel increments the stateless RNG counter (see below) to derive new random directions without touching global memory.
- Polarization: by default an isotropic source generates random polarization orthogonal to the direction. If the source is unpolarized, the kernel sets polarization to zero or samples uniformly; if a user supplies polarization explicitly, the flag is not set.

## Stateless RNG Strategy
- Adopt a counter-based generator such as **Philox4x32-10** or **PCG-XSL-RR 128/64**. Both provide high statistical quality and efficient GPU implementations.
- RNG inputs:
  - **Key**: `(event_id << 32) | photon_index` ensures different photons have independent sequences even when events overlap.
  - **Counter**: a 64-bit value formed by concatenating the photon micro-step (number of random draws since initialization) with the lane-local sample index. Each RNG invocation increments this counter.
- RNG outputs: 128 bits per call (`uint4`). Kernels unpack to floats using `__uint2float_rn` and multiply by `2^-32` to obtain `U(0,1)` samples.
- Determinism: as long as the kernel increments the counter in a consistent order (e.g., per-physics step), replays reproduce identical behaviour regardless of queue scheduling.

## CUDA Implementation Outline
1. **Header (`chroma2/runtime/rng/stateless_rng.cuh`)**
   - `struct StatelessRngKey { uint32_t hi, lo; };`
   - `__device__ uint4 philox4x32(StatelessRngKey key, uint64_t counter);`
   - Helper to convert to `float4 uniform = rng_next(key, &counter);`
2. **Host Utilities (`chroma2/runtime/rng/__init__.py`)**
   - Functions to compute seeds/counters for ingestion, plus CPU reference implementation for testing.
   - Conversion helpers to pack/unpack direction seeds in staging buffers.
3. **Kernel Integration**
   - Persistent propagation kernel obtains `key` from photon metadata.
   - Before each stochastic operation, kernel calls `rng_uniform4(key, counter++)` to get four uniform variates.
   - For isotropic initialization the kernel uses the same API with counter `0`.
4. **Memory Layout Changes**
   - Photon SoA pages store `direction_seed` and `draw_counter` fields (both 32-bit).
   - Host staging buffers follow the optimized 64-byte layout documented in `doc/chroma2_memory_budget.md`.

## Testing Plan
- **Unit tests**: CPU reference generator matches CUDA implementation bit-for-bit for a battery of keys/counters. (Add under `test/test_stateless_rng.py`.)
- **Statistical validation**: Run PractRand / TestU01 on sequences derived from a selection of keys to ensure no detectable bias.
- **Integration**: Modify the queue benchmark to include synthetic isotropic photons and confirm throughput/parity with explicit-direction mode.

## Migration Strategy
1. Introduce the stateless RNG utility alongside the existing XORWOW usage and gate with a feature flag (`enable_stateless_rng`).
2. Teach the photon ingestion path to detect isotropic batches and pack the compact format.
3. Update propagation kernels to branch on the feature flag, ensuring legacy paths remain available.
4. Measure memory savings and throughput; remove legacy RNG state once confidence is high.

## Open Questions
- Do secondary photons inherit isotropic status, or should they fall back to explicit directions if they originate from anisotropic surfaces?
- How should we encode polarization for isotropic but polarized sources without expanding the optimized layout?
- Can we share RNG counters across warps to reduce metadata further, or does that complicate reproducibility?
