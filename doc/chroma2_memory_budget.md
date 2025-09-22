# Chroma2 Photon Memory Budget

## Scope & Assumptions
- Target workloads simulate 50–200 million photons concurrently on a single GPU.
- Photon state is stored in a structure-of-arrays layout with 16-byte aligned vector fields for coalesced access unless otherwise noted.
- Legacy baseline mirrors Chroma today: each photon carries an explicit direction vector and a `curandStateXORWOW` RNG payload.
- Optimized variants introduce an isotropic-source flag plus stateless counter-based RNG to reduce per-photon footprint.
- Active, spawn, and finished work queues are dimensioned to powers of two (≥ 2^28 entries) to avoid wrap-around while covering the 200M-photon case.
- Double-buffered staging arenas keep one ingestion chunk resident on the host while another is in flight to the device.

## Baseline Device-Resident Photon State
| Field (SoA array) | Bytes / photon | 50M photons (GiB) | 200M photons (GiB) | Notes |
| --- | ---: | ---: | ---: | --- |
| Position (`float4`) | 16 | 0.75 | 2.98 | xyz padded to 16B for alignment |
| Direction (`float4`) | 16 | 0.75 | 2.98 | normalized ray direction |
| Polarization (`float4`) | 16 | 0.75 | 2.98 | retains parity with Chroma photons |
| Wavelength (`float32`) | 4 | 0.19 | 0.75 | nanometres |
| Time-of-flight (`float32`) | 4 | 0.19 | 0.75 | nanoseconds |
| Weight (`float32`) | 4 | 0.19 | 0.75 | for weighted transport |
| History / status (`uint32`) | 4 | 0.19 | 0.75 | process flags |
| Last hit / surface id (`int32`) | 4 | 0.19 | 0.75 | triangle or analytic surface |
| Event / page index (`uint32`) | 4 | 0.19 | 0.75 | ties back to host descriptors |
| RNG state (`curandStateXORWOW`) | 48 | 2.23 | 8.94 | dominates footprint |
| **Baseline subtotal** | **120** | **5.59** | **22.35** | Reference configuration |

> GiB values use powers-of-two (1 GiB = 1,073,741,824 bytes) and are rounded to two decimals.

## Optimized Photon State (Isotropic Flag + Stateless RNG)
| Field (SoA array) | Bytes / photon | 50M photons (GiB) | 200M photons (GiB) | Notes |
| --- | ---: | ---: | ---: | --- |
| Position (`float4`) | 16 | 0.75 | 2.98 | Shared with baseline |
| Polarization (`float4`) | 16 | 0.75 | 2.98 | Optional if source is unpolarized |
| Wavelength (`float32`) | 4 | 0.19 | 0.75 |  |
| Time-of-flight (`float32`) | 4 | 0.19 | 0.75 |  |
| Weight (`float32`) | 4 | 0.19 | 0.75 |  |
| History / flags (`uint32`) | 4 | 0.19 | 0.75 | Includes isotropic bit |
| Last hit / surface id (`int32`) | 4 | 0.19 | 0.75 |  |
| Event / page index (`uint32`) | 4 | 0.19 | 0.75 |  |
| Direction seed (`uint32`) | 4 | 0.19 | 0.75 | Counter used to reconstruct isotropic direction |
| Stateless RNG counter (`uint32`) | 4 | 0.19 | 0.75 | Tracks draws per photon chunk |
| **Optimized subtotal** | **64** | **2.98** | **11.93** | 46.7% reduction vs baseline |

- Photons that require explicit anisotropic directions can still supply `float4` vectors; they simply omit the isotropic flag and skip the 4-byte seed, keeping compatibility.
- Stateless RNG uses a Philox/PCG-style hash fed by `(event_id, photon_index, micro_step)` tuples. Only a 32-bit counter per photon is retained, and hot kernels can keep that counter in registers while processing a chunk.

## Device Work Queues & Scratch
| Component | Capacity assumption | Bytes | GiB | Notes |
| --- | ---: | ---: | ---: | --- |
| Active photon queue | 2^28 entries × 4 B | 1,073,741,824 | 1.00 | Headroom beyond 200M to absorb bursty fills |
| Spawn queue | 2^28 entries × 4 B | 1,073,741,824 | 1.00 | Holds secondaries prior to compaction |
| Finished queue | 2^27 entries × 4 B | 536,870,912 | 0.50 | Sized for ~50% headroom; scale if host drains slower |
| Queue metadata & counters | — | ~4,194,304 | 0.00 | Heads/tails, cached block state |
| Photon staging scratch (device) | 8M photons × 64 B | 536,870,912 | 0.50 | Matches optimized per-photon payload |
| **Total queue & scratch** | — | **3,224,548,376** | **3.00** | Shrinks by ~0.5 GiB with smaller scratch |

## Host-Pinned Staging
| Component | Capacity assumption | Bytes | GiB | Notes |
| --- | ---: | ---: | ---: | --- |
| Ingestion buffer A | 8M photons × 64 B | 536,870,912 | 0.50 | Pinned for async H2D |
| Ingestion buffer B | 8M photons × 64 B | 536,870,912 | 0.50 | Double buffer to overlap copies |
| Result readback ring | 4M photons × 48 B | 201,326,592 | 0.19 | Finished photons store compact payload |
| **Total host-pinned** | — | **1,275,068,416** | **1.19** | Almost half the baseline requirement |

## Example Budgets
| Photon count | Baseline device mem (GiB) | Optimized device mem (GiB) | Notes |
| --- | ---: | ---: | --- |
| 50M | 11.09 | 6.53 | 50M isotropic photons with stateless RNG |
| 200M | 28.85 | 19.89 | Fits comfortably on 24–32 GiB assuming geometry ≤3 GiB |

## Implications & Recommendations
- **Isotropic flag**: Pack an `ISOTROPIC_INIT` bit into the history word and store a 32-bit direction seed. During ingestion the host leaves the direction array empty; the persistent kernel reconstructs directions with `uniform_sphere(counter_based_rng(seed, step))`.
- **Stateless RNG**: Adopt a counter-based generator (Philox4x32 or PCG-XSL-RR). Inputs combine `(event_id << 32) | photon_index` as the key and a per-photon draw counter. This eliminates the 48-byte XORWOW payload and reduces global memory traffic.
- **Mixed workloads**: Maintain dual layouts so anisotropic or polarized sources can still upload explicit direction vectors; staging buffers should detect homogeneous isotropic batches and switch to the compact format on the fly.
- **Telemetry**: Track the fraction of isotropic photons to report real memory savings and guide future queue sizing.
- **Roadmap impact**: Incorporate the stateless RNG utility into runtime services so both queue benchmarks and future pipeline kernels can share the implementation.
