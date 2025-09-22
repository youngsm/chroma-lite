#include "device_queue.cuh"

namespace chroma2 {
namespace queue {

__global__ void dq_reset_kernel(DeviceQueue queue) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *queue.head = 0u;
        *queue.tail = 0u;
    }
}

__global__ void dq_push_kernel(DeviceQueue queue,
                               const unsigned int *values,
                               unsigned int count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = tid; i < count; i += stride) {
        unsigned int value = values[i];
        while (!dq_try_push(queue, value)) {
            dq_pause();
        }
    }
}

__global__ void dq_pop_kernel(DeviceQueue queue,
                              unsigned int *out_values,
                              unsigned int count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = tid; i < count; i += stride) {
        unsigned int value;
        while (!dq_try_pop(queue, &value)) {
            dq_pause();
        }
        out_values[i] = value;
    }
}

__global__ void dq_persistent_kernel(DeviceQueue active_queue,
                                      DeviceQueue spawn_queue,
                                      DeviceQueue finished_queue,
                                      unsigned int max_iterations,
                                      unsigned int idle_threshold,
                                      unsigned int spawn_interval,
                                      unsigned int payload_increment) {
    const bool spawn_enabled = (spawn_queue.buffer != nullptr) && (spawn_interval > 0u);
    unsigned int processed = 0u;
    unsigned int idle = 0u;
    unsigned int rng_state = 0x9E3779B9u * (blockIdx.x * blockDim.x + threadIdx.x + 1u);

    while (processed < max_iterations) {
        // TODO(#physics-integration): replace the synthetic payload/secondary logic below
        // with calls into the actual propagation kernel so emission obeys detector physics.
        // This harness only stresses queue plumbing and intentionally uses fake RNG.
        unsigned int value = 0u;
        if (dq_try_pop(active_queue, &value, 512)) {
            idle = 0u;
            ++processed;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            const unsigned int mixed = rng_state + value + processed;
            const bool emit_secondary = spawn_enabled && (spawn_interval > 0u) && ((mixed % spawn_interval) == 0u);
            unsigned int payload = value + payload_increment;
            if (emit_secondary) {
                payload ^= 0x5bd1e995u;
                while (!dq_try_push(spawn_queue, payload, 512)) {
                    dq_pause();
                }
            } else {
                while (!dq_try_push(finished_queue, payload, 512)) {
                    dq_pause();
                }
            }
        } else {
            ++idle;
            if (idle >= idle_threshold) {
                break;
            }
            dq_pause();
        }
    }
}

__global__ void dq_drain_kernel(DeviceQueue queue, unsigned int *counter) {
    unsigned int drained = 0u;
    unsigned int value;
    while (dq_try_pop(queue, &value)) {
        ++drained;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *counter = drained;
    }
}

}  // namespace queue
}  // namespace chroma2

extern "C" cudaError_t dq_launch_reset(chroma2::queue::DeviceQueue queue, cudaStream_t stream) {
    chroma2::queue::dq_reset_kernel<<<1, 1, 0, stream>>>(queue);
    return cudaGetLastError();
}

extern "C" cudaError_t dq_launch_push(chroma2::queue::DeviceQueue queue,
                                       const unsigned int *values,
                                       unsigned int count,
                                       dim3 grid,
                                       dim3 block,
                                       cudaStream_t stream) {
    chroma2::queue::dq_push_kernel<<<grid, block, 0, stream>>>(queue, values, count);
    return cudaGetLastError();
}

extern "C" cudaError_t dq_launch_pop(chroma2::queue::DeviceQueue queue,
                                      unsigned int *out_values,
                                      unsigned int count,
                                      dim3 grid,
                                      dim3 block,
                                      cudaStream_t stream) {
    chroma2::queue::dq_pop_kernel<<<grid, block, 0, stream>>>(queue, out_values, count);
    return cudaGetLastError();
}

extern "C" cudaError_t dq_launch_persistent(chroma2::queue::DeviceQueue active_queue,
                                               chroma2::queue::DeviceQueue spawn_queue,
                                               chroma2::queue::DeviceQueue finished_queue,
                                               unsigned int max_iterations,
                                               unsigned int idle_threshold,
                                               unsigned int spawn_interval,
                                               unsigned int payload_increment,
                                               dim3 grid,
                                               dim3 block,
                                               cudaStream_t stream) {
    chroma2::queue::dq_persistent_kernel<<<grid, block, 0, stream>>>(active_queue, spawn_queue, finished_queue, max_iterations, idle_threshold, spawn_interval, payload_increment);
    return cudaGetLastError();
}

extern "C" cudaError_t dq_launch_drain(chroma2::queue::DeviceQueue queue,
                                        unsigned int *counter,
                                        cudaStream_t stream) {
    chroma2::queue::dq_drain_kernel<<<1, 1, 0, stream>>>(queue, counter);
    return cudaGetLastError();
}