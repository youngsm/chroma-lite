#pragma once

#include <cuda_runtime.h>

namespace chroma2 {
namespace queue {

struct DeviceQueue {
    unsigned int *buffer;
    unsigned int *head;
    unsigned int *tail;
    unsigned int capacity;
    unsigned int mask;
};

#if defined(__CUDACC__)

__device__ inline void dq_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(64);
#else
    ;
#endif
}

__device__ inline bool dq_try_push(const DeviceQueue &q, unsigned int value, int max_spin = 64) {
    for (int attempt = 0; attempt < max_spin; ++attempt) {
        unsigned int head = atomicAdd(q.head, 0u);
        unsigned int tail = atomicAdd(q.tail, 0u);
        if (tail - head >= q.capacity) {
            return false;
        }
        if (atomicCAS(q.tail, tail, tail + 1u) == tail) {
            q.buffer[tail & q.mask] = value;
            __threadfence();
            return true;
        }
        dq_pause();
    }
    return false;
}

__device__ inline bool dq_try_pop(const DeviceQueue &q, unsigned int *value_out, int max_spin = 64) {
    for (int attempt = 0; attempt < max_spin; ++attempt) {
        unsigned int head = atomicAdd(q.head, 0u);
        unsigned int tail = atomicAdd(q.tail, 0u);
        if (tail <= head) {
            return false;
        }
        if (atomicCAS(q.head, head, head + 1u) == head) {
            __threadfence();
            *value_out = q.buffer[head & q.mask];
            return true;
        }
        dq_pause();
    }
    return false;
}

#endif  // defined(__CUDACC__)

}  // namespace queue
}  // namespace chroma2

extern "C" cudaError_t dq_launch_reset(chroma2::queue::DeviceQueue queue, cudaStream_t stream);
extern "C" cudaError_t dq_launch_push(chroma2::queue::DeviceQueue queue,
                                       const unsigned int *values,
                                       unsigned int count,
                                       dim3 grid,
                                       dim3 block,
                                       cudaStream_t stream);
extern "C" cudaError_t dq_launch_pop(chroma2::queue::DeviceQueue queue,
                                      unsigned int *out_values,
                                      unsigned int count,
                                      dim3 grid,
                                      dim3 block,
                                      cudaStream_t stream);
extern "C" cudaError_t dq_launch_persistent(chroma2::queue::DeviceQueue active_queue,
                                               chroma2::queue::DeviceQueue spawn_queue,
                                               chroma2::queue::DeviceQueue finished_queue,
                                               unsigned int max_iterations,
                                               unsigned int idle_threshold,
                                               unsigned int spawn_interval,
                                               unsigned int payload_increment,
                                               dim3 grid,
                                               dim3 block,
                                               cudaStream_t stream);
extern "C" cudaError_t dq_launch_drain(chroma2::queue::DeviceQueue queue,
                                        unsigned int *counter,
                                        cudaStream_t stream);
