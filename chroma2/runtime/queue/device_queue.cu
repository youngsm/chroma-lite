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

extern "C" cudaError_t dq_launch_drain(chroma2::queue::DeviceQueue queue,
                                        unsigned int *counter,
                                        cudaStream_t stream) {
    chroma2::queue::dq_drain_kernel<<<1, 1, 0, stream>>>(queue, counter);
    return cudaGetLastError();
}
