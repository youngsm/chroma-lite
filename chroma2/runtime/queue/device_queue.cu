#include "device_queue.cuh"

extern "C" __global__ void dq_reset(DeviceQueue queue) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *queue.head = 0u;
        *queue.tail = 0u;
    }
}

extern "C" __global__ void dq_push_kernel(DeviceQueue queue,
                                           const unsigned int *values,
                                           unsigned int count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = tid; i < count; i += stride) {
        unsigned int value = values[i];
        while (!dq_try_push(queue, value)) {
            __nanosleep(64);
        }
    }
}

extern "C" __global__ void dq_pop_kernel(DeviceQueue queue,
                                          unsigned int *out_values,
                                          unsigned int count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = tid; i < count; i += stride) {
        unsigned int value;
        while (!dq_try_pop(queue, &value)) {
            __nanosleep(64);
        }
        out_values[i] = value;
    }
}

extern "C" __global__ void dq_drain(DeviceQueue queue, unsigned int *counter) {
    unsigned int drained = 0u;
    unsigned int value;
    while (dq_try_pop(queue, &value)) {
        ++drained;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *counter = drained;
    }
}
