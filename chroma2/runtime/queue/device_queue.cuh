#pragma once

#include <cuda.h>

struct DeviceQueue {
    unsigned int *buffer;   // ring buffer storage
    unsigned int *head;     // pointer to next item to pop
    unsigned int *tail;     // pointer to next slot to push
    unsigned int capacity;  // total slots (power of two)
    unsigned int mask;      // capacity - 1 for modulo
};

__device__ __forceinline__ unsigned int dq_count(const DeviceQueue &q) {
    unsigned int head = atomicAdd(q.head, 0u);
    unsigned int tail = atomicAdd(q.tail, 0u);
    return tail - head;
}

__device__ __forceinline__ bool dq_try_push(const DeviceQueue &q, unsigned int value, int max_spin = 64) {
    unsigned int capacity = q.capacity;
    for (int attempt = 0; attempt < max_spin; ++attempt) {
        unsigned int head = atomicAdd(q.head, 0u);
        unsigned int tail = atomicAdd(q.tail, 0u);
        if (tail - head >= capacity) {
            return false; // queue full
        }
        if (atomicCAS(q.tail, tail, tail + 1u) == tail) {
            q.buffer[tail & q.mask] = value;
            __threadfence();
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool dq_try_pop(const DeviceQueue &q, unsigned int *value_out, int max_spin = 64) {
    for (int attempt = 0; attempt < max_spin; ++attempt) {
        unsigned int head = atomicAdd(q.head, 0u);
        unsigned int tail = atomicAdd(q.tail, 0u);
        if (tail <= head) {
            return false; // queue empty
        }
        if (atomicCAS(q.head, head, head + 1u) == head) {
            __threadfence();
            *value_out = q.buffer[head & q.mask];
            return true;
        }
    }
    return false;
}
