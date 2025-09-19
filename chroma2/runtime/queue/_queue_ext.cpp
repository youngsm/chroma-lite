#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace chroma2 {
namespace queue {

namespace {

#define CUDA_DRV_CHECK(call) chroma2::queue::check_cuda_driver((call), #call, __FILE__, __LINE__)
#define CUDA_RT_CHECK(call) chroma2::queue::check_cuda_runtime((call), #call, __FILE__, __LINE__)
#define NVRTC_CHECK(call) chroma2::queue::check_nvrtc((call), #call, __FILE__, __LINE__)

[[noreturn]] inline void throw_runtime_error(const std::string &prefix, const std::string &msg,
                                             const char *file, int line) {
    std::ostringstream oss;
    oss << prefix << " failed at " << file << ':' << line << " -> " << msg;
    throw std::runtime_error(oss.str());
}

inline void check_cuda_driver(CUresult result, const char *expr, const char *file, int line) {
    if (result != CUDA_SUCCESS) {
        const char *err_name = nullptr;
        const char *err_str = nullptr;
        cuGetErrorName(result, &err_name);
        cuGetErrorString(result, &err_str);
        std::ostringstream oss;
        oss << expr << " : " << (err_name ? err_name : "unknown") << " ("
            << (err_str ? err_str : "no description") << ")";
        throw_runtime_error("CUDA driver call", oss.str(), file, line);
    }
}

inline void check_cuda_runtime(cudaError_t result, const char *expr, const char *file, int line) {
    if (result != cudaSuccess) {
        std::ostringstream oss;
        oss << expr << " : " << cudaGetErrorName(result) << " (" << cudaGetErrorString(result) << ")";
        throw_runtime_error("CUDA runtime call", oss.str(), file, line);
    }
}

inline void check_nvrtc(nvrtcResult result, const char *expr, const char *file, int line) {
    if (result != NVRTC_SUCCESS) {
        std::ostringstream oss;
        oss << expr << " : " << nvrtcGetErrorString(result);
        throw_runtime_error("NVRTC call", oss.str(), file, line);
    }
}

struct DeviceQueue {
    unsigned int *buffer;
    unsigned int *head;
    unsigned int *tail;
    unsigned int capacity;
    unsigned int mask;
};

struct KernelModule {
    CUmodule module;
    CUfunction reset;
    CUfunction push;
    CUfunction pop;
    CUfunction drain;
};

static const char *kQueueKernels = R"(
extern "C" {

struct DeviceQueue {
    unsigned int *buffer;
    unsigned int *head;
    unsigned int *tail;
    unsigned int capacity;
    unsigned int mask;
};

__device__ __forceinline__ bool dq_try_push(const DeviceQueue &q, unsigned int value, int max_spin = 64) {
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
    }
    return false;
}

__device__ __forceinline__ bool dq_try_pop(const DeviceQueue &q, unsigned int *value_out, int max_spin = 64) {
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
    }
    return false;
}

__global__ void dq_reset(DeviceQueue queue) {
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
            __nanosleep(64);
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
            __nanosleep(64);
        }
        out_values[i] = value;
    }
}

__global__ void dq_drain(DeviceQueue queue, unsigned int *counter) {
    unsigned int drained = 0u;
    unsigned int value;
    while (dq_try_pop(queue, &value)) {
        ++drained;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *counter = drained;
    }
}

} // extern "C"
)";

KernelModule compile_kernels() {
    CUDA_RT_CHECK(cudaFree(nullptr)); // ensure context
    CUDA_DRV_CHECK(cuInit(0));

    nvrtcProgram program;
    NVRTC_CHECK(nvrtcCreateProgram(&program, kQueueKernels, "device_queue.cu", 0, nullptr, nullptr));
    std::vector<const char *> options = {
        "--std=c++14",
        "--device-as-default-execution-space",
        "--fmad=false",
        "--gpu-architecture=compute_70"
    };
    nvrtcResult compile_result = nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());

    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
    std::string log(log_size, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));
    if (!log.empty() && log[0] != '\0') {
        py::print("[chroma2.queue] NVRTC log:\n" + log);
    }
    if (compile_result != NVRTC_SUCCESS) {
        NVRTC_CHECK(compile_result);
    }

    size_t ptx_size = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
    std::string ptx(ptx_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&program));

    CUmodule module;
    CUDA_DRV_CHECK(cuModuleLoadData(&module, ptx.c_str()));

    KernelModule kernels{};
    kernels.module = module;
    CUDA_DRV_CHECK(cuModuleGetFunction(&kernels.reset, module, "dq_reset"));
    CUDA_DRV_CHECK(cuModuleGetFunction(&kernels.push, module, "dq_push_kernel"));
    CUDA_DRV_CHECK(cuModuleGetFunction(&kernels.pop, module, "dq_pop_kernel"));
    CUDA_DRV_CHECK(cuModuleGetFunction(&kernels.drain, module, "dq_drain"));
    return kernels;
}

KernelModule &kernel_module() {
    static KernelModule module;
    static std::once_flag flag;
    std::call_once(flag, []() { module = compile_kernels(); });
    return module;
}

class DeviceQueueWrapper {
  public:
    DeviceQueueWrapper(std::size_t capacity, int block_dim, int grid_dim)
        : capacity_(static_cast<unsigned int>(capacity)),
          mask_(static_cast<unsigned int>(capacity - 1)),
          block_dim_(block_dim),
          grid_dim_(grid_dim) {
        if ((capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("capacity must be a power of two");
        }
        if (capacity == 0 || capacity > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::invalid_argument("capacity must be within 1..2^32-1");
        }
        if (block_dim <= 0 || grid_dim <= 0) {
            throw std::invalid_argument("block_dim and grid_dim must be positive");
        }
        CUDA_RT_CHECK(cudaMalloc(&buffer_, capacity * sizeof(unsigned int)));
        CUDA_RT_CHECK(cudaMalloc(&head_, sizeof(unsigned int)));
        CUDA_RT_CHECK(cudaMalloc(&tail_, sizeof(unsigned int)));
        reset();
    }

    ~DeviceQueueWrapper() {
        if (buffer_) {
            cudaFree(buffer_);
        }
        if (head_) {
            cudaFree(head_);
        }
        if (tail_) {
            cudaFree(tail_);
        }
        if (scratch_) {
            cudaFree(scratch_);
        }
    }

    void reset() {
        DeviceQueue queue = device_queue();
        void *args[] = {&queue};
        auto &kernels = kernel_module();
        CUDA_DRV_CHECK(cuLaunchKernel(kernels.reset,
                                      1, 1, 1,
                                      1, 1, 1,
                                      0, nullptr,
                                      args, nullptr));
        CUDA_RT_CHECK(cudaDeviceSynchronize());
    }

    std::size_t capacity() const { return capacity_; }

    int block_dim() const { return block_dim_; }
    int grid_dim() const { return grid_dim_; }

    std::pair<int, int> launch_config() const { return {block_dim_, grid_dim_}; }

    void set_launch_config(std::pair<int, int> config) {
        auto [block_dim, grid_dim] = config;
        if (block_dim <= 0 || grid_dim <= 0) {
            throw std::invalid_argument("block_dim and grid_dim must be positive");
        }
        block_dim_ = block_dim;
        grid_dim_ = grid_dim;
    }

    void push(py::array_t<unsigned int, py::array::c_style | py::array::forcecast> values) {
        auto buf = values.request();
        const auto count = static_cast<std::size_t>(buf.size);
        if (count == 0) {
            return;
        }
        if (count > capacity_) {
            throw std::invalid_argument("push batch larger than queue capacity");
        }
        ensure_scratch(count);
        CUDA_RT_CHECK(cudaMemcpy(scratch_, buf.ptr, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        DeviceQueue queue = device_queue();
        unsigned int count32 = static_cast<unsigned int>(count);
        void *args[] = {&queue, &scratch_, &count32};
        auto &kernels = kernel_module();
        CUDA_DRV_CHECK(cuLaunchKernel(kernels.push,
                                      grid_dim_, 1, 1,
                                      block_dim_, 1, 1,
                                      0, nullptr,
                                      args, nullptr));
        CUDA_RT_CHECK(cudaDeviceSynchronize());
    }

    py::array_t<unsigned int> pop(std::size_t count) {
        if (count == 0) {
            return py::array_t<unsigned int>(0);
        }
        if (count > capacity_) {
            throw std::invalid_argument("pop count larger than queue capacity");
        }
        ensure_scratch(count);
        DeviceQueue queue = device_queue();
        unsigned int count32 = static_cast<unsigned int>(count);
        void *args[] = {&queue, &scratch_, &count32};
        auto &kernels = kernel_module();
        CUDA_DRV_CHECK(cuLaunchKernel(kernels.pop,
                                      grid_dim_, 1, 1,
                                      block_dim_, 1, 1,
                                      0, nullptr,
                                      args, nullptr));
        CUDA_RT_CHECK(cudaDeviceSynchronize());
        py::array_t<unsigned int> host(count);
        CUDA_RT_CHECK(cudaMemcpy(host.mutable_data(), scratch_, count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        return host;
    }

    unsigned int drain() {
        unsigned int *device_counter = nullptr;
        CUDA_RT_CHECK(cudaMalloc(&device_counter, sizeof(unsigned int)));
        CUDA_RT_CHECK(cudaMemset(device_counter, 0, sizeof(unsigned int)));
        DeviceQueue queue = device_queue();
        void *args[] = {&queue, &device_counter};
        auto &kernels = kernel_module();
        CUDA_DRV_CHECK(cuLaunchKernel(kernels.drain,
                                      1, 1, 1,
                                      1, 1, 1,
                                      0, nullptr,
                                      args, nullptr));
        CUDA_RT_CHECK(cudaDeviceSynchronize());
        unsigned int host_counter = 0;
        CUDA_RT_CHECK(cudaMemcpy(&host_counter, device_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        cudaFree(device_counter);
        return host_counter;
    }

  private:
    DeviceQueue device_queue() const {
        DeviceQueue queue{};
        queue.buffer = buffer_;
        queue.head = head_;
        queue.tail = tail_;
        queue.capacity = capacity_;
        queue.mask = mask_;
        return queue;
    }

    void ensure_scratch(std::size_t count) {
        if (count > scratch_capacity_) {
            if (scratch_) {
                cudaFree(scratch_);
                scratch_ = nullptr;
            }
            CUDA_RT_CHECK(cudaMalloc(&scratch_, count * sizeof(unsigned int)));
            scratch_capacity_ = count;
        }
    }

    unsigned int capacity_;
    unsigned int mask_;
    int block_dim_;
    int grid_dim_;
    unsigned int *buffer_ = nullptr;
    unsigned int *head_ = nullptr;
    unsigned int *tail_ = nullptr;
    unsigned int *scratch_ = nullptr;
    std::size_t scratch_capacity_ = 0;
};

}  // namespace

}  // namespace queue
}  // namespace chroma2

PYBIND11_MODULE(_queue_ext, m) {

}  // namespace

}  // namespace queue
}  // namespace chroma2

PYBIND11_MODULE(_queue_ext, m) {
    m.doc() = "Chroma2 device queue prototype with NVRTC-compiled kernels";

    m.def("device_count", []() {
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) {
            return 0;
        }
        return count;
    }, "Return the number of CUDA devices visible to the runtime.");

    py::class_<chroma2::queue::DeviceQueueWrapper>(m, "DeviceQueue")
        .def(py::init<std::size_t, int, int>(),
             py::arg("capacity"),
             py::arg("block_dim") = 128,
             py::arg("grid_dim") = 4)
        .def("reset", &chroma2::queue::DeviceQueueWrapper::reset)
        .def("push", &chroma2::queue::DeviceQueueWrapper::push, py::arg("values"))
        .def("pop", &chroma2::queue::DeviceQueueWrapper::pop, py::arg("count"))
        .def("drain", &chroma2::queue::DeviceQueueWrapper::drain)
        .def_property("launch_config",
                      &chroma2::queue::DeviceQueueWrapper::launch_config,
                      &chroma2::queue::DeviceQueueWrapper::set_launch_config,
                      "Get or set the (block_dim, grid_dim) launch configuration.")
        .def_property_readonly("capacity", &chroma2::queue::DeviceQueueWrapper::capacity);
}
