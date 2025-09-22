#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include "device_queue.cuh"

namespace py = pybind11;

namespace chroma2 {
namespace queue {

#define CUDA_RT_CHECK(call) chroma2::queue::check_cuda_runtime((call), #call, __FILE__, __LINE__)

inline void check_cuda_runtime(cudaError_t result, const char *expr, const char *file, int line) {
    if (result != cudaSuccess) {
        std::string msg = std::string(expr) + " : " + cudaGetErrorName(result) + " (" + cudaGetErrorString(result) + ")";
        throw std::runtime_error(std::string("CUDA runtime call failed at ") + file + ":" + std::to_string(line) + " -> " + msg);
    }
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
        CUDA_RT_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_RT_CHECK(cudaMalloc(&buffer_, capacity * sizeof(unsigned int)));
        CUDA_RT_CHECK(cudaMalloc(&head_, sizeof(unsigned int)));
        CUDA_RT_CHECK(cudaMalloc(&tail_, sizeof(unsigned int)));
        reset();
    }

    ~DeviceQueueWrapper() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
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
        CUDA_RT_CHECK(dq_launch_reset(device_queue(), stream_));
        CUDA_RT_CHECK(cudaStreamSynchronize(stream_));
    }

    std::size_t capacity() const { return capacity_; }

    std::pair<int, int> launch_config() const { return {block_dim_, grid_dim_}; }

    queue::DeviceQueue raw_queue() const { return device_queue(); }

    cudaStream_t stream() const { return stream_; }

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
        CUDA_RT_CHECK(cudaMemcpyAsync(scratch_, buf.ptr, count * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_));
        unsigned int count32 = static_cast<unsigned int>(count);
        CUDA_RT_CHECK(dq_launch_push(device_queue(), scratch_, count32,
                                     dim3(grid_dim_, 1, 1), dim3(block_dim_, 1, 1), stream_));
        CUDA_RT_CHECK(cudaStreamSynchronize(stream_));
    }

    py::array_t<unsigned int> pop(std::size_t count) {
        if (count == 0) {
            return py::array_t<unsigned int>(0);
        }
        if (count > capacity_) {
            throw std::invalid_argument("pop count larger than queue capacity");
        }
        ensure_scratch(count);
        unsigned int count32 = static_cast<unsigned int>(count);
        CUDA_RT_CHECK(dq_launch_pop(device_queue(), scratch_, count32,
                                    dim3(grid_dim_, 1, 1), dim3(block_dim_, 1, 1), stream_));
        py::array_t<unsigned int> host(count);
        CUDA_RT_CHECK(cudaMemcpyAsync(host.mutable_data(), scratch_, count * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream_));
        CUDA_RT_CHECK(cudaStreamSynchronize(stream_));
        return host;
    }

    unsigned int drain() {
        unsigned int *device_counter = nullptr;
        CUDA_RT_CHECK(cudaMalloc(&device_counter, sizeof(unsigned int)));
        CUDA_RT_CHECK(cudaMemsetAsync(device_counter, 0, sizeof(unsigned int), stream_));
        CUDA_RT_CHECK(dq_launch_drain(device_queue(), device_counter, stream_));
        unsigned int host_counter = 0;
        CUDA_RT_CHECK(cudaMemcpyAsync(&host_counter, device_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream_));
        CUDA_RT_CHECK(cudaStreamSynchronize(stream_));
        cudaFree(device_counter);
        return host_counter;
    }

  private:
    queue::DeviceQueue device_queue() const {
        queue::DeviceQueue queue{};
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
    cudaStream_t stream_ = nullptr;
    unsigned int *buffer_ = nullptr;
    unsigned int *head_ = nullptr;
    unsigned int *tail_ = nullptr;
    unsigned int *scratch_ = nullptr;
    std::size_t scratch_capacity_ = 0;
};

}  // namespace queue
}  // namespace chroma2

PYBIND11_MODULE(_queue_ext, m) {
    m.doc() = "Chroma2 device queue prototype with statically compiled CUDA kernels";

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

    m.def("run_persistent_kernel",
          [](chroma2::queue::DeviceQueueWrapper &active,
             chroma2::queue::DeviceQueueWrapper &spawn,
             chroma2::queue::DeviceQueueWrapper &finished,
             unsigned int max_iterations,
             unsigned int idle_threshold,
             unsigned int spawn_interval,
             unsigned int payload_increment,
             int block_dim,
             int grid_dim) {
              if (max_iterations == 0u) {
                  throw std::invalid_argument("max_iterations must be positive");
              }
              if (block_dim <= 0 || grid_dim <= 0) {
                  throw std::invalid_argument("block_dim and grid_dim must be positive");
              }
              auto stream = active.stream();
              CUDA_RT_CHECK(dq_launch_persistent(active.raw_queue(),
                                                spawn.raw_queue(),
                                                finished.raw_queue(),
                                                max_iterations,
                                                idle_threshold,
                                                spawn_interval,
                                                payload_increment,
                                                dim3(grid_dim, 1, 1),
                                                dim3(block_dim, 1, 1),
                                                stream));
              CUDA_RT_CHECK(cudaStreamSynchronize(stream));
          },
          py::arg("active"),
          py::arg("spawn"),
          py::arg("finished"),
          py::arg("max_iterations"),
          py::arg("idle_threshold") = 1024u,
          py::arg("spawn_interval") = 0u,
          py::arg("payload_increment") = 0u,
          py::arg("block_dim") = 128,
          py::arg("grid_dim") = 1,
          "Run a minimal persistent kernel that drains the active queue and routes items to spawn or finished queues.");
}
