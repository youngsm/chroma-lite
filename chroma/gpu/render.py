import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda.elementwise import ElementwiseKernel

_UPDATE_ORIGINS_KERNEL = ElementwiseKernel(
    "float *orig, const float *dir, const float *dist, int stride, float eps",
    "int ray = i / stride; float d = dist[ray]; if (d >= 0.0f) { orig[i] += dir[i] * (d + eps); }",
    "update_origins_for_optix"
)


class _DeviceArrayHolder(cuda.PointerHolderBase):
    def __init__(self, device_array):
        super().__init__()
        self.device_array = device_array

    def get_pointer(self):
        return int(self.device_array.ptr)


def _device_array_to_gpuarray(device_array):
    iface = device_array.__cuda_array_interface__
    shape = tuple(int(dim) for dim in iface['shape'])
    strides = iface.get('strides')
    dtype = np.dtype(iface['typestr'])

    holder = _DeviceArrayHolder(device_array)
    gpu_arr = ga.GPUArray(shape, dtype, gpudata=holder, strides=strides)
    gpu_arr._device_array_holder = holder
    gpu_arr._device_array_ref = device_array
    return gpu_arr

from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    to_float3

class GPURays(object):
    """The GPURays class holds arrays of ray positions and directions
    on the GPU that are used to render a geometry."""
    def __init__(self, pos, dir, max_alpha_depth=10, nblocks=64):
        self.pos = ga.to_gpu(to_float3(pos))
        self.dir = ga.to_gpu(to_float3(dir))

        self.max_alpha_depth = max_alpha_depth

        self.nblocks = nblocks

        transform_module = get_cu_module('transform.cu', options=cuda_options)
        self.transform_funcs = GPUFuncs(transform_module)

        render_module = get_cu_module('render.cu', options=cuda_options)
        self.render_funcs = GPUFuncs(render_module)

        self.dx = ga.empty(max_alpha_depth*self.pos.size, dtype=np.float32)
        self.color = ga.empty(self.dx.size, dtype=ga.vec.float4)
        self.dxlen = ga.zeros(self.pos.size, dtype=np.uint32)

    def rotate(self, phi, n):
        "Rotate by an angle phi around the axis `n`."
        self.transform_funcs.rotate(np.int32(self.pos.size), self.pos, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
        self.transform_funcs.rotate(np.int32(self.dir.size), self.dir, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.dir.size//self.nblocks+1,1))

    def rotate_around_point(self, phi, n, point):
        """"Rotate by an angle phi around the axis `n` passing through
        the point `point`."""
        self.transform_funcs.rotate_around_point(np.int32(self.pos.size), self.pos, np.float32(phi), ga.vec.make_float3(*n), ga.vec.make_float3(*point), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
        self.transform_funcs.rotate(np.int32(self.dir.size), self.dir, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.dir.size//self.nblocks+1,1))

    def translate(self, v):
        "Translate the ray positions by the vector `v`."
        self.transform_funcs.translate(np.int32(self.pos.size), self.pos, ga.vec.make_float3(*v), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))

    def render(self, gpu_geometry, pixels, alpha_depth=10,
               keep_last_render=False,bg_color=0x00000000):
        """Render `gpu_geometry` and fill the GPU array `pixels` with pixel
        colors."""
        if not keep_last_render:
            self.dxlen.fill(0)

        if alpha_depth > self.max_alpha_depth:
            raise Exception('alpha_depth > max_alpha_depth')

        if not isinstance(pixels, ga.GPUArray):
            raise TypeError('`pixels` must be a %s instance.' % ga.GPUArray)

        if pixels.size != self.pos.size:
            raise ValueError('`pixels`.size != number of rays')

        total_rays = self.pos.size
        optix_raycaster = getattr(gpu_geometry, 'optix_raycaster', None)
        use_optix = optix_raycaster is not None and alpha_depth > 0
        optix_distances_gpu = None
        optix_triangles_gpu = None

        if use_optix:
            try:
                origins_base = self.pos.view(np.float32).reshape(total_rays, 3)
                directions_gpu = self.dir.view(np.float32).reshape(total_rays, 3)
                origins_current = origins_base.copy()

                optix_distances_gpu = ga.empty((alpha_depth, total_rays), dtype=np.float32)
                optix_distances_gpu.fill(-1.0)
                optix_triangles_gpu = ga.empty((alpha_depth, total_rays), dtype=np.int32)
                optix_triangles_gpu.fill(-1)

                stride = np.int32(3)
                eps = np.float32(1e-4)

                for depth in range(alpha_depth):
                    distances_dev, triangles_dev, normals_dev = optix_raycaster.trace_many(
                        origins_current,
                        directions_gpu,
                        tmin=1e-4,
                        tmax=1e16,
                        return_device=True,
                    )

                    distances_gpu = _device_array_to_gpuarray(distances_dev)
                    triangles_gpu = _device_array_to_gpuarray(triangles_dev)

                    cuda.memcpy_dtod(
                        int(optix_distances_gpu.gpudata) + depth * total_rays * distances_gpu.dtype.itemsize,
                        int(distances_gpu.gpudata),
                        distances_gpu.nbytes,
                    )
                    cuda.memcpy_dtod(
                        int(optix_triangles_gpu.gpudata) + depth * total_rays * triangles_gpu.dtype.itemsize,
                        int(triangles_gpu.gpudata),
                        triangles_gpu.nbytes,
                    )

                    _UPDATE_ORIGINS_KERNEL(
                        origins_current.reshape(-1),
                        directions_gpu.reshape(-1),
                        distances_gpu,
                        stride,
                        eps,
                    )

                    del distances_gpu
                    del triangles_gpu
                    del distances_dev
                    del triangles_dev
                    del normals_dev

            except Exception:
                optix_distances_gpu = None
                optix_triangles_gpu = None
                use_optix = False

        dist_ptr = self.wavelengths.gpudata if not use_optix else optix_distances_gpu.reshape(-1).gpudata
        tri_ptr = self.last_hit_triangles.gpudata if not use_optix else optix_triangles_gpu.reshape(-1).gpudata

        self.render_funcs.render(
            np.int32(total_rays),
            self.pos,
            self.dir,
            gpu_geometry.gpudata,
            np.uint32(alpha_depth),
            pixels,
            self.dx,
            self.dxlen,
            self.color,
            np.uint32(bg_color),
            dist_ptr,
            tri_ptr,
            np.int32(1 if use_optix else 0),
            block=(self.nblocks, 1, 1),
            grid=(total_rays // self.nblocks + 1, 1),
        )

    def snapshot(self, gpu_geometry, alpha_depth=10):
        "Render `gpu_geometry` and return a numpy array of pixel colors."
        pixels = ga.empty(self.pos.size, dtype=np.uint32)
        self.render(gpu_geometry, pixels, alpha_depth)
        return pixels.get()
