import unittest
import numpy as np

import chroma.gpu.optix as optix


class TestOptixLoader(unittest.TestCase):
    def test_initialize_and_create_raycaster(self):
        if not optix.is_available():  # pragma: no cover -- depends on GPU availability
            self.skipTest("OptiX backend not available in this environment")

        # ensure initialization is idempotent
        optix.ensure_initialized()
        optix.ensure_initialized()

        raycaster_cls = optix.raycaster_class()
        self.assertTrue(callable(raycaster_cls))

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)

        caster = optix.create_raycaster(vertices, triangles)
        distance, triangle_idx, normal = caster.trace([0.25, 0.25, 1.0], [0.0, 0.0, -1.0])
        self.assertGreater(distance, 0.0)
        self.assertEqual(triangle_idx, 0)
        self.assertAlmostEqual(normal[2], 1.0, places=5)

        suppressed_distance, suppressed_triangle, _ = caster.trace(
            [0.25, 0.25, 1.0], [0.0, 0.0, -1.0], last_hit=triangle_idx
        )
        self.assertLess(suppressed_distance, 0.0)
        self.assertEqual(suppressed_triangle, -1)

        origins = np.array(
            [
                [0.25, 0.25, 1.0],
                [0.25, 0.25, 1.0],
            ],
            dtype=np.float32,
        )
        directions = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        last_hits = np.array([triangle_idx, -1], dtype=np.int32)

        distances, triangles, normals = caster.trace_many(origins, directions, last_hits=last_hits)
        self.assertEqual(distances.shape, (2,))
        # first ray should be suppressed because last_hits instructs us to ignore it
        self.assertLess(distances[0], 0.0)
        self.assertEqual(triangles[0], -1)
        self.assertLess(distances[1], 0.0)
        self.assertEqual(triangles[1], -1)

        distances2, triangles2, normals2 = caster.trace_many(origins, directions)
        self.assertAlmostEqual(distances2[0], distance, places=5)
        self.assertEqual(triangles2[0], triangle_idx)
        self.assertAlmostEqual(normals2[0, 2], 1.0, places=5)

        distances2, triangles2, _ = caster.trace_many(origins, directions)
        self.assertEqual(triangles2[0], triangle_idx)

        import pycuda.autoinit  # noqa: F401
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda

        origins_gpu = gpuarray.to_gpu(origins.astype(np.float32))
        directions_gpu = gpuarray.to_gpu(directions.astype(np.float32))
        last_hits_gpu = gpuarray.to_gpu(last_hits.astype(np.int32))

        class _Holder(cuda.PointerHolderBase):
            def __init__(self, device_array):
                super().__init__()
                self.device_array = device_array

            def get_pointer(self):
                return int(self.device_array.ptr)

        distances_dev, triangles_dev, normals_dev = caster.trace_many(
            origins_gpu, directions_gpu, last_hits=last_hits_gpu, return_device=True
        )

        iface = distances_dev.__cuda_array_interface__
        holder = _Holder(distances_dev)
        distances_gpu = gpuarray.GPUArray(iface['shape'], np.dtype(iface['typestr']), gpudata=holder, strides=iface.get('strides'))
        distances_gpu._holder = holder
        distances_gpu._device_array = distances_dev

        iface_tri = triangles_dev.__cuda_array_interface__
        holder_tri = _Holder(triangles_dev)
        triangles_gpu = gpuarray.GPUArray(iface_tri['shape'], np.dtype(iface_tri['typestr']), gpudata=holder_tri, strides=iface_tri.get('strides'))
        triangles_gpu._holder = holder_tri
        triangles_gpu._device_array = triangles_dev

        self.assertLess(distances_gpu.get()[0], 0.0)

        distances_gpu2, triangles_gpu2, _ = caster.trace_many(
            origins_gpu, directions_gpu, return_device=False
        )
        self.assertAlmostEqual(distances_gpu2[0], distance, places=5)
        self.assertEqual(triangles_gpu2[0], triangle_idx)


if __name__ == "__main__":
    unittest.main()
