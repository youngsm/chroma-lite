import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda import characterize

from chroma.geometry import standard_wavelengths
from chroma.gpu.tools import get_cu_module, get_cu_source, cuda_options, \
    chunk_iterator, format_array, format_size, to_uint3, to_float3, \
    make_gpu_struct, GPUFuncs

from chroma.log import logger

def round_up_to_multiple(x, multiple):
    remainder = x % multiple
    if remainder == 0:
        return x
    else:
        return x + multiple - remainder

def compute_layer_configuration(n, branch_degree):
    if n == 1:
        # Special case for root
        return [ (1, 1) ]
    else:
        layer_conf = [ (n, round_up_to_multiple(n, branch_degree)) ]

    while layer_conf[0][1] > 1:
        nparent = int(np.ceil( float(layer_conf[0][1]) / branch_degree ))
        if nparent == 1:
            layer_conf = [ (1, 1) ] + layer_conf
        else:
            layer_conf = [ (nparent, round_up_to_multiple(nparent, branch_degree)) ] + layer_conf

    return layer_conf

def optimize_bvh_layer(layer, bvh_funcs):
    n = len(layer)
    areas = ga.empty(shape=n, dtype=np.uint32)
    union_areas = ga.empty(shape=n, dtype=np.uint32)
    nthreads_per_block = 128
    min_areas = ga.empty(shape=int(np.ceil(n/float(nthreads_per_block))), dtype=np.uint32)
    min_index = ga.empty_like(min_areas)

    update = 50000

    skip_size = 1
    flag = cuda.pagelocked_empty(shape=skip_size, dtype=np.uint32, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
    flag_gpu = np.intp(flag.base.get_device_pointer())
    print 'starting optimization'

    i = 0
    skips = 0
    while i < (n/2 - 1):
        # How are we doing?
        if i % update == 0:
            for first_index, elements_this_iter, nblocks_this_iter in \
                    chunk_iterator(n-1, nthreads_per_block, max_blocks=10000):

                bvh_funcs.distance_to_prev(np.uint32(first_index + 1),
                                           np.uint32(elements_this_iter),
                                           layer,
                                           union_areas,
                                           block=(nthreads_per_block,1,1),
                                           grid=(nblocks_this_iter,1))

            union_areas_host = union_areas.get()[1::2]
            print 'Area of parent layer: %1.12e' % union_areas_host.astype(float).sum()
            print 'Area of parent layer so far (%d): %1.12e' % (i*2, union_areas_host.astype(float)[:i].\
sum())
            print 'Skips:', skips

        test_index = i * 2

        blocks = 0
        look_forward = min(8192*400, n - test_index - 2)
        skip_this_round = min(skip_size, n - test_index - 1)
        flag[:] = 0
        for first_index, elements_this_iter, nblocks_this_iter in \
                chunk_iterator(look_forward, nthreads_per_block, max_blocks=10000):
            bvh_funcs.min_distance_to(np.uint32(first_index + test_index + 2),
                                      np.uint32(elements_this_iter),
                                      np.uint32(test_index),
                                      layer,
                                      np.uint32(blocks),
                                      min_areas,
                                      min_index,
                                      flag_gpu,
                                      block=(nthreads_per_block,1,1),
                                      grid=(nblocks_this_iter, skip_this_round))
            blocks += nblocks_this_iter
        cuda.Context.get_current().synchronize()

        if flag[0] == 0:
            flag_nonzero = flag.nonzero()[0]
            if len(flag_nonzero) == 0:
                no_swap_required = skip_size
            else:
                no_swap_required = flag_nonzero[0]
            i += no_swap_required
            skips += no_swap_required
            continue

        areas_host = min_areas[:blocks].get()
        min_index_host = min_index[:blocks].get()
        best_block = areas_host.argmin()
        better_i = min_index_host[best_block]

        if i % update == 0:
            print 'swapping %d and %d' % (test_index + 1, better_i)

        bvh_funcs.swap(np.uint32(test_index+1), np.uint32(better_i),
                       layer, block=(1,1,1), grid=(1,1))
        i += 1

    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(n-1, nthreads_per_block, max_blocks=10000):

        bvh_funcs.distance_to_prev(np.uint32(first_index + 1),
                                   np.uint32(elements_this_iter),
                                   layer,
                                   union_areas,
                                   block=(nthreads_per_block,1,1),
                                   grid=(nblocks_this_iter,1))

    union_areas_host = union_areas.get()[1::2]
    print 'Final area of parent layer: %1.12e' % union_areas_host.sum()
    print 'Skips:', skips

def make_bvh(vertices, gpu_vertices, ntriangles, gpu_triangles, branch_degree):
    assert branch_degree > 1
    bvh_module = get_cu_module('bvh.cu', options=cuda_options, 
                                include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)

    world_min = vertices.min(axis=0)
    # Full scale at 2**16 - 2 in order to ensure there is dynamic range to round
    # up by one count after quantization
    world_scale = np.max((vertices.max(axis=0) - world_min)) / (2**16 - 2)

    world_origin = ga.vec.make_float3(*world_min)
    world_scale  = np.float32(world_scale)

    layer_conf = compute_layer_configuration(ntriangles, branch_degree)
    layer_offsets = list(np.cumsum([npad for n, npad in layer_conf]))

    # Last entry is number of nodes, trim off and add zero to get offset of each layer
    n_nodes = int(layer_offsets[-1])
    layer_offsets = [0] + layer_offsets[:-1]

    leaf_nodes = ga.empty(shape=ntriangles, dtype=ga.vec.uint4)
    morton_codes = ga.empty(shape=ntriangles, dtype=np.uint64)

    # Step 1: Make leaves
    nthreads_per_block=256
    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(ntriangles, nthreads_per_block, max_blocks=10000):
        bvh_funcs.make_leaves(np.uint32(first_index),
                              np.uint32(elements_this_iter),
                              gpu_triangles, gpu_vertices, 
                              world_origin, world_scale,
                              leaf_nodes, morton_codes,
                              block=(nthreads_per_block,1,1),
                              grid=(nblocks_this_iter,1))

    # argsort on the CPU because I'm too lazy to do it on the GPU
    argsort = morton_codes.get().argsort().astype(np.uint32)
    del morton_codes
    local_leaf_nodes = leaf_nodes.get()[argsort]
    del leaf_nodes
    #del remap_order
    #
    #remap_order = ga.to_gpu(argsort)
    #m = morton_codes.get()
    #m.sort()
    #print m
    #assert False
    # Step 2: sort leaf nodes into full node list
    #print cuda.mem_get_info(), leaf_nodes.nbytes
    nodes = ga.zeros(shape=n_nodes, dtype=ga.vec.uint4)
    areas = ga.zeros(shape=n_nodes, dtype=np.uint32)
    cuda.memcpy_htod(int(nodes.gpudata)+int(layer_offsets[-1]), local_leaf_nodes)

    #for first_index, elements_this_iter, nblocks_this_iter in \
    #       chunk_iterator(ntriangles, nthreads_per_block, max_blocks=10000):
    #   bvh_funcs.reorder_leaves(np.uint32(first_index),
    #                            np.uint32(elements_this_iter),
    #                            leaf_nodes, nodes[layer_offsets[-1]:], remap_order,
    #                            block=(nthreads_per_block,1,1),
    #                            grid=(nblocks_this_iter,1))


    # Step 3: Create parent layers in reverse order
    layer_parameters = zip(layer_offsets[:-1], layer_offsets[1:], layer_conf)
    layer_parameters.reverse()

    i = len(layer_parameters)
    for parent_offset, child_offset, (nparent, nparent_pad) in layer_parameters:
        #if i < 30:
        #    optimize_bvh_layer(nodes[child_offset:child_offset+nparent*branch_degree],
        #                  bvh_funcs)

        for first_index, elements_this_iter, nblocks_this_iter in \
                chunk_iterator(nparent * branch_degree, nthreads_per_block,
                               max_blocks=10000):
            bvh_funcs.node_area(np.uint32(first_index+child_offset),
                                np.uint32(elements_this_iter),
                                nodes,
                                areas,
                                block=(nthreads_per_block,1,1),
                                grid=(nblocks_this_iter,1))

        print 'area', i, nparent * branch_degree, '%e' % areas[child_offset:child_offset+nparent*branch_degree].get().astype(float).sum()

        for first_index, elements_this_iter, nblocks_this_iter in \
                chunk_iterator(nparent, nthreads_per_block, max_blocks=10000):
            bvh_funcs.build_layer(np.uint32(first_index),
                                  np.uint32(elements_this_iter),
                                  np.uint32(branch_degree),
                                  nodes,
                                  np.uint32(parent_offset),
                                  np.uint32(child_offset),
                                  block=(nthreads_per_block,1,1),
                                  grid=(nblocks_this_iter,1))

        i -= 1

    return world_origin, world_scale, nodes

class GPUGeometry(object):
    def __init__(self, geometry, wavelengths=None, print_usage=False, branch_degree=2):
        if wavelengths is None:
            wavelengths = standard_wavelengths

        try:
            wavelength_step = np.unique(np.diff(wavelengths)).item()
        except ValueError:
            raise ValueError('wavelengths must be equally spaced apart.')

        geometry_source = get_cu_source('geometry_types.h')
        material_struct_size = characterize.sizeof('Material', geometry_source)
        surface_struct_size = characterize.sizeof('Surface', geometry_source)
        geometry_struct_size = characterize.sizeof('Geometry', geometry_source)

        self.material_data = []
        self.material_ptrs = []

        def interp_material_property(wavelengths, property):
            # note that it is essential that the material properties be
            # interpolated linearly. this fact is used in the propagation
            # code to guarantee that probabilities still sum to one.
            return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)

        for i in range(len(geometry.unique_materials)):
            material = geometry.unique_materials[i]

            if material is None:
                raise Exception('one or more triangles is missing a material.')

            refractive_index = interp_material_property(wavelengths, material.refractive_index)
            refractive_index_gpu = ga.to_gpu(refractive_index)
            absorption_length = interp_material_property(wavelengths, material.absorption_length)
            absorption_length_gpu = ga.to_gpu(absorption_length)
            scattering_length = interp_material_property(wavelengths, material.scattering_length)
            scattering_length_gpu = ga.to_gpu(scattering_length)

            self.material_data.append(refractive_index_gpu)
            self.material_data.append(absorption_length_gpu)
            self.material_data.append(scattering_length_gpu)

            material_gpu = \
                make_gpu_struct(material_struct_size,
                                [refractive_index_gpu, absorption_length_gpu,
                                 scattering_length_gpu,
                                 np.uint32(len(wavelengths)),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0])])

            self.material_ptrs.append(material_gpu)

        self.material_pointer_array = \
            make_gpu_struct(8*len(self.material_ptrs), self.material_ptrs)

        self.surface_data = []
        self.surface_ptrs = []

        for i in range(len(geometry.unique_surfaces)):
            surface = geometry.unique_surfaces[i]

            if surface is None:
                # need something to copy to the surface array struct
                # that is the same size as a 64-bit pointer.
                # this pointer will never be used by the simulation.
                self.surface_ptrs.append(np.uint64(0))
                continue

            detect = interp_material_property(wavelengths, surface.detect)
            detect_gpu = ga.to_gpu(detect)
            absorb = interp_material_property(wavelengths, surface.absorb)
            absorb_gpu = ga.to_gpu(absorb)
            reflect_diffuse = interp_material_property(wavelengths, surface.reflect_diffuse)
            reflect_diffuse_gpu = ga.to_gpu(reflect_diffuse)
            reflect_specular = interp_material_property(wavelengths, surface.reflect_specular)
            reflect_specular_gpu = ga.to_gpu(reflect_specular)

            self.surface_data.append(detect_gpu)
            self.surface_data.append(absorb_gpu)
            self.surface_data.append(reflect_diffuse_gpu)
            self.surface_data.append(reflect_specular_gpu)

            surface_gpu = \
                make_gpu_struct(surface_struct_size,
                                [detect_gpu, absorb_gpu,
                                 reflect_diffuse_gpu,
                                 reflect_specular_gpu,
                                 np.uint32(len(wavelengths)),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0])])

            self.surface_ptrs.append(surface_gpu)

        self.surface_pointer_array = \
            make_gpu_struct(8*len(self.surface_ptrs), self.surface_ptrs)

        self.pagelocked_vertices = cuda.pagelocked_empty(shape=len(geometry.mesh.vertices),
                                                         dtype=ga.vec.float3,
                                                         mem_flags=cuda.host_alloc_flags.DEVICEMAP | cuda.host_alloc_flags.WRITECOMBINED)
        self.pagelocked_triangles = cuda.pagelocked_empty(shape=len(geometry.mesh.triangles),
                                                         dtype=ga.vec.uint3,
                                                         mem_flags=cuda.host_alloc_flags.DEVICEMAP | cuda.host_alloc_flags.WRITECOMBINED)
        self.pagelocked_vertices[:] = to_float3(geometry.mesh.vertices)
        self.pagelocked_triangles[:] = to_uint3(geometry.mesh.triangles)
        self.vertices = np.intp(self.pagelocked_vertices.base.get_device_pointer())
        self.triangles = np.intp(self.pagelocked_triangles.base.get_device_pointer())


        self.branch_degree = branch_degree
        print 'bvh', cuda.mem_get_info()
        self.world_origin, self.world_scale, self.nodes = make_bvh(geometry.mesh.vertices,
                                                                   self.vertices,
                                                                   len(geometry.mesh.triangles),
                                                                   self.triangles,
                                                                   self.branch_degree)
        print 'bvh after', cuda.mem_get_info()

        material_codes = (((geometry.material1_index & 0xff) << 24) |
                          ((geometry.material2_index & 0xff) << 16) |
                          ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
        self.material_codes = ga.to_gpu(material_codes)
        colors = geometry.colors.astype(np.uint32)
        self.colors = ga.to_gpu(colors)
        self.solid_id_map = ga.to_gpu(geometry.solid_id.astype(np.uint32))

        self.gpudata = make_gpu_struct(geometry_struct_size,
                                       [self.vertices, self.triangles,
                                        self.material_codes,
                                        self.colors, self.nodes,
                                        self.material_pointer_array,
                                        self.surface_pointer_array,
                                        self.world_origin,
                                        self.world_scale,
                                        np.uint32(self.branch_degree)])

        self.geometry = geometry

        if print_usage:
            self.print_device_usage()
        logger.info(self.device_usage_str())

    def device_usage_str(self):
        '''Returns a formatted string displaying the memory usage.'''
        s = 'device usage:\n'
        s += '-'*10 + '\n'
        #s += format_array('vertices', self.vertices) + '\n'
        #s += format_array('triangles', self.triangles) + '\n'
        s += format_array('nodes', self.nodes) + '\n'
        s += '%-15s %6s %6s' % ('total', '', format_size(self.nodes.nbytes)) + '\n'
        s += '-'*10 + '\n'
        free, total = cuda.mem_get_info()
        s += '%-15s %6s %6s' % ('device total', '', format_size(total)) + '\n'
        s += '%-15s %6s %6s' % ('device used', '', format_size(total-free)) + '\n'
        s += '%-15s %6s %6s' % ('device free', '', format_size(free)) + '\n'
        return s

    def print_device_usage(self):
        print self.device_usage_str()
        print 

    def reset_colors(self):
        self.colors.set_async(self.geometry.colors.astype(np.uint32))

    def color_solids(self, solid_hit, colors, nblocks_per_thread=64,
                     max_blocks=1024):
        solid_hit_gpu = ga.to_gpu(np.array(solid_hit, dtype=np.bool))
        solid_colors_gpu = ga.to_gpu(np.array(colors, dtype=np.uint32))

        module = get_cu_module('mesh.h', options=cuda_options)
        color_solids = module.get_function('color_solids')

        for first_triangle, triangles_this_round, blocks in \
                chunk_iterator(self.triangles.size, nblocks_per_thread,
                               max_blocks):
            color_solids(np.int32(first_triangle),
                         np.int32(triangles_this_round), self.solid_id_map,
                         solid_hit_gpu, solid_colors_gpu, self.gpudata,
                         block=(nblocks_per_thread,1,1), 
                         grid=(blocks,1))

