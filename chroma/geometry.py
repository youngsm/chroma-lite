import sys
import os
from hashlib import md5
import cPickle as pickle
import gzip
import numpy as np
import time

from chroma.itertoolset import *
from chroma.tools import timeit, profile_if_possible, filled_array
from chroma.log import logger

# all material/surface properties are interpolated at these
# wavelengths when they are sent to the gpu
standard_wavelengths = np.arange(200, 810, 20).astype(np.float32)

class Mesh(object):
    "Triangle mesh object."
    def __init__(self, vertices, triangles, remove_duplicate_vertices=False):
        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32)

        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError('shape mismatch')

        if len(triangles.shape) != 2 or triangles.shape[1] != 3:
            raise ValueError('shape mismatch')

        if (triangles < 0).any():
            raise ValueError('indices in `triangles` must be positive.')

        if (triangles >= len(vertices)).any():
            raise ValueError('indices in `triangles` must be less than the '
                             'length of the vertex array.')

        self.vertices = vertices
        self.triangles = triangles

        if remove_duplicate_vertices:
            self.remove_duplicate_vertices()

    def get_bounds(self):
        "Return the lower and upper bounds for the mesh as a tuple."
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def remove_duplicate_vertices(self):
        "Remove any duplicate vertices in the mesh."
        # view the vertices as a structured array in order to identify unique
        # rows, i.e. unique vertices
        unique_vertices, inverse = np.unique(self.vertices.view([('', self.vertices.dtype)]*self.vertices.shape[1]), return_inverse=True)
        # turn the structured vertex array back into a normal array
        self.vertices = unique_vertices.view(self.vertices.dtype).reshape((unique_vertices.shape[0], self.vertices.shape[1]))
        # remap the triangles to the unique vertices
        self.triangles = inverse[self.triangles]

    def assemble(self, key=slice(None), group=True):
        """
        Return an assembled triangle mesh; i.e. return the vertex positions
        of every triangle. If `group` is True, the array returned will have
        an extra axis denoting the triangle number; i.e. if the mesh contains
        N triangles, the returned array will have the shape (N,3,3). If `group`
        is False, return just the vertex positions without any grouping; in
        this case the grouping is implied (i.e. elements [0:3] are the first
        triangle, [3:6] the second, and so on.

        The `key` argument is a slice object if you just want to assemble
        a certain range of the triangles, i.e. to get the assembled mesh for
        triangles 3 through 6, key = slice(3,7).
        """
        if group:
            vertex_indices = self.triangles[key]
        else:
            vertex_indices = self.triangles[key].flatten()

        return self.vertices[vertex_indices]

    def __add__(self, other):
        return Mesh(np.concatenate((self.vertices, other.vertices)), np.concatenate((self.triangles, other.triangles + len(self.vertices))))

def memoize_method_with_dictionary_arg(func):
    def lookup(*args):
        # based on function by Michele Simionato
        # http://www.phyast.pitt.edu/~micheles/python/
        # Modified to work for class method with dictionary argument

        assert len(args) == 2
        # create hashable arguments by replacing dictionaries with tuples of items
        dict_items = args[1].items()
        dict_items.sort()
        hashable_args = (args[0], tuple(dict_items))
        try:
            return func._memoize_dic[hashable_args]
        except AttributeError:
            # _memoize_dic doesn't exist yet.

            result = func(*args)
            func._memoize_dic = {hashable_args: result}
            return result
        except KeyError:
            result = func(*args)
            func._memoize_dic[hashable_args] = result
            return result
    return lookup

class Solid(object):
    """Solid object attaches materials, surfaces, and colors to each triangle
    in a Mesh object."""
    def __init__(self, mesh, material1=None, material2=None, surface=None, color=0xffffff):
        self.mesh = mesh

        if np.iterable(material1):
            if len(material1) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.material1 = np.array(material1, dtype=np.object)
        else:
            self.material1 = np.tile(material1, len(self.mesh.triangles))

        if np.iterable(material2):
            if len(material2) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.material2 = np.array(material2, dtype=np.object)
        else:
            self.material2 = np.tile(material2, len(self.mesh.triangles))

        if np.iterable(surface):
            if len(surface) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.surface = np.array(surface, dtype=np.object)
        else:
            self.surface = np.tile(surface, len(self.mesh.triangles))

        if np.iterable(color):
            if len(color) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.color = np.array(color, dtype=np.uint32)
        else:
            self.color = np.tile(color, len(self.mesh.triangles)).astype(np.uint32)

        self.unique_materials = \
            np.unique(np.concatenate([self.material1, self.material2]))

        self.unique_surfaces = np.unique(self.surface)

    def __add__(self, other):
        return Solid(self.mesh + other.mesh, np.concatenate((self.material1, other.material1)), np.concatenate((self.material2, other.material2)), np.concatenate((self.surface, other.surface)), np.concatenate((self.color, other.color)))

    @memoize_method_with_dictionary_arg
    def material1_indices(self, material_lookup):
        return np.fromiter(imap(material_lookup.get, self.material1), dtype=np.int32, count=len(self.material1))

    @memoize_method_with_dictionary_arg
    def material2_indices(self, material_lookup):
        return np.fromiter(imap(material_lookup.get, self.material2), dtype=np.int32, count=len(self.material2))

    @memoize_method_with_dictionary_arg
    def surface_indices(self, surface_lookup):
        return np.fromiter(imap(surface_lookup.get, self.surface), dtype=np.int32, count=len(self.surface))

class Material(object):
    """Material optical properties."""
    def __init__(self, name='none'):
        self.name = name

        self.refractive_index = None
        self.absorption_length = None
        self.scattering_length = None
        self.density = 0.0 # g/cm^3
        self.composition = {} # by mass

    def set(self, name, value, wavelengths=standard_wavelengths):
        if np.iterable(value):
            if len(value) != len(wavelengths):
                raise ValueError('shape mismatch')
        else:
            value = np.tile(value, len(wavelengths))

        self.__dict__[name] = np.array(zip(wavelengths, value), dtype=np.float32)

# Empty material
vacuum = Material('vacuum')
vacuum.set('refractive_index', 1.0)
vacuum.set('absorption_length', 1e6)
vacuum.set('scattering_length', 1e6)


class Surface(object):
    """Surface optical properties."""
    def __init__(self, name='none'):
        self.name = name

        self.set('detect', 0)
        self.set('absorb', 0)
        self.set('reflect_diffuse', 0)
        self.set('reflect_specular', 0)

    def set(self, name, value, wavelengths=standard_wavelengths):
        if np.iterable(value):
            if len(value) != len(wavelengths):
                raise ValueError('shape mismatch')
        else:
            value = np.tile(value, len(wavelengths))

        if (np.asarray(value) < 0.0).any():
            raise Exception('all probabilities must be >= 0.0')

        self.__dict__[name] = np.array(zip(wavelengths, value), dtype=np.float32)
    def __repr__(self):
        return '<Surface %s>' % self.name
        
def interleave(arr, bits):
    """
    Interleave the bits of quantized three-dimensional points in space.

    Example
        >>> interleave(np.identity(3, dtype=np.int))
        array([4, 2, 1], dtype=uint64)
    """
    if len(arr.shape) != 2 or arr.shape[1] != 3:
        raise Exception('shape mismatch')

    z = np.zeros(arr.shape[0], dtype=np.uint64)
    for i in range(bits):
        z |= (arr[:,2] & 1 << i) << (2*i) | \
             (arr[:,1] & 1 << i) << (2*i+1) | \
             (arr[:,0] & 1 << i) << (2*i+2)
    return z

def morton_order(mesh, bits):
    """
    Return a list of zvalues for triangles in `mesh` by interleaving the
    bits of the quantized center coordinates of each triangle. Each coordinate
    axis is quantized into 2**bits bins.
    """
    lower_bound, upper_bound = mesh.get_bounds()

    if bits <= 0 or bits > 21:
        raise Exception('number of bits must be in the range (0,21].')

    max_value = 2**bits - 1

    def quantize(x):
        return np.uint64((x-lower_bound)*max_value/(upper_bound-lower_bound))

    mean_positions = quantize(np.mean(mesh.assemble(), axis=1))

    return interleave(mean_positions, bits)

class Geometry(object):
    "Geometry object."
    def __init__(self, detector_material=None):
        self.detector_material = detector_material
        self.solids = []
        self.solid_rotations = []
        self.solid_displacements = []

    def add_solid(self, solid, rotation=None, displacement=None):
        """
        Add the solid `solid` to the geometry. When building the final triangle
        mesh, `solid` will be placed by rotating it with the rotation matrix
        `rotation` and displacing it by the vector `displacement`.
        """

        if rotation is None:
            rotation = np.identity(3)
        else:
            rotation = np.asarray(rotation, dtype=np.float32)

        if rotation.shape != (3,3):
            raise ValueError('rotation matrix has the wrong shape.')

        self.solid_rotations.append(rotation.astype(np.float32))

        if displacement is None:
            displacement = np.zeros(3)
        else:
            displacement = np.asarray(displacement, dtype=np.float32)

        if displacement.shape != (3,):
            raise ValueError('displacement vector has the wrong shape.')

        self.solid_displacements.append(displacement)

        self.solids.append(solid)

        return len(self.solids)-1

    @profile_if_possible
    def build(self, bits=11, shift=3, use_cache=True):
        """
        Build the bounding volume hierarchy, material/surface code arrays, and
        color array for this geometry. If the bounding volume hierarchy is
        cached, load the cache instead of rebuilding, else build and cache it.

        Args:
            - bits: int, *optional*
                The number of bits to quantize each linear dimension with when
                morton ordering the triangle centers for building the bounding
                volume hierarchy. Defaults to 8.
            - shift: int, *optional*
                The number of bits to shift the zvalue of each node when
                building the next layer of the bounding volume hierarchy.
                Defaults to 3.
            - use_cache: bool, *optional*
                If true, the on-disk cache in ~/.chroma/ will be checked for
                a previously built version of this geometry, otherwise the
                BVH will be computed and saved to the cache.  If false,
                the cache is ignored and also not updated.
        """
        nv = np.cumsum([0] + [len(solid.mesh.vertices) for solid in self.solids])
        nt = np.cumsum([0] + [len(solid.mesh.triangles) for solid in self.solids])

        vertices = np.empty((nv[-1],3), dtype=np.float32)
        triangles = np.empty((nt[-1],3), dtype=np.uint32)
        

        logger.info('Setting up BVH for detector mesh...')
        logger.info('  triangles: %d' % len(triangles))
        logger.info('  vertices:  %d' % len(vertices))


        for i, solid in enumerate(self.solids):
            vertices[nv[i]:nv[i+1]] = \
                np.inner(solid.mesh.vertices, self.solid_rotations[i]) + self.solid_displacements[i]
            triangles[nt[i]:nt[i+1]] = solid.mesh.triangles + nv[i]

        # Different solids are very unlikely to share vertices, so this goes much faster
        self.mesh = Mesh(vertices, triangles, remove_duplicate_vertices=False)

        self.colors = np.concatenate([solid.color for solid in self.solids])

        self.solid_id = np.concatenate([filled_array(i, shape=len(solid.mesh.triangles), dtype=np.uint32) for i, solid in enumerate(self.solids)])

        self.unique_materials = list(np.unique(np.concatenate([solid.unique_materials for solid in self.solids])))

        material_lookup = dict(zip(self.unique_materials, range(len(self.unique_materials))))

        self.material1_index = np.concatenate([solid.material1_indices(material_lookup) for solid in self.solids])

        self.material2_index = np.concatenate([solid.material2_indices(material_lookup) for solid in self.solids])

        self.unique_surfaces = list(np.unique(np.concatenate([solid.unique_surfaces for solid in self.solids])))

        surface_lookup = dict(zip(self.unique_surfaces, range(len(self.unique_surfaces))))

        self.surface_index = np.concatenate([solid.surface_indices(surface_lookup) for solid in self.solids])

        try:
            self.surface_index[self.surface_index == surface_lookup[None]] = -1
        except KeyError:
            pass

        checksum = md5(str(bits))
        checksum.update(str(shift))
        checksum.update(self.mesh.vertices)
        checksum.update(self.mesh.triangles)

        cache_dir = os.path.expanduser('~/.chroma')
        cache_file = checksum.hexdigest()+'.npz'
        cache_path = os.path.join(cache_dir, cache_file)

        if use_cache:
            try:
                npz_file = np.load(cache_path)
            except IOError:
                pass
            else:
                logger.info('Loading BVH from cache.')
                data = dict(npz_file)

                # take() is faster than fancy indexing by 5x!
                # tip from http://wesmckinney.com/blog/?p=215
                reorder = data.pop('reorder')
                self.mesh.triangles = self.mesh.triangles.take(reorder, axis=0)
                self.material1_index = self.material1_index.take(reorder, axis=0)
                self.material2_index = self.material2_index.take(reorder, axis=0)
                self.surface_index = self.surface_index.take(reorder, axis=0)
                self.colors = self.colors.take(reorder, axis=0)
                self.solid_id = self.solid_id.take(reorder, axis=0)

                for key, value in data.iteritems():
                    setattr(self, key, value)

                logger.info('  nodes: %d' % len(self.upper_bounds))
                return

        logger.info('Constructing new BVH from mesh.  This may take several minutes.')

        start_time = time.time()

        zvalues_mesh = morton_order(self.mesh, bits)
        reorder = np.argsort(zvalues_mesh)
        zvalues_mesh = zvalues_mesh[reorder]

        if (np.diff(zvalues_mesh) < 0).any():
            raise Exception('zvalues_mesh out of order.')

        self.mesh.triangles = self.mesh.triangles[reorder]

        self.material1_index = self.material1_index[reorder]
        self.material2_index = self.material2_index[reorder]
        self.surface_index = self.surface_index[reorder]
        self.colors = self.colors[reorder]
        self.solid_id = self.solid_id[reorder]

        unique_zvalues = np.unique(zvalues_mesh)

        while unique_zvalues.size > zvalues_mesh.size/np.e:
            zvalues_mesh = zvalues_mesh >> shift
            unique_zvalues = np.unique(zvalues_mesh)

        self.lower_bounds = np.empty((unique_zvalues.size,3), dtype=np.float32)
        self.upper_bounds = np.empty((unique_zvalues.size,3), dtype=np.float32)

        assembled_mesh = self.mesh.assemble(group=False)
        self.node_map = np.searchsorted(zvalues_mesh, unique_zvalues)
        self.node_map_end = np.searchsorted(zvalues_mesh, unique_zvalues, side='right')

        for i, (zi1, zi2) in enumerate(izip(self.node_map, self.node_map_end)):
            self.lower_bounds[i] = assembled_mesh[zi1*3:zi2*3].min(axis=0)
            self.upper_bounds[i] = assembled_mesh[zi1*3:zi2*3].max(axis=0)

        self.layers = np.zeros(unique_zvalues.size, dtype=np.uint32)
        self.first_node = unique_zvalues.size

        begin_last_layer = 0

        for layer in count(1):
            bit_shifted_zvalues = unique_zvalues >> shift
            unique_zvalues = np.unique(bit_shifted_zvalues)

            i0 = begin_last_layer + bit_shifted_zvalues.size

            self.node_map.resize(self.node_map.size+unique_zvalues.size)
            self.node_map[i0:] = np.searchsorted(bit_shifted_zvalues, unique_zvalues) + begin_last_layer
            self.node_map_end.resize(self.node_map_end.size+unique_zvalues.size)
            self.node_map_end[i0:] = np.searchsorted(bit_shifted_zvalues, unique_zvalues, side='right') + begin_last_layer

            self.layers.resize(self.layers.size+unique_zvalues.size)
            self.layers[i0:] = layer

            self.lower_bounds.resize((self.lower_bounds.shape[0]+unique_zvalues.size,3))
            self.upper_bounds.resize((self.upper_bounds.shape[0]+unique_zvalues.size,3))

            for i, zi1, zi2 in izip(count(i0), self.node_map[i0:], self.node_map_end[i0:]):
                self.lower_bounds[i] = self.lower_bounds[zi1:zi2].min(axis=0)
                self.upper_bounds[i] = self.upper_bounds[zi1:zi2].max(axis=0)

            begin_last_layer += bit_shifted_zvalues.size

            if unique_zvalues.size == 1:
                break

        self.start_node = self.node_map.size - 1

        logger.info('BVH construction completed in %1.1f seconds.' % (time.time() - start_time))
        logger.info('  nodes: %d' % len(self.upper_bounds))

        if use_cache:
            logger.info('Writing BVH to ~/.chroma cache directory...')
            sys.stdout.flush()

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            data = {}
            for key in ['lower_bounds', 'upper_bounds', 'node_map', 'node_map_end', 'layers', 'first_node', 'start_node']:
                data[key] = getattr(self, key)
            data['reorder'] = reorder
            np.savez_compressed(cache_path, **data)
