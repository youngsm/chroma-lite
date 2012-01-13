'''chroma.cache: On-disk geometry and bounding volume hierarchy cache.

The ``Cache`` class is used to manage an on-disk cache of geometry and
BVH objects, which are slow to calculate.  By default, the cache is in
$HOME/.chroma.

Currently this cache is not thread or process-safe when a writing
process is present.
'''

import os
import cPickle as pickle

from chroma.log import logger

class GeometryNotFoundError(Exception):
    '''A requested geometry was not found in the on-disk cache.'''
    def __init__(self, msg):
        Exception.__init__(self, msg)

def verify_or_create_dir(dirname, exception_msg, logger_msg=None):
    '''Checks if ``dirname`` exists and is a directory.  If it does not exist,
    then it is created.  If it does exist, but is not a directory, an IOError
    is raised with ``exception_message`` as the description.

    If the directory is created, an info message will be sent to the 
    Chroma logger if ``logger_message`` is not None.
    '''
    if not os.path.isdir(dirname):
        if os.path.exists(dirname):
            raise IOError(exception_msg)
        else:
            if logger_msg is not None:
                logger.info(logger_msg)
            os.mkdir(dirname)

class Cache(object):
    '''Class for manipulating a Chroma disk cache directory.

    Use this class to read and write cached geometries or bounding
    volume hierarchies rather than reading and writing disk directly.
    '''
    
    def __init__(self, cache_dir=os.path.expanduser('~/.chroma/')):
        '''Open a Chroma cache stored at ``cache_dir``.
        
        If ``cache_dir`` does not already exist, it will be created.  By default,
        the cache is in the ``.chroma`` directory under the user's home directory.
        '''
        self.cache_dir = cache_dir
        verify_or_create_dir(self.cache_dir,
            exception_msg='Path for cache already exists, '
                          'but is not a directory: %s' % cache_dir,
            logger_msg='Creating new Chroma cache directory at %s' 
                       % cache_dir)


        self.geo_dir = os.path.join(cache_dir, 'geo')
        verify_or_create_dir(self.geo_dir,
            exception_msg='Path for geometry directory in cache '
                          'already exists, but is not a directory: %s' 
                          % self.geo_dir)

        self.bvh_dir = os.path.join(cache_dir, 'bvh')
        verify_or_create_dir(self.bvh_dir,
            exception_msg='Path for BVH directory in cache already '
                          'exists, but is not a directory: %s' % self.bvh_dir)


    def get_geometry_filename(self, name):
        '''Return the full pathname for the geometry file corresponding to 
        ``name``.
        '''
        return os.path.join(self.geo_dir, name)

    def list_geometry(self):
        '''Returns a list of all geometry names in the cache.'''
        return os.listdir(self.geo_dir)

    def save_geometry(self, name, geometry):
        '''Save ``geometry`` in the cache with the name ``name``.'''
        geo_file = self.get_geometry_filename(name)
        with open(geo_file, 'wb') as output_file:
            pickle.dump(geometry.mesh.md5(), output_file, 
                        pickle.HIGHEST_PROTOCOL)
            pickle.dump(geometry, output_file, 
                        pickle.HIGHEST_PROTOCOL)

    def load_geometry(self, name):
        '''Returns the chroma.geometry.Geometry object associated with
        ``name`` in the cache.

           Raises ``GeometryNotFoundError`` if ``name`` is not in the cache.
        '''
        geo_file = self.get_geometry_filename(name)
        if not os.path.exists(geo_file):
            raise GeometryNotFoundError(name)
        with open(geo_file, 'rb') as input_file:
            _ = pickle.load(input_file) # skip mesh hash
            geo = pickle.load(input_file)
            return geo

    def remove_geometry(self, name):
        '''Remove the geometry file associated with ``name`` from the cache.
        
        If ``name`` does not exist, no action is taken.
        '''
        geo_file = self.get_geometry_filename(name)
        if os.path.exists(geo_file):
            os.remove(geo_file)

    def get_geometry_hash(self, name):
        '''Get the mesh hash for the geometry associated with ``name``.

        This is faster than loading the entire geometry file and calling the
        hash() method on the mesh.
        '''
        geo_file = self.get_geometry_filename(name)
        if not os.path.exists(geo_file):
            raise GeometryNotFoundError(name)
        with open(geo_file, 'rb') as input_file:
            return pickle.load(input_file)

    def load_default_geometry(self):
        '''Load the default geometry as set by previous call to
        set_default_geometry().

        If no geometry has been designated the default, raise 
        GeometryNotFoundError.
        '''
        return self.load_geometry('.default')

    def set_default_geometry(self, name):
        '''Set the geometry in the cache corresponding to ``name`` to
        be the default geometry returned by load_default_geometry().
        '''
        default_geo_file = self.get_geometry_filename('.default')
        geo_file = self.get_geometry_filename(name)
        if not os.path.exists(geo_file):
            raise GeometryNotFoundError(name)
        
        if os.path.exists(default_geo_file):
            if os.path.islink(default_geo_file):
                os.remove(default_geo_file)
            else:
                raise IOError('Non-symlink found where expected a symlink: '
                              +default_geo_file)
        os.symlink(geo_file, default_geo_file)


    def list_bvh(self, mesh_hash):
        pass

    def save_bvh(self, mesh_hash, name=None):
        pass

    def load_bvh(self, mesh_hash, name=None):
        pass


