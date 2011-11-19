import numpy as np

# Photon history bits (see photon.h for source)
NO_HIT           = 0x1 << 0,
BULK_ABSORB      = 0x1 << 1,
SURFACE_DETECT   = 0x1 << 2,
SURFACE_ABSORB   = 0x1 << 3,
RAYLEIGH_SCATTER = 0x1 << 4,
REFLECT_DIFFUSE  = 0x1 << 5,
REFLECT_SPECULAR = 0x1 << 6,
NAN_ABORT        = 0x1 << 31

class Vertex(object):
    def __init__(self, particle_name, pos, dir, ke, t0=0.0, pol=None):
        '''Create a particle vertex.

           particle_name: string
               Name of particle, following the GEANT4 convention.  
               Examples: e-, e+, gamma, mu-, mu+, pi0

           pos: array-like object, length 3
               Position of particle vertex (mm)

           dir: array-like object, length 3
               Normalized direction vector

           ke: float
               Kinetic energy (MeV)

           t0: float
               Initial time of particle (ns)
               
           pol: array-like object, length 3
               Normalized polarization vector.  By default, set to None,
               and the particle is treated as having a random polarization.
        '''
        self.particle_name = particle_name
        self.pos = pos
        self.dir = dir
        self.pol = pol
        self.ke = ke
        self.t0 = t0

class Photons(object):
    def __init__(self, pos, dir, pol, wavelengths, t=None, last_hit_triangles=None, flags=None):
        '''Create a new list of n photons.

            pos: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Position 3-vectors (mm)

            dir: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Direction 3-vectors (normalized)

            pol: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Polarization direction 3-vectors (normalized)

            wavelengths: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon wavelengths (nm)

            t: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon times (ns)

            last_hit_triangles: numpy.ndarray(dtype=numpy.int32, shape=n)
               ID number of last intersected triangle.  -1 if no triangle hit in last step
               If set to None, a default array filled with -1 is created

            flags: numpy.ndarray(dtype=numpy.uint32, shape=n)
               Bit-field indicating the physics interaction history of the photon.  See 
               history bit constants in chroma.event for definition.
        '''
        self.pos = pos
        self.dir = dir
        self.pol = pol
        self.wavelengths = wavelengths

        if t is None:
            self.t = np.zeros(len(pos), dtype=np.float32)
        else:
            self.t = t

        if last_hit_triangles is None:
            self.last_hit_triangles = np.empty(len(pos), dtype=np.int32)
            self.last_hit_triangles.fill(-1)
        else:
            self.last_hit_triangles = last_hit_triangles

        if flags is None:
            self.flags = np.zeros(len(pos), dtype=np.uint32)
        else:
            self.flags = flags

    def __add__(self, other):
        '''Concatenate two Photons objects into one list of photons.

           other: chroma.event.Photons
              List of photons to add to self.

           Returns: new instance of chroma.event.Photons containing the photons in self and other.
        '''
        pos = np.concatenate((self.pos, other.pos))
        dir = np.concatenate((self.dir, other.dir))
        pol = np.concatenate((self.pol, other.pol))
        wavelengths = np.concatenate((self.wavelengths, other.wavelengths))
        t = np.concatenate((self.t, other.t))
        last_hit_triangles = np.concatenate((self.last_hit_triangles, other.last_hit_triangles))
        flags = np.concatenate((self.flags, other.flags))
        return Photons(pos, dir, pol, wavelengths, t, last_hit_triangles, flags)

    def __len__(self):
        '''Returns the number of photons in self.'''
        return len(self.pos)

class Channels(object):
    def __init__(self, hit, t, q, flags=None):
        '''Create a list of n channels.  All channels in the detector must 
        be included, regardless of whether they were hit.

           hit: numpy.ndarray(dtype=bool, shape=n)
             Hit state of each channel.

           t: numpy.ndarray(dtype=numpy.float32, shape=n)
             Hit time of each channel. (ns)

           q: numpy.ndarray(dtype=numpy.float32, shape=n)
             Integrated charge from hit.  (units same as charge 
             distribution in detector definition)
        '''
        self.hit = hit
        self.t = t
        self.q = q
        self.flags = flags

    def hit_channels(self):
        '''Extract a list of hit channels.
        
        Returns: array of hit channel IDs, array of hit times, array of charges on hit channels
        '''
        return self.hit.nonzero(), self.t[self.hit], self.q[self.hit]

class Event(object):
    def __init__(self, id=0, primary_vertex=None, vertices=None, photons_beg=None, photons_end=None, channels=None):
        '''Create an event.

            id: int
              ID number of this event

            primary_vertex: chroma.event.Vertex
              Vertex information for primary generating particle.
              
            vertices: list of chroma.event.Vertex objects
              Starting vertices to propagate in this event.  By default
              this is the primary vertex, but complex interactions
              can be representing by putting vertices for the
              outgoing products in this list.

            photons_beg: chroma.event.Photons
              Set of initial photon vertices in this event

            photons_end: chroma.event.Photons
              Set of final photon vertices in this event

            channels: chroma.event.Channels
              Electronics channel readout information.  Every channel
              should be included, with hit or not hit status indicated
              by the channels.hit flags.
        '''
        self.id = id

        self.nphotons = None

        self.primary_vertex = primary_vertex

        if vertices is not None:
            if np.iterable(vertices):
                self.vertices = vertices
            else:
                self.vertices = [vertices]
        else:
            self.vertices = []

        self.photons_beg = photons_beg
        self.photons_end = photons_end
        self.channels = channels
