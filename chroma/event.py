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
        self.particle_name = particle_name
        self.pos = pos
        self.dir = dir
        self.pol = pol
        self.ke = ke
        self.t0 = t0

class Photons(object):
    def __init__(self, pos, dir, pol, wavelengths, t=None, last_hit_triangles=None, flags=None):
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
        pos = np.concatenate((self.pos, other.pos))
        dir = np.concatenate((self.dir, other.dir))
        pol = np.concatenate((self.pol, other.pol))
        wavelengths = np.concatenate((self.wavelengths, other.wavelengths))
        t = np.concatenate((self.t, other.t))
        last_hit_triangles = np.concatenate((self.last_hit_triangles, other.last_hit_triangles))
        flags = np.concatenate((self.flags, other.flags))
        return Photons(pos, dir, pol, wavelengths, t, last_hit_triangles, flags)

    def __len__(self):
        return len(self.pos)

class Channels(object):
    def __init__(self, hit, t, q, flags=None):
        self.hit = hit
        self.t = t
        self.q = q
        self.flags = flags

    def hit_channels(self):
        return self.hit.nonzero(), self.t[self.hit], self.q[self.hit]

class Event(object):
    def __init__(self, id=0, primary_vertex=None, vertices=None, photons_beg=None, photons_end=None, channels=None):
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
