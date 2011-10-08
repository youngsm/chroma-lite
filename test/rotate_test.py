import os
import numpy as np
from pycuda import autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda import gpuarray
float3 = gpuarray.vec.float3

def rotate(x, phi, n):
    x = np.asarray(x)
    n = np.asarray(n)

    r = np.cos(phi)*np.identity(3) + (1-np.cos(phi))*np.outer(n,n) + \
        np.sin(phi)*np.array([[0,n[2],-n[1]],[-n[2],0,n[0]],[n[1],-n[0],0]])

    return np.inner(x,r)

print 'device %s' % autoinit.device.name()

current_directory = os.path.split(os.path.realpath(__file__))[0]
from chroma.cuda import srcdir as source_directory

source = open(current_directory + '/rotate_test.cu').read()

mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)

rotate_gpu = mod.get_function('rotate')

size = {'block': (100,1,1), 'grid': (1,1)}

a = np.empty(size['block'][0], dtype=float3)
n = np.empty(size['block'][0], dtype=float3)
phi = np.random.random_sample(size=a.size).astype(np.float32)

a['x'] = np.random.random_sample(size=a.size)
a['y'] = np.random.random_sample(size=a.size)
a['z'] = np.random.random_sample(size=a.size)

n['x'] = np.random.random_sample(size=n.size)
n['y'] = np.random.random_sample(size=n.size)
n['z'] = np.random.random_sample(size=n.size)

a['x'] = np.ones(a.size)
a['y'] = np.zeros(a.size)
a['z'] = np.zeros(a.size)

n['x'] = np.zeros(n.size)
n['y'] = np.zeros(n.size)
n['z'] = np.ones(n.size)

phi = np.array([np.pi/2]*a.size).astype(np.float32)

def testrotate():
    dest = np.empty(a.size, dtype=float3)
    rotate_gpu(cuda.In(a), cuda.In(phi), cuda.In(n), cuda.Out(dest), **size)
    for v, theta, w, rdest in zip(a,phi,n,dest):
        r = rotate((v['x'], v['y'], v['z']), theta, (w['x'], w['y'], w['z']))
        if not np.allclose(rdest['x'], r[0]) or \
                not np.allclose(rdest['y'], r[1]) or \
                not np.allclose(rdest['z'], r[2]):
            print v
            print theta
            print w
            print r
            print rdest
            assert False

