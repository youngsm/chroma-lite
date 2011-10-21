#!/usr/bin/env python
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
import time
from uncertainties import ufloat
import sys
import itertools

from chroma import gpu
from chroma import event
from chroma import sample
from chroma import generator
from chroma import tools
from chroma.transform import normalize
from chroma.demo.optics import water

# Generator processes need to fork BEFORE the GPU context is setup
g4generator = generator.photon.G4ParallelGenerator(4, water)

def intersect(gpu_geometry, number=100, nphotons=500000, nthreads_per_block=64,
              max_blocks=1024):
    "Returns the average number of ray intersections per second."
    distances_gpu = ga.empty(nphotons, dtype=np.float32)

    module = gpu.get_cu_module('mesh.h', options=('--use_fast_math',))
    gpu_funcs = gpu.GPUFuncs(module)

    run_times = []
    for i in tools.progress(range(number)):
        pos = ga.zeros(nphotons, dtype=ga.vec.float3)
        dir = ga.to_gpu(gpu.to_float3(sample.uniform_sphere(nphotons)))

        t0 = time.time()
        gpu_funcs.distance_to_mesh(np.int32(pos.size), pos, dir, gpu_geometry.gpudata, distances_gpu, block=(nthreads_per_block,1,1), grid=(pos.size//nthreads_per_block+1,1))
        cuda.Context.get_current().synchronize()
        elapsed = time.time() - t0

        if i > 0:
            # first kernel call incurs some driver overhead
            run_times.append(elapsed)

    return nphotons/ufloat((np.mean(run_times),np.std(run_times)))

def load_photons(number=100, nphotons=500000):
    """Returns the average number of photons moved to the GPU device memory
    per second."""
    pos = np.zeros((nphotons,3))
    dir = sample.uniform_sphere(nphotons)
    pol = normalize(np.cross(sample.uniform_sphere(nphotons), dir))
    wavelengths = np.random.uniform(400,800,size=nphotons)
    photons = event.Photons(pos, dir, pol, wavelengths)

    run_times = []
    for i in tools.progress(range(number)):
        t0 = time.time()
        gpu_photons = gpu.GPUPhotons(photons)
        cuda.Context.get_current().synchronize()
        elapsed = time.time() - t0

        if i > 0:
            # first kernel call incurs some driver overhead
            run_times.append(elapsed)

    return nphotons/ufloat((np.mean(run_times),np.std(run_times)))

def propagate(gpu_detector, number=10, nphotons=500000, nthreads_per_block=64,
              max_blocks=1024):
    "Returns the average number of photons propagated on the GPU per second."
    rng_states = gpu.get_rng_states(nthreads_per_block*max_blocks)

    run_times = []
    for i in tools.progress(range(number)):
        pos = np.zeros((nphotons,3))
        dir = sample.uniform_sphere(nphotons)
        pol = normalize(np.cross(sample.uniform_sphere(nphotons), dir))
        wavelengths = np.random.uniform(400,800,size=nphotons)
        photons = event.Photons(pos, dir, pol, wavelengths)
        gpu_photons = gpu.GPUPhotons(photons)

        t0 = time.time()
        gpu_photons.propagate(gpu_detector, rng_states, nthreads_per_block,
                              max_blocks)
        cuda.Context.get_current().synchronize()
        elapsed = time.time() - t0

        if i > 0:
            # first kernel call incurs some driver overhead
            run_times.append(elapsed)

    return nphotons/ufloat((np.mean(run_times),np.std(run_times)))

@tools.profile_if_possible
def pdf(gpu_detector, npdfs=10, nevents=100, nreps=16, ndaq=1,
        nthreads_per_block=64, max_blocks=1024):
    """
    Returns the average number of 100 MeV events per second that can be
    histogrammed per second.

    Args:
        - gpu_instance, chroma.gpu.GPU
            The GPU instance passed to the GPUDetector constructor.
        - npdfs, int
            The number of pdf generations to average.
        - nevents, int
            The number of 100 MeV events to generate for each PDF.
        - nreps, int
            The number of times to propagate each event and add to PDF
        - ndaq, int
            The number of times to run the DAQ simulation on the propagated
            event and add it to the PDF.
    """
    rng_states = gpu.get_rng_states(nthreads_per_block*max_blocks)

    gpu_daq = gpu.GPUDaq(gpu_detector)
    gpu_pdf = gpu.GPUPDF()
    gpu_pdf.setup_pdf(gpu_detector.nchannels, 100, (-0.5, 999.5), 10, (-0.5, 9.5))

    run_times = []
    for i in tools.progress(range(npdfs)):
        t0 = time.time()
        gpu_pdf.clear_pdf()

        vertex_gen = generator.vertex.constant_particle_gun('e-', (0,0,0),
                                                            (1,0,0), 100)
        vertex_iter = itertools.islice(vertex_gen, nevents)

        for ev in g4generator.generate_events(vertex_iter):
            gpu_photons = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)

            gpu_photons.propagate(gpu_detector, rng_states,
                                  nthreads_per_block, max_blocks)
            for gpu_photon_slice in gpu_photons.iterate_copies():
                for idaq in xrange(ndaq):
                    gpu_channels = gpu_daq.acquire(gpu_photon_slice, rng_states,
                                                   nthreads_per_block, max_blocks)
                    gpu_pdf.add_hits_to_pdf(gpu_channels, nthreads_per_block)

        hitcount, pdf = gpu_pdf.get_pdfs()

        elapsed = time.time() - t0

        if i > 0:
            # first kernel call incurs some driver overhead
            run_times.append(elapsed)

    return nevents*nreps*ndaq/ufloat((np.mean(run_times),np.std(run_times)))

@tools.profile_if_possible
def pdf_eval(gpu_detector, npdfs=10, nevents=25, nreps=16, ndaq=128,
             nthreads_per_block=64, max_blocks=1024):
    """
    Returns the average number of 100 MeV events that can be
    histogrammed per second.

    Args:
        - gpu_instance, chroma.gpu.GPU
            The GPU instance passed to the GPUDetector constructor.
        - npdfs, int
            The number of pdf generations to average.
        - nevents, int
            The number of 100 MeV events to generate for each PDF.
        - nreps, int
            The number of times to propagate each event and add to PDF
        - ndaq, int
            The number of times to run the DAQ simulation on the propagated
            event and add it to the PDF.
    """
    rng_states = gpu.get_rng_states(nthreads_per_block*max_blocks)

    # Make data event
    data_ev = g4generator.generate_events(itertools.islice(generator.vertex.constant_particle_gun('e-', (0,0,0),
                                                                                                  (1,0,0), 100),
                                                           1)).next()
    gpu_photons = gpu.GPUPhotons(data_ev.photons_beg)

    gpu_photons.propagate(gpu_detector, rng_states,
                          nthreads_per_block, max_blocks)
    gpu_daq = gpu.GPUDaq(gpu_detector)
    data_ev_channels = gpu_daq.acquire(gpu_photons, rng_states, nthreads_per_block, max_blocks).get()
    
    # Setup PDF evaluation
    gpu_daq = gpu.GPUDaq(gpu_detector, ndaq=ndaq)
    gpu_pdf = gpu.GPUPDF()
    gpu_pdf.setup_pdf_eval(data_ev_channels.hit,
                           data_ev_channels.t,
                           data_ev_channels.q,
                           0.05,
                           (-0.5, 999.5),
                           1.0,
                           (-0.5, 20),
                           min_bin_content=20,
                           time_only=True)

    run_times = []
    for i in tools.progress(range(npdfs)):
        t0 = time.time()
        gpu_pdf.clear_pdf_eval()

        vertex_gen = generator.vertex.constant_particle_gun('e-', (0,0,0),
                                                            (1,0,0), 100)
        vertex_iter = itertools.islice(vertex_gen, nevents)

        for ev in g4generator.generate_events(vertex_iter):
            gpu_photons = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)

            gpu_photons.propagate(gpu_detector, rng_states,
                                  nthreads_per_block, max_blocks)
            for gpu_photon_slice in gpu_photons.iterate_copies():
                gpu_photon_slice = gpu_photon_slice.select(event.SURFACE_DETECT)
                gpu_channels = gpu_daq.acquire(gpu_photon_slice, rng_states,
                                               nthreads_per_block, max_blocks)
                gpu_pdf.accumulate_pdf_eval(gpu_channels, nthreads_per_block)

        cuda.Context.get_current().synchronize()        
        elapsed = time.time() - t0

        if i > 0:
            # first kernel call incurs some driver overhead
            run_times.append(elapsed)

    return nevents*nreps*ndaq/ufloat((np.mean(run_times),np.std(run_times)))


if __name__ == '__main__':
    from chroma import demo
    import gc

    tools.enable_debug_on_crash()

    # Default to run all tests
    tests = ['ray', 'load', 'propagate', 'pdf', 'pdf_eval']
    if len(sys.argv) > 1:
        tests = sys.argv[1:] # unless test names given on command line

    detector = demo.detector()
    detector.build(bits=11)

    context = gpu.create_cuda_context()
    gpu_detector = gpu.GPUDetector(detector)
    
    if 'ray' in tests:
        print '%s ray intersections/sec.' % \
            tools.ufloat_to_str(intersect(gpu_detector))
        # run garbage collection since there is a reference loop
        # in the GPUArray class.
        gc.collect()

    if 'load' in tests:
        print '%s photons loaded/sec.' % tools.ufloat_to_str(load_photons())
        gc.collect()

    if 'propagate' in tests:
        print '%s photons propagated/sec.' % \
            tools.ufloat_to_str(propagate(gpu_detector))
        gc.collect()

    if 'pdf' in tests:
        print '%s 100 MeV events histogrammed/s' % \
            tools.ufloat_to_str(pdf(gpu_detector))
        gc.collect()

    if 'pdf_eval' in tests:
        print '%s 100 MeV events/s accumulated in PDF evaluation data structure (100 GEANT4 x 16 Chroma x 128 DAQ)' % \
            tools.ufloat_to_str(pdf_eval(gpu_detector))

    context.pop()
