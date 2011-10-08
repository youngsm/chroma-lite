import numpy as np
from pycuda import gpuarray as ga

from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    chunk_iterator
from chroma import event

class GPUChannels(object):
    def __init__(self, t, q, flags):
        self.t = t
        self.q = q
        self.flags = flags

    def get(self):
        t = self.t.get()
        q = self.q.get()

        # For now, assume all channels with small
        # enough hit time were hit.
        return event.Channels(t<1e8, t, q, self.flags.get())

class GPUDaq(object):
    def __init__(self, gpu_detector):
        self.earliest_time_gpu = ga.empty(gpu_detector.nchannels, dtype=np.float32)
        self.earliest_time_int_gpu = ga.empty(gpu_detector.nchannels, dtype=np.uint32)
        self.channel_history_gpu = ga.zeros_like(self.earliest_time_int_gpu)
        self.channel_q_int_gpu = ga.zeros_like(self.earliest_time_int_gpu)
        self.channel_q_gpu = ga.zeros(len(self.earliest_time_int_gpu), dtype=np.float32)
        self.detector_gpu = gpu_detector.detector_gpu
        self.solid_id_map_gpu = gpu_detector.solid_id_map
        self.solid_id_to_channel_index_gpu = gpu_detector.solid_id_to_channel_index_gpu

        self.module = get_cu_module('daq.cu', options=cuda_options, 
                                    include_source_directory=True)
        self.gpu_funcs = GPUFuncs(self.module)

    def acquire(self, gpuphotons, rng_states, nthreads_per_block=64, max_blocks=1024):
        self.gpu_funcs.reset_earliest_time_int(np.float32(1e9), np.int32(len(self.earliest_time_int_gpu)), self.earliest_time_int_gpu, block=(nthreads_per_block,1,1), grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1))
        self.channel_q_int_gpu.fill(0)
        self.channel_q_gpu.fill(0)
        self.channel_history_gpu.fill(0)

        n = len(gpuphotons.pos)

        for first_photon, photons_this_round, blocks in \
                chunk_iterator(n, nthreads_per_block, max_blocks):
            self.gpu_funcs.run_daq(rng_states, np.uint32(0x1 << 2), 
                                   np.int32(first_photon), np.int32(photons_this_round), gpuphotons.t, 
                                   gpuphotons.flags, gpuphotons.last_hit_triangles, 
                                   self.solid_id_map_gpu,
                                   self.detector_gpu,
                                   self.earliest_time_int_gpu, 
                                   self.channel_q_int_gpu, self.channel_history_gpu, 
                                   block=(nthreads_per_block,1,1), grid=(blocks,1))

        self.gpu_funcs.convert_sortable_int_to_float(np.int32(len(self.earliest_time_int_gpu)), self.earliest_time_int_gpu, self.earliest_time_gpu, block=(nthreads_per_block,1,1), grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1))

        self.gpu_funcs.convert_charge_int_to_float(self.detector_gpu, self.channel_q_int_gpu, self.channel_q_gpu, block=(nthreads_per_block,1,1), grid=(len(self.channel_q_int_gpu)//nthreads_per_block+1,1))

        return GPUChannels(self.earliest_time_gpu, self.channel_q_gpu, self.channel_history_gpu)

