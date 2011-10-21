//-*-c-*-

#include "linalg.h"
#include "geometry.h"
#include "photon.h"

#include "stdio.h"

extern "C"
{

__global__ void
photon_duplicate(int first_photon, int nthreads,
		 float3 *positions, float3 *directions,
		 float *wavelengths, float3 *polarizations,
		 float *times, unsigned int *histories,
		 int *last_hit_triangles, 
		 int copies, int stride)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    int photon_id = first_photon + id;

    Photon p;
    p.position = positions[photon_id];
    p.direction = directions[photon_id];
    p.polarization = polarizations[photon_id];
    p.wavelength = wavelengths[photon_id];
    p.time = times[photon_id];
    p.last_hit_triangle = last_hit_triangles[photon_id];
    p.history = histories[photon_id];

    for (int i=1; i <= copies; i++) {
      int target_photon_id = photon_id + stride * i;

      positions[target_photon_id] = p.position;
      directions[target_photon_id] = p.direction;
      polarizations[target_photon_id] = p.polarization;
      wavelengths[target_photon_id] = p.wavelength;
      times[target_photon_id] = p.time;
      last_hit_triangles[target_photon_id] = p.last_hit_triangle;
      histories[target_photon_id] = p.history;
    }
}

__global__ void
count_photons(int first_photon, int nthreads, unsigned int target_flag,
	      unsigned int *index_counter,
	      unsigned int *histories)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ unsigned int counter;

    if (threadIdx.x == 0)
	counter = 0;
    __syncthreads();

    if (id < nthreads) {
	int photon_id = first_photon + id;

	if (histories[photon_id] & target_flag) {
	    atomicAdd(&counter, 1);
	}
	    
    }

    __syncthreads();

    if (threadIdx.x == 0)
	atomicAdd(index_counter, counter);
}

__global__ void
copy_photons(int first_photon, int nthreads, unsigned int target_flag,
	     unsigned int *index_counter,
	     float3 *positions, float3 *directions,
	     float *wavelengths, float3 *polarizations,
	     float *times, unsigned int *histories,
	     int *last_hit_triangles, 
	     float3 *new_positions, float3 *new_directions,
	     float *new_wavelengths, float3 *new_polarizations,
	     float *new_times, unsigned int *new_histories,
	     int *new_last_hit_triangles)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (id >= nthreads)
	return;
    
    int photon_id = first_photon + id;

    if (histories[photon_id] & target_flag) {
	int offset = atomicAdd(index_counter, 1);

	new_positions[offset] = positions[photon_id];
	new_directions[offset] = directions[photon_id];
	new_polarizations[offset] = polarizations[photon_id];
	new_wavelengths[offset] = wavelengths[photon_id];
	new_times[offset] = times[photon_id];
	new_histories[offset] = histories[photon_id];
	new_last_hit_triangles[offset] = last_hit_triangles[photon_id];
    }
}

	      
__global__ void
propagate(int first_photon, int nthreads, unsigned int *input_queue,
	  unsigned int *output_queue, curandState *rng_states,
	  float3 *positions, float3 *directions,
	  float *wavelengths, float3 *polarizations,
	  float *times, unsigned int *histories,
	  int *last_hit_triangles, int max_steps,
	  Geometry *g)
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
	sg = *g;

    __syncthreads();

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    g = &sg;

    curandState rng = rng_states[id];

    int photon_id = input_queue[first_photon + id];

    Photon p;
    p.position = positions[photon_id];
    p.direction = directions[photon_id];
    p.direction /= norm(p.direction);
    p.polarization = polarizations[photon_id];
    p.polarization /= norm(p.polarization);
    p.wavelength = wavelengths[photon_id];
    p.time = times[photon_id];
    p.last_hit_triangle = last_hit_triangles[photon_id];
    p.history = histories[photon_id];

    if (p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB))
	return;

    State s;

    int steps = 0;
    while (steps < max_steps) {
	steps++;

	int command;

	// check for NaN and fail
	if (isnan(p.direction.x*p.direction.y*p.direction.z*p.position.x*p.position.y*p.position.z)) {
	    p.history |= NO_HIT | NAN_ABORT;
	    break;
	}

	fill_state(s, p, g);

	if (p.last_hit_triangle == -1)
	    break;

	command = propagate_to_boundary(p, s, rng);

	if (command == BREAK)
	    break;

	if (command == CONTINUE)
	    continue;

	if (s.surface_index != -1) {
	    command = propagate_at_surface(p, s, rng, g);

	    if (command == BREAK)
		break;

	    if (command == CONTINUE)
		continue;
	}

	propagate_at_boundary(p, s, rng);

    } // while (steps < max_steps)

    rng_states[id] = rng;
    positions[photon_id] = p.position;
    directions[photon_id] = p.direction;
    polarizations[photon_id] = p.polarization;
    wavelengths[photon_id] = p.wavelength;
    times[photon_id] = p.time;
    histories[photon_id] = p.history;
    last_hit_triangles[photon_id] = p.last_hit_triangle;
	
    // Not done, put photon in output queue
    if ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB)) == 0) {
	int out_idx = atomicAdd(output_queue, 1);
	output_queue[out_idx] = photon_id;
    }
} // propagate

} // extern "C"
