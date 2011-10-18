// -*-c++-*-
#include <curand_kernel.h>

extern "C"
{

__global__ void
bin_hits(int nchannels, float *channel_q, float *channel_time,
	 unsigned int *hitcount, int tbins, float tmin, float tmax, int qbins,
	 float qmin, float qmax, unsigned int *pdf)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id >= nchannels)
	return;

    unsigned int q = channel_q[id];
    float t = channel_time[id];
	
    if (t < 1e8 && t >= tmin && t < tmax && q >= qmin && q < qmax) {
	hitcount[id] += 1;
		
	int tbin = (t - tmin) / (tmax - tmin) * tbins;
	int qbin = (q - qmin) / (qmax - qmin) * qbins;
	    
	// row major order (channel, t, q)
	int bin = id * (tbins * qbins) + tbin * qbins + qbin;
	pdf[bin] += 1;
    }
}
	
__global__ void
accumulate_pdf_eval(int time_only, int nchannels, unsigned int *event_hit,
		    float *event_time, float *event_charge, float *mc_time,
		    float *mc_charge, // quirk of DAQ!
		    unsigned int *hitcount, unsigned int *bincount,
		    float min_twidth, float tmin, float tmax,
		    float min_qwidth, float qmin, float qmax,
		    float *nearest_mc, int min_bin_content)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
	
    if (id >= nchannels)
	return;
	
    // Was this channel hit in the Monte Carlo?
    float channel_mc_time = mc_time[id];
    if (channel_mc_time >= 1e8)
	return; // Nothing else to do
	
    // Is this channel inside the range of the PDF?
    float distance;
    int channel_bincount = bincount[id];
    if (time_only) {
	if (channel_mc_time < tmin || channel_mc_time > tmax)
	    return;  // Nothing else to do
		
	// This MC information is contained in the PDF
	hitcount[id] += 1;
	  
	// Was this channel hit in the event-of-interest?
	int channel_event_hit = event_hit[id];
	if (!channel_event_hit)
	    return; // No need to update PDF value for unhit channel
		
	// Are we inside the minimum size bin?
	float channel_event_time = event_time[id];
	distance = fabsf(channel_mc_time - channel_event_time);
	if (distance < min_twidth/2.0)
	    bincount[id] = channel_bincount + 1;

    }
    else { // time and charge PDF
	float channel_mc_charge = mc_charge[id]; // int->float conversion because DAQ just returns an integer
	
	if (channel_mc_time < tmin || channel_mc_time > tmax ||
	    channel_mc_charge < qmin || channel_mc_charge > qmax)
	    return;  // Nothing else to do
		
	// This MC information is contained in the PDF
	hitcount[id] += 1;
	  
	// Was this channel hit in the event-of-interest?
	int channel_event_hit = event_hit[id];
	if (!channel_event_hit)
	    return; // No need to update PDF value for unhit channel
		
	// Are we inside the minimum size bin?
	float channel_event_time = event_time[id];
	float channel_event_charge = event_charge[id];
	float normalized_time_distance = fabsf(channel_event_time - channel_mc_time)/min_twidth/2.0;
	float normalized_charge_distance = fabsf(channel_event_charge - channel_mc_charge)/min_qwidth/2.0;
	distance = sqrt(normalized_time_distance*normalized_time_distance + normalized_charge_distance*normalized_charge_distance);

	if (distance < 1.0f)
	    bincount[id] = channel_bincount + 1;
    }

    // Do we need to keep updating the nearest_mc list?
    if (channel_bincount >= min_bin_content)
      return; // No need to perform insertion into nearest_mc because bincount is a better estimate of the PDF value

    // insertion sort the distance into the array of nearest MC points
    int offset = min_bin_content * id;
    int entry = min_bin_content - 1;
	
    // If last entry less than new entry, nothing to update
    if (distance > nearest_mc[offset + entry])
	return;

    // Find where to insert the new entry while sliding the rest
    // to the right
    entry--;
    while (entry >= 0) {
	if (nearest_mc[offset+entry] >= distance)
	    nearest_mc[offset+entry+1] = nearest_mc[offset+entry];
	else
	    break;
	entry--;
    }

    nearest_mc[offset+entry+1] = distance;
}


__global__ void
accumulate_moments(int time_only, int nchannels,
		   float *mc_time,
		   float *mc_charge,
		   float tmin, float tmax,
		   float qmin, float qmax,
		   unsigned int *mom0,
		   float *t_mom1,
		   float *t_mom2,
		   float *q_mom1,
		   float *q_mom2)

{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
	
    if (id >= nchannels)
	return;
	
    // Was this channel hit in the Monte Carlo?
    float channel_mc_time = mc_time[id];

    // Is this channel inside the range of the PDF?
    if (time_only) {
	if (channel_mc_time < tmin || channel_mc_time > tmax)
	    return;  // Nothing else to do
	
	mom0[id] += 1;
	t_mom1[id] += channel_mc_time;
	t_mom2[id] += channel_mc_time*channel_mc_time;
    }
    else { // time and charge PDF
	float channel_mc_charge = mc_charge[id]; // int->float conversion because DAQ just returns an integer
	
	if (channel_mc_time < tmin || channel_mc_time > tmax ||
	    channel_mc_charge < qmin || channel_mc_charge > qmax)
	    return;  // Nothing else to do

	mom0[id] += 1;
	t_mom1[id] += channel_mc_time;
	t_mom2[id] += channel_mc_time*channel_mc_time;
	q_mom1[id] += channel_mc_charge;
	q_mom2[id] += channel_mc_charge*channel_mc_charge;
    }
}

static const float invroot2 = 0.70710678118654746f; // 1/sqrt(2)
static const float rootPiBy2 = 1.2533141373155001f; // sqrt(M_PI/2)

__global__ void
accumulate_kernel_eval(int time_only, int nchannels, unsigned int *event_hit,
		       float *event_time, float *event_charge, float *mc_time,
		       float *mc_charge,
		       float tmin, float tmax,
		       float qmin, float qmax,
		       float *inv_time_bandwidths,
		       float *inv_charge_bandwidths,
		       unsigned int *hitcount,
		       float *time_pdf_values,
		       float *charge_pdf_values)
		       
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
	
    if (id >= nchannels)
	return;
	
    // Was this channel hit in the Monte Carlo?
    float channel_mc_time = mc_time[id];
	
    // Is this channel inside the range of the PDF?
    if (time_only) {
	if (channel_mc_time < tmin || channel_mc_time > tmax)
	    return;  // Nothing else to do
		
	// This MC information is contained in the PDF
	hitcount[id] += 1;
	  
	// Was this channel hit in the event-of-interest?
	int channel_event_hit = event_hit[id];
	if (!channel_event_hit)
	    return; // No need to update PDF value for unhit channel

	// Kernel argument
	float channel_event_time = event_time[id];
	float inv_bandwidth = inv_time_bandwidths[id];
	float arg = (channel_mc_time - channel_event_time) * inv_bandwidth;

	// evaluate 1D Gaussian normalized within time window
	float term = expf(-0.5f * arg * arg) * inv_bandwidth;

	float norm = tmax - tmin;
	if (inv_bandwidth > 0.0f) {
	  float loarg = (tmin - channel_mc_time)*inv_bandwidth*invroot2;
	  float hiarg = (tmax - channel_mc_time)*inv_bandwidth*invroot2;
	  norm = (erff(hiarg) - erff(loarg)) * rootPiBy2;
	}
	time_pdf_values[id] += term / norm;
    }
    else { // time and charge PDF
	float channel_mc_charge = mc_charge[id]; // int->float conversion because DAQ just returns an integer
	
	if (channel_mc_time < tmin || channel_mc_time > tmax ||
	    channel_mc_charge < qmin || channel_mc_charge > qmax)
	    return;  // Nothing else to do
		
	// This MC information is contained in the PDF
	hitcount[id] += 1;
	  
	// Was this channel hit in the event-of-interest?
	int channel_event_hit = event_hit[id];
	if (!channel_event_hit)
	    return; // No need to update PDF value for unhit channel


	// Kernel argument: time dim
	float channel_event_obs = event_time[id];
	float inv_bandwidth = inv_time_bandwidths[id];
	float arg = (channel_mc_time - channel_event_obs) * inv_bandwidth;

	float norm = tmax - tmin;
	if (inv_bandwidth > 0.0f) {
	  float loarg = (tmin - channel_mc_time)*inv_bandwidth*invroot2;
	  float hiarg = (tmax - channel_mc_time)*inv_bandwidth*invroot2;
	  norm = (erff(hiarg) - erff(loarg)) * rootPiBy2;
	}
	float term = expf(-0.5f * arg * arg);

	time_pdf_values[id] += term / norm;

	// Kernel argument: charge dim
	channel_event_obs = event_charge[id];
	inv_bandwidth = inv_charge_bandwidths[id];
	arg = (channel_mc_charge - channel_event_obs) * inv_bandwidth;
	
	norm = qmax - qmin;
	if (inv_bandwidth > 0.0f) {
	  float loarg = (qmin - channel_mc_charge)*inv_bandwidth*invroot2;
	  float hiarg = (qmax - channel_mc_charge)*inv_bandwidth*invroot2;
	  norm = (erff(hiarg) - erff(loarg)) * rootPiBy2;
	}

	term = expf(-0.5f * arg * arg);

	charge_pdf_values[id] += term / norm;
    }
}


} // extern "C"
