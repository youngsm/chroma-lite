#ifndef __PHOTON_H__
#define __PHOTON_H__

#include "stdio.h"
#include "linalg.h"
#include "rotate.h"
#include "random.h"
#include "physical_constants.h"
#include "mesh.h"
#include "geometry.h"

#define WEIGHT_LOWER_THRESHOLD 0.01f

struct Photon
{
    float3 position;
    float3 direction;
    float3 polarization;
    float wavelength;
    float time;
  
    float weight;
  
    unsigned int history;

    int last_hit_triangle;
};

struct State
{
    bool inside_to_outside;

    float3 surface_normal;

    float refractive_index1, refractive_index2;
    float absorption_length;
    float scattering_length;

    int surface_index;

    float distance_to_boundary;
};

enum
{
    NO_HIT           = 0x1 << 0,
    BULK_ABSORB      = 0x1 << 1,
    SURFACE_DETECT   = 0x1 << 2,
    SURFACE_ABSORB   = 0x1 << 3,
    RAYLEIGH_SCATTER = 0x1 << 4,
    REFLECT_DIFFUSE  = 0x1 << 5,
    REFLECT_SPECULAR = 0x1 << 6,
    NAN_ABORT        = 0x1 << 31
}; // processes

enum {BREAK, CONTINUE, PASS}; // return value from propagate_to_boundary

__device__ int
convert(int c)
{
    if (c & 0x80)
	return (0xFFFFFF00 | c);
    else
	return c;
}

__device__ float
get_theta(const float3 &a, const float3 &b)
{
    return acosf(fmaxf(-1.0f,fminf(1.0f,dot(a,b))));
}

__device__ void
fill_state(State &s, Photon &p, Geometry *g)
{
    p.last_hit_triangle = intersect_mesh(p.position, p.direction, g,
					 s.distance_to_boundary,
					 p.last_hit_triangle);

    if (p.last_hit_triangle == -1) {
	p.history |= NO_HIT;
	return;
    }
    
    Triangle t = get_triangle(g, p.last_hit_triangle);
    
    unsigned int material_code = g->material_codes[p.last_hit_triangle];
    
    int inner_material_index = convert(0xFF & (material_code >> 24));
    int outer_material_index = convert(0xFF & (material_code >> 16));
    s.surface_index = convert(0xFF & (material_code >> 8));
    
    float3 v01 = t.v1 - t.v0;
    float3 v12 = t.v2 - t.v1;
    
    s.surface_normal = normalize(cross(v01, v12));
				 
    Material *material1, *material2;
    if (dot(s.surface_normal,-p.direction) > 0.0f) {
	// outside to inside
	material1 = g->materials[outer_material_index];
	material2 = g->materials[inner_material_index];

	s.inside_to_outside = false;
    }
    else {
	// inside to outside
	material1 = g->materials[inner_material_index];
	material2 = g->materials[outer_material_index];
	s.surface_normal = -s.surface_normal;

	s.inside_to_outside = true;
    }

    s.refractive_index1 = interp_property(material1, p.wavelength,
					  material1->refractive_index);
    s.refractive_index2 = interp_property(material2, p.wavelength,
					  material2->refractive_index);
    s.absorption_length = interp_property(material1, p.wavelength,
					  material1->absorption_length);
    s.scattering_length = interp_property(material1, p.wavelength,
					  material1->scattering_length);

} // fill_state

__device__ float3
pick_new_direction(float3 axis, float theta, float phi)
{
    // Taken from SNOMAN rayscatter.for
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    float cos_phi   = cosf(phi);
    float sin_phi   = sinf(phi);
	
    float sin_axis_theta = sqrt(1.0f - axis.z*axis.z);
    float cos_axis_phi, sin_axis_phi;
	
    if (isnan(sin_axis_theta) || sin_axis_theta < 0.00001f) {
	cos_axis_phi = 1.0f;
	sin_axis_phi = 0.0f;
    }
    else {
	cos_axis_phi = axis.x / sin_axis_theta;
	sin_axis_phi = axis.y / sin_axis_theta;
    }

    float dirx = cos_theta*axis.x +
	sin_theta*(axis.z*cos_phi*cos_axis_phi - sin_phi*sin_axis_phi);
    float diry = cos_theta*axis.y +
	sin_theta*(cos_phi*axis.z*sin_axis_phi - sin_phi*cos_axis_phi);
    float dirz = cos_theta*axis.z - sin_theta*cos_phi*sin_axis_theta;

    return make_float3(dirx, diry, dirz);
}

__device__ void
rayleigh_scatter(Photon &p, curandState &rng)
{
    float cos_theta = 2.0f*cosf((acosf(1.0f - 2.0f*curand_uniform(&rng))-2*PI)/3.0f);
    if (cos_theta > 1.0f)
	cos_theta = 1.0f;
    else if (cos_theta < -1.0f)
	cos_theta = -1.0f;

    float theta = acosf(cos_theta);
    float phi = uniform(&rng, 0.0f, 2.0f * PI);

    p.direction = pick_new_direction(p.polarization, theta, phi);

    if (1.0f - fabsf(cos_theta) < 1e-6f) {
	p.polarization = pick_new_direction(p.polarization, PI/2.0f, phi);
    }
    else {
	// linear combination of old polarization and new direction
	p.polarization = p.polarization - cos_theta * p.direction;
    }

    p.direction /= norm(p.direction);
    p.polarization /= norm(p.polarization);
} // scatter

__device__ int propagate_to_boundary(Photon &p, State &s, curandState &rng,
                                     bool use_weights=false, int scatter_first=0)
{
    float absorption_distance = -s.absorption_length*logf(curand_uniform(&rng));
    float scattering_distance = -s.scattering_length*logf(curand_uniform(&rng));

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD) // Prevent absorption
	absorption_distance = 1e30;
    else
	use_weights = false;

    if (scatter_first == 1) {
	// Force scatter
	float scatter_prob = 1.0f - expf(-s.distance_to_boundary/s.scattering_length);

	if (scatter_prob > WEIGHT_LOWER_THRESHOLD) {
	    int i=0;
	    const int max_i = 1000;
	    while (i < max_i && scattering_distance > s.distance_to_boundary) {
		scattering_distance = -s.scattering_length*logf(curand_uniform(&rng));
		i++;
	    }
	    p.weight *= scatter_prob;
	}

    } else if (scatter_first == -1) {
	// Prevent scatter
	float no_scatter_prob = expf(-s.distance_to_boundary/s.scattering_length);

	if (no_scatter_prob > WEIGHT_LOWER_THRESHOLD) {
	    int i=0;
	    const int max_i = 1000;
	    while (i < max_i && scattering_distance <= s.distance_to_boundary) {
		scattering_distance = -s.scattering_length*logf(curand_uniform(&rng));
		i++;
	    }
	    p.weight *= no_scatter_prob;
	}
    }

    if (absorption_distance <= scattering_distance) {
	if (absorption_distance <= s.distance_to_boundary) {
	    p.time += absorption_distance/(SPEED_OF_LIGHT/s.refractive_index1);
	    p.position += absorption_distance*p.direction;
	    p.history |= BULK_ABSORB;

	    p.last_hit_triangle = -1;

	    return BREAK;
	} // photon is absorbed in material1
    }
    else {
	if (scattering_distance <= s.distance_to_boundary) {

	    // Scale weight by absorption probability along this distance
	    if (use_weights)
		p.weight *= expf(-scattering_distance/s.absorption_length);

	    p.time += scattering_distance/(SPEED_OF_LIGHT/s.refractive_index1);
	    p.position += scattering_distance*p.direction;

	    rayleigh_scatter(p, rng);

	    p.history |= RAYLEIGH_SCATTER;

	    p.last_hit_triangle = -1;

	    return CONTINUE;
	} // photon is scattered in material1
    } // if scattering_distance < absorption_distance

    // Scale weight by absorption probability along this distance
    if (use_weights)
	p.weight *= expf(-s.distance_to_boundary/s.absorption_length);
    
    p.position += s.distance_to_boundary*p.direction;
    p.time += s.distance_to_boundary/(SPEED_OF_LIGHT/s.refractive_index1);

    return PASS;

} // propagate_to_boundary

__device__ void
propagate_at_boundary(Photon &p, State &s, curandState &rng)
{
    float incident_angle = get_theta(s.surface_normal,-p.direction);
    float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);

    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    float incident_plane_normal_length = norm(incident_plane_normal);

    // Photons at normal incidence do not have a unique plane of incidence,
    // so we have to pick the plane normal to be the polarization vector
    // to get the correct logic below
    if (incident_plane_normal_length < 1e-6f)
	incident_plane_normal = p.polarization;
    else
	incident_plane_normal /= incident_plane_normal_length;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);
    float normal_probability = normal_coefficient*normal_coefficient;

    float reflection_coefficient;
    if (curand_uniform(&rng) < normal_probability) {
	// photon polarization normal to plane of incidence
	reflection_coefficient = -sinf(incident_angle-refracted_angle)/sinf(incident_angle+refracted_angle);

	if ((curand_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) {
	    p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
			
	    p.history |= REFLECT_SPECULAR;
	}
	else {
	    p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
	}

	p.polarization = incident_plane_normal;
    }
    else {
	// photon polarization parallel to plane of incidence
	reflection_coefficient = tanf(incident_angle-refracted_angle)/tanf(incident_angle+refracted_angle);

	if ((curand_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) {
	    p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
			
	    p.history |= REFLECT_SPECULAR;
	}
	else {
	    p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
	}

	p.polarization = cross(incident_plane_normal, p.direction);
	p.polarization /= norm(p.polarization);
    }

} // propagate_at_boundary

__device__ int
propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
                     bool use_weights=false)
{
    Surface *surface = geometry->surfaces[s.surface_index];

    /* since the surface properties are interpolated linearly, we are
       guaranteed that they still sum to 1.0. */

    float detect = interp_property(surface, p.wavelength, surface->detect);
    float absorb = interp_property(surface, p.wavelength, surface->absorb);
    float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);

    float uniform_sample = curand_uniform(&rng);

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD 
	&& absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
	// Prevent absorption and reweight accordingly
	float survive = 1.0f - absorb;
	absorb = 0.0f;
	p.weight *= survive;

	// Renormalize remaining probabilities
	detect /= survive;
	reflect_diffuse /= survive;
	reflect_specular /= survive;
    }

    if (use_weights && detect > 0.0f) {
	p.history |= SURFACE_DETECT;
	p.weight *= detect;
	return BREAK;
    }

    if (uniform_sample < absorb) {
	p.history |= SURFACE_ABSORB;
	return BREAK;
    }
    else if (uniform_sample < absorb + detect) {
	p.history |= SURFACE_DETECT;
	return BREAK;
    }
    else if (uniform_sample < absorb + detect + reflect_diffuse) {
	// diffusely reflect
	p.direction = uniform_sphere(&rng);

	if (dot(p.direction, s.surface_normal) < 0.0f)
	    p.direction = -p.direction;

	// randomize polarization?
	p.polarization = cross(uniform_sphere(&rng), p.direction);
	p.polarization /= norm(p.polarization);

	p.history |= REFLECT_DIFFUSE;

	return CONTINUE;
    }
    else {
	// specularly reflect
	float incident_angle = get_theta(s.surface_normal,-p.direction);
	float3 incident_plane_normal = cross(p.direction, s.surface_normal);
	incident_plane_normal /= norm(incident_plane_normal);

	p.direction = rotate(s.surface_normal, incident_angle,
			     incident_plane_normal);

	p.history |= REFLECT_SPECULAR;

	return CONTINUE;
    }

} // propagate_at_surface

#endif
