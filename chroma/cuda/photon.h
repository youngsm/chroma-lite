#ifndef __PHOTON_H__
#define __PHOTON_H__

#include "stdio.h"
#include "linalg.h"
#include "rotate.h"
#include "random.h"
#include "physical_constants.h"
#include "mesh.h"
#include "geometry.h"
#include "cx.h"

#define WEIGHT_LOWER_THRESHOLD 0.0001f

// ---- Device-side debug counters (classification + material1 index) ----
extern "C" __device__ unsigned long long chroma_dbg_mesh_dir_counts[2];      // [0]=outside->inside, [1]=inside->outside
extern "C" __device__ unsigned long long chroma_dbg_mesh_mat1_counts[256];   // index of chosen material1 (pre-boundary)
extern "C" __device__ unsigned long long chroma_dbg_wp_dir_counts[2];        // analytic wireplane
extern "C" __device__ unsigned long long chroma_dbg_wp_mat1_counts[256];
extern "C" __device__ unsigned long long chroma_dbg_mesh_mat1_by_dir[2][256];
extern "C" __device__ unsigned long long chroma_dbg_wp_mat1_by_dir[2][256];

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
    
    unsigned int evidx;
};

struct State
{
    bool inside_to_outside;

    float3 surface_normal;

    float refractive_index1, refractive_index2;
    float absorption_length;
    float scattering_length;
    float reemission_prob;
    Material *material1;

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
    SURFACE_REEMIT   = 0x1 << 7,
    SURFACE_TRANSMIT = 0x1 << 8,
    BULK_REEMIT      = 0x1 << 9,
    CHERENKOV        = 0x1 << 10,
    SCINTILLATION    = 0x1 << 11,
    NAN_ABORT        = 0x1 << 31
}; // processes

enum { BREAK, CONTINUE, PASS }; // return value from propagate_to_boundary

// Interpret 8-bit packed indices from material_code.
// Only 0xFF is the sentinel for -1; all other values [0,254] are valid non-negative indices.
__device__ int
convert(int c)
{
    return (c == 0xFF) ? -1 : c;
}

__device__ float
get_theta(const float3 &a, const float3 &b)
{
    return acosf(fmaxf(-1.0f,fminf(1.0f,dot(a,b))));
}

// compute fast analytic intersection with a periodic array of parallel cylinders (wire plane)
// returns true if a valid hit closer than max_distance is found, and fills t_hit, normal
__device__ bool intersect_wireplane(const float3 &origin, const float3 &direction,
                                    const WirePlane &wp, const float max_distance,
                                    float &t_hit, float3 &normal)
{
    // orthonormal frame: u (wire axis), v (in-plane perp to wires), n = u x v
    float3 u = normalize(wp.u);
    float3 v = normalize(wp.v);
    float3 n = normalize(cross(u, v));

    // local ray components
    float3 w = origin - wp.origin;
    float du = dot(direction, u);
    float dv = dot(direction, v);
    float dn = dot(direction, n);
    float wu = dot(w, u);
    float wv0 = dot(w, v) - wp.v0; // v relative to offset
    float wn0 = dot(w, n);

    // quadratic coeffs in (v,n) plane
    float A = dv*dv + dn*dn;
    if (A < 1e-12f) // ray effectively parallel to wire axis within numerical tolerance
        return false;

    // valid k range by plane bounds (centerlines)
    float kmin_f = (wp.vmin - wp.v0) / wp.pitch;
    float kmax_f = (wp.vmax - wp.v0) / wp.pitch;
    int kmin = (int)ceilf(kmin_f);
    int kmax = (int)floorf(kmax_f);

    // nearest axis index to current v position
    int k0 = (int)floorf(wv0 / wp.pitch + 0.5f);
    if (k0 < kmin) k0 = kmin;
    if (k0 > kmax) k0 = kmax;

    float best_t = max_distance;
    bool hit = false;
    float3 best_normal = make_float3(0.f,0.f,0.f);

    // check a small neighborhood of axes around k0 to catch first crossing
    // this is sufficient bc solutions repeat with period |pitch/dv| along t
    for (int dk = -1; dk <= 1; ++dk) {
        int k = k0 + dk;
        if (k < kmin || k > kmax) continue;
        float wv = wv0 - (float)k * wp.pitch;

        // quadratic: A t^2 + 2*(wv*dv + wn0*dn) t + (wv^2 + wn0^2 - r^2) = 0
        float B = 2.0f * (wv*dv + wn0*dn);
        float C = wv*wv + wn0*wn0 - wp.radius*wp.radius;
        float disc = B*B - 4.0f*A*C;
        if (disc < 0.0f) continue;

        float sqrt_disc = sqrtf(fmaxf(0.0f, disc));
        // two roots, pick smallest positive larger than epsilon and within current limit
        float t1 = (-B - sqrt_disc) / (2.0f*A);
        float t2 = (-B + sqrt_disc) / (2.0f*A);

        // evaluate both candidates
        for (int root = 0; root < 2; ++root) {
            float t = (root == 0) ? t1 : t2;
            if (!(t > CHROMA_EPSILON)) continue; // must move forward
            if (t >= best_t) continue;           // must improve

            // check u-bound limits
            float uc = wu + du * t;
            if (uc < wp.umin - 1e-6f || uc > wp.umax + 1e-6f) continue;

            // normal at hit in (v,n) plane: r = (wv + t*dv) v + (wn0 + t*dn) n
            float rv = wv + t*dv;
            float rn = wn0 + t*dn;
            float inv_len = rsqrtf(rv*rv + rn*rn);
            float3 nrm = rv*inv_len * v + rn*inv_len * n;

            best_t = t;
            best_normal = nrm; // outward (radial)
            hit = true;
        }
    }

    if (hit) {
        t_hit = best_t;
        normal = best_normal;
    }
    return hit;
}

__device__ void
fill_state(State &s, Photon &p, Geometry *g)
{
    // default
    s.distance_to_boundary = CHROMA_INFINITY;
    int mesh_triangle = -1;

    float mesh_distance = -1.0f;
    mesh_triangle = intersect_mesh(p.position, p.direction, g,
                                   mesh_distance,
                                   p.last_hit_triangle);
    if (mesh_triangle != -1) {
        s.distance_to_boundary = mesh_distance;
    }

    int which_wireplane = -1;
    float3 wp_normal = make_float3(0.f,0.f,0.f);
    if (g->nwireplanes > 0) {
        for (int i = 0; i < g->nwireplanes; ++i) {
            float t_hit;
            float3 nrm;
            if (intersect_wireplane(p.position, p.direction, g->wireplanes[i], s.distance_to_boundary, t_hit, nrm)) {
                // Prefer the closer of mesh and wireplane
                if (t_hit < s.distance_to_boundary) {
                    s.distance_to_boundary = t_hit;
                    which_wireplane = i;
                    wp_normal = nrm;
                }
            }
        }
    }

    // hit analytic wire plane: fill optical state using stored indices
    if (which_wireplane >= 0) {
        const WirePlane &wp = g->wireplanes[which_wireplane];
        s.surface_index = wp.surface_index;
        s.surface_normal = wp_normal;

        Material *material1, *material2;

        // compute current radial distance to nearest wire axis
        float3 u = normalize(wp.u);
        float3 v = normalize(wp.v);
        float3 n = normalize(cross(u, v));
        float3 wcur = p.position - wp.origin;
        float wv0 = dot(wcur, v) - wp.v0;
        float wn0 = dot(wcur, n);
        int k0 = (int)floorf(wv0 / wp.pitch + 0.5f);
        float rv = wv0 - (float)k0 * wp.pitch;
        float rn = wn0;
        float rnow = sqrtf(rv*rv + rn*rn);
        const float side_eps = 1e-6f;
        bool in_outer = (rnow >= (wp.radius - side_eps));
        material1 = g->materials[in_outer ? wp.material_outer_index : wp.material_inner_index];
        material2 = g->materials[in_outer ? wp.material_inner_index : wp.material_outer_index];
        s.inside_to_outside = !in_outer;
        // orient normal to face incoming photon
        if (dot(s.surface_normal, -p.direction) < 0.0f)
            s.surface_normal = -s.surface_normal;
        bool outside_to_inside = in_outer; // for debug counters

        s.refractive_index1 = interp_property(material1, p.wavelength,
                                              material1->refractive_index);
        s.refractive_index2 = interp_property(material2, p.wavelength,
                                              material2->refractive_index);
        s.absorption_length = interp_property(material1, p.wavelength,
                                              material1->absorption_length);
        s.scattering_length = interp_property(material1, p.wavelength,
                                             material1->scattering_length);
        s.reemission_prob = 0.0f; // overridden later if needed
        s.material1 = material1;

        // debug counters for analytic wireplanes
        atomicAdd(&chroma_dbg_wp_dir_counts[outside_to_inside ? 0 : 1], 1ULL);
        int m1idx_wp = outside_to_inside ? wp.material_outer_index : wp.material_inner_index;
        if (m1idx_wp < 0 || m1idx_wp > 255) m1idx_wp = 255; // bucket invalid
        atomicAdd(&chroma_dbg_wp_mat1_counts[m1idx_wp], 1ULL);
        atomicAdd(&chroma_dbg_wp_mat1_by_dir[outside_to_inside ? 0 : 1][m1idx_wp], 1ULL);

        // encode a negative last_hit_triangle to mark analytic surface hits and avoid false triangle reuse
        p.last_hit_triangle = -2 - which_wireplane;
        return;
    }

    // no wire-plane analytic hit was closer; fall back to mesh-only path
    p.last_hit_triangle = mesh_triangle;

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
    // robust classification for mesh triangles: plane-side test
    float side = dot(s.surface_normal, p.position - t.v0);
    const float side_eps2 = 1e-6f;
    bool in_outer = (side >= -side_eps2);
    material1 = g->materials[in_outer ? outer_material_index : inner_material_index];
    material2 = g->materials[in_outer ? inner_material_index : outer_material_index];
    s.inside_to_outside = !in_outer;
    if (dot(s.surface_normal, -p.direction) < 0.0f)
        s.surface_normal = -s.surface_normal;
    bool outside_to_inside = in_outer; // for debug counters

    s.refractive_index1 = interp_property(material1, p.wavelength,
                                          material1->refractive_index);
    s.refractive_index2 = interp_property(material2, p.wavelength,
                                          material2->refractive_index);
    s.absorption_length = interp_property(material1, p.wavelength,
                                          material1->absorption_length);
    s.scattering_length = interp_property(material1, p.wavelength,
                                          material1->scattering_length);

    s.material1 = material1;

    // debug counters for mesh triangles
    atomicAdd(&chroma_dbg_mesh_dir_counts[outside_to_inside ? 0 : 1], 1ULL);
    int m1idx_mesh = outside_to_inside ? outer_material_index : inner_material_index;
    if (m1idx_mesh < 0 || m1idx_mesh > 255) m1idx_mesh = 255; // bucket invalid
    atomicAdd(&chroma_dbg_mesh_mat1_counts[m1idx_mesh], 1ULL);
    atomicAdd(&chroma_dbg_mesh_mat1_by_dir[outside_to_inside ? 0 : 1][m1idx_mesh], 1ULL);
} // fill_state

__device__ float3
pick_new_direction(float3 axis, float theta, float phi)
{
    // Taken from SNOMAN rayscatter.for
    float cos_theta, sin_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    float cos_phi, sin_phi;
    sincosf(phi, &sin_phi, &cos_phi);
        
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
        sin_theta*(cos_phi*axis.z*sin_axis_phi + sin_phi*cos_axis_phi);
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

__device__
int propagate_to_boundary(Photon &p, State &s, curandState &rng,
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

    // possible refactor? TODO: check if this is correct
    // if (scatter_first == 1) {
    //     float d = s.distance_to_boundary;
    //     float invL = 1.0f / s.scattering_length;
    //     float e = expf(-d * invL);
    //     float p = 1.0f - e;                       // P(scatter before boundary)
    //     if (p > WEIGHT_LOWER_THRESHOLD) {
    //         float u = curand_uniform(&rng);       // in (0,1]
    //         // x = -λ * log(1 - u * p)  ; use log1p for stability
    //         scattering_distance = -s.scattering_length * log1pf(-u * p);
    //         p.weight *= p;
    //     }
    // } else if (scatter_first == -1) {
    //     float d = s.distance_to_boundary;
    //     float invL = 1.0f / s.scattering_length;
    //     float p0 = expf(-d * invL);               // P(no scatter before boundary)
    //     if (p0 > WEIGHT_LOWER_THRESHOLD) {
    //         float u = curand_uniform(&rng);       // in (0,1]
    //         // x = d + Exp(λ) = d - λ * log u
    //         scattering_distance = d - s.scattering_length * logf(u);
    //         p.weight *= p0;
    //     }
    // }


    if (absorption_distance <= scattering_distance) {
        if (absorption_distance <= s.distance_to_boundary) {
            p.time += absorption_distance/(SPEED_OF_LIGHT/s.refractive_index1);
            p.position += absorption_distance*p.direction;
            
            if (s.material1->num_comp == 0) {
                p.last_hit_triangle = -1;
                p.history |= BULK_ABSORB;
                return BREAK;
            }
            
            float uniform_sample_comp = curand_uniform(&rng);
            float prob = 0.0f;
            int comp;
            for (comp = 0; ; comp++) {
                float comp_abs = interp_property(s.material1, p.wavelength, s.material1->comp_absorption_length[comp]);
                prob += s.absorption_length/comp_abs;
                if (uniform_sample_comp < prob or comp+1 == s.material1->num_comp) break;
            }
            
            float uniform_sample_reemit = curand_uniform(&rng);
            float comp_reemit_prob = interp_property(s.material1, p.wavelength, s.material1->comp_reemission_prob[comp]);
            if (uniform_sample_reemit < comp_reemit_prob) {
                p.wavelength = sample_cdf(&rng, s.material1->wavelength_n, 
                                          s.material1->wavelength_start,
                                          s.material1->wavelength_step,
                                          s.material1->comp_reemission_wvl_cdf[comp]);
                p.time += sample_cdf(&rng, s.material1->time_n, 
                                          s.material1->time_start,
                                          s.material1->time_step,
                                          s.material1->comp_reemission_time_cdf[comp]);
                p.direction = uniform_sphere(&rng);
                p.polarization = cross(uniform_sphere(&rng), p.direction);
                p.polarization /= norm(p.polarization);
                p.last_hit_triangle = -1;
                p.history |= BULK_REEMIT;
                return CONTINUE;
            } // photon is reemitted isotropically
            else {
                p.last_hit_triangle = -1;
                p.history |= BULK_ABSORB;
                return BREAK;
            } // photon is absorbed in material1
        }
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

__noinline__ __device__ void
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
propagate_at_specular_reflector(Photon &p, State &s)
{
    float incident_angle = get_theta(s.surface_normal, -p.direction);
    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    incident_plane_normal /= norm(incident_plane_normal);

    p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);

    p.history |= REFLECT_SPECULAR;

    return CONTINUE;
} // propagate_at_specular_reflector

__device__ int
propagate_at_diffuse_reflector(Photon &p, State &s, curandState &rng)
{
    float ndotv;
    do {
        p.direction = uniform_sphere(&rng);
        ndotv = dot(p.direction, s.surface_normal);
        if (ndotv < 0.0f) {
            p.direction = -p.direction;
            ndotv = -ndotv;
        }
    } while (! (curand_uniform(&rng) < ndotv) );

    p.polarization = cross(uniform_sphere(&rng), p.direction);
    p.polarization /= norm(p.polarization);

    p.history |= REFLECT_DIFFUSE;

    return CONTINUE;
} // propagate_at_diffuse_reflector

__device__ int
propagate_complex(Photon &p, State &s, curandState &rng, Surface* surface, bool use_weights=false)
{
    float detect = interp_property(surface, p.wavelength, surface->detect);
    float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    float n2_eta = interp_property(surface, p.wavelength, surface->eta);
    float n2_k = interp_property(surface, p.wavelength, surface->k);

    // thin film optical model, adapted from RAT PMT optical model by P. Jones
    cuFloatComplex n1 = make_cuFloatComplex(s.refractive_index1, 0.0f);
    cuFloatComplex n2 = make_cuFloatComplex(n2_eta, n2_k);
    cuFloatComplex n3 = make_cuFloatComplex(s.refractive_index2, 0.0f);

    float cos_t1 = dot(p.direction, s.surface_normal);
    if (cos_t1 < 0.0f)
        cos_t1 = -cos_t1;
    float theta = acosf(cos_t1);

    cuFloatComplex cos1 = make_cuFloatComplex(cosf(theta), 0.0f);
    cuFloatComplex sin1 = make_cuFloatComplex(sinf(theta), 0.0f);

    float e = 2.0f * PI * surface->thickness / p.wavelength;
    cuFloatComplex ratio13sin = cuCmulf(cuCmulf(cuCdivf(n1, n3), cuCdivf(n1, n3)), cuCmulf(sin1, sin1));
    cuFloatComplex cos3 = cuCsqrtf(cuCsubf(make_cuFloatComplex(1.0f,0.0f), ratio13sin));
    cuFloatComplex ratio12sin = cuCmulf(cuCmulf(cuCdivf(n1, n2), cuCdivf(n1, n2)), cuCmulf(sin1, sin1));
    cuFloatComplex cos2 = cuCsqrtf(cuCsubf(make_cuFloatComplex(1.0f,0.0f), ratio12sin));
    float u = cuCrealf(cuCmulf(n2, cos2));
    float v = cuCimagf(cuCmulf(n2, cos2));

    // s polarization
    cuFloatComplex s_n1c1 = cuCmulf(n1, cos1);
    cuFloatComplex s_n2c2 = cuCmulf(n2, cos2);
    cuFloatComplex s_n3c3 = cuCmulf(n3, cos3);
    cuFloatComplex s_r12 = cuCdivf(cuCsubf(s_n1c1, s_n2c2), cuCaddf(s_n1c1, s_n2c2));
    cuFloatComplex s_r23 = cuCdivf(cuCsubf(s_n2c2, s_n3c3), cuCaddf(s_n2c2, s_n3c3));
    cuFloatComplex s_t12 = cuCdivf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), s_n1c1), cuCaddf(s_n1c1, s_n2c2));
    cuFloatComplex s_t23 = cuCdivf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), s_n2c2), cuCaddf(s_n2c2, s_n3c3));
    cuFloatComplex s_g = cuCdivf(s_n3c3, s_n1c1);

    float s_abs_r12 = cuCabsf(s_r12);
    float s_abs_r23 = cuCabsf(s_r23);
    float s_abs_t12 = cuCabsf(s_t12);
    float s_abs_t23 = cuCabsf(s_t23);
    float s_arg_r12 = cuCargf(s_r12);
    float s_arg_r23 = cuCargf(s_r23);
    float s_exp1 = exp(2.0f * v * e);

    float s_exp2 = 1.0f / s_exp1;
    float s_denom = s_exp1 +
                    s_abs_r12 * s_abs_r12 * s_abs_r23 * s_abs_r23 * s_exp2 +
                    2.0f * s_abs_r12 * s_abs_r23 * cosf(s_arg_r23 + s_arg_r12 + 2.0f * u * e);

    float s_r = s_abs_r12 * s_abs_r12 * s_exp1 + s_abs_r23 * s_abs_r23 * s_exp2 +
                2.0f * s_abs_r12 * s_abs_r23 * cosf(s_arg_r23 - s_arg_r12 + 2.0f * u * e);
    s_r /= s_denom;

    float s_t = cuCrealf(s_g) * s_abs_t12 * s_abs_t12 * s_abs_t23 * s_abs_t23;
    s_t /= s_denom;

    // p polarization
    cuFloatComplex p_n2c1 = cuCmulf(n2, cos1);
    cuFloatComplex p_n3c2 = cuCmulf(n3, cos2);
    cuFloatComplex p_n2c3 = cuCmulf(n2, cos3);
    cuFloatComplex p_n1c2 = cuCmulf(n1, cos2);
    cuFloatComplex p_r12 = cuCdivf(cuCsubf(p_n2c1, p_n1c2), cuCaddf(p_n2c1, p_n1c2));
    cuFloatComplex p_r23 = cuCdivf(cuCsubf(p_n3c2, p_n2c3), cuCaddf(p_n3c2, p_n2c3));
    cuFloatComplex p_t12 = cuCdivf(cuCmulf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), n1), cos1), cuCaddf(p_n2c1, p_n1c2));
    cuFloatComplex p_t23 = cuCdivf(cuCmulf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), n2), cos2), cuCaddf(p_n3c2, p_n2c3));
    cuFloatComplex p_g = cuCdivf(cuCmulf(n3, cos3), cuCmulf(n1, cos1));

    float p_abs_r12 = cuCabsf(p_r12);
    float p_abs_r23 = cuCabsf(p_r23);
    float p_abs_t12 = cuCabsf(p_t12);
    float p_abs_t23 = cuCabsf(p_t23);
    float p_arg_r12 = cuCargf(p_r12);
    float p_arg_r23 = cuCargf(p_r23);
    float p_exp1 = exp(2.0f * v * e);

    float p_exp2 = 1.0f / p_exp1;
    float p_denom = p_exp1 +
                    p_abs_r12 * p_abs_r12 * p_abs_r23 * p_abs_r23 * p_exp2 +
                    2.0f * p_abs_r12 * p_abs_r23 * cosf(p_arg_r23 + p_arg_r12 + 2.0f * u * e);

    float p_r = p_abs_r12 * p_abs_r12 * p_exp1 + p_abs_r23 * p_abs_r23 * p_exp2 +
                2.0f * p_abs_r12 * p_abs_r23 * cosf(p_arg_r23 - p_arg_r12 + 2.0f * u * e);
    p_r /= p_denom;

    float p_t = cuCrealf(p_g) * p_abs_t12 * p_abs_t12 * p_abs_t23 * p_abs_t23;
    p_t /= p_denom;

    // calculate s polarization fraction, identical to propagate_at_boundary
    float incident_angle = get_theta(s.surface_normal,-p.direction);
    float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);

    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    float incident_plane_normal_length = norm(incident_plane_normal);

    if (incident_plane_normal_length < 1e-6f)
        incident_plane_normal = p.polarization;
    else
        incident_plane_normal /= incident_plane_normal_length;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);
    float normal_probability = normal_coefficient * normal_coefficient; // i.e. s polarization fraction

    float transmit = normal_probability * s_t + (1.0f - normal_probability) * p_t;
    if (!surface->transmissive)
        transmit = 0.0f;

    float reflect = normal_probability * s_r + (1.0f - normal_probability) * p_r;
    float absorb = 1.0f - transmit - reflect;

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        // Prevent absorption and reweight accordingly
        float survive = 1.0f - absorb;
        absorb = 0.0f;
        p.weight *= survive;

        detect /= survive;
        reflect /= survive;
        transmit /= survive;
    }

    if (use_weights && detect > 0.0f) {
        p.history |= SURFACE_DETECT;
        p.weight *= detect;
        return BREAK;
    }

    float uniform_sample = curand_uniform(&rng);

    if (uniform_sample < absorb) {
        // detection probability is conditional on absorption here
        float uniform_sample_detect = curand_uniform(&rng);
        if (uniform_sample_detect < detect)
            p.history |= SURFACE_DETECT;
        else
            p.history |= SURFACE_ABSORB;

        return BREAK;
    }
    else if (uniform_sample < absorb + reflect || !surface->transmissive) {
        // reflect, specularly (default) or diffusely
        float uniform_sample_reflect = curand_uniform(&rng);
        if (uniform_sample_reflect < reflect_diffuse)
            return propagate_at_diffuse_reflector(p, s, rng);
        else
            return propagate_at_specular_reflector(p, s);
    }
    else {
        // refract and transmit
        p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
        p.polarization = cross(incident_plane_normal, p.direction);
        p.polarization /= norm(p.polarization);
        p.history |= SURFACE_TRANSMIT;
        return CONTINUE;
    }
} // propagate_complex

__device__ int
propagate_at_wls(Photon &p, State &s, curandState &rng, Surface *surface, bool use_weights=false)
{
    float absorb = interp_property(surface, p.wavelength, surface->absorb);
    float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    float reemit = interp_property(surface, p.wavelength, surface->reemit);

    float uniform_sample = curand_uniform(&rng);

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        // Prevent absorption and reweight accordingly
        float survive = 1.0f - absorb;
        absorb = 0.0f;
        p.weight *= survive;
        reflect_diffuse /= survive;
        reflect_specular /= survive;
    }

    if (uniform_sample < absorb) {
        float uniform_sample_reemit = curand_uniform(&rng);
        if (uniform_sample_reemit < reemit) {
            p.history |= SURFACE_REEMIT;
            p.wavelength = sample_cdf(&rng, surface->wavelength_n, surface->wavelength_start, surface->wavelength_step, surface->reemission_cdf);
            p.direction = uniform_sphere(&rng);
            p.polarization = cross(uniform_sphere(&rng), p.direction);
            p.polarization /= norm(p.polarization);
            return CONTINUE;
        } else {
          p.history |= SURFACE_ABSORB;
          return BREAK;
        }
    }
    else if (uniform_sample < absorb + reflect_specular + reflect_diffuse) {
        // choose how to reflect, defaulting to diffuse
        float uniform_sample_reflect = curand_uniform(&rng) * (reflect_specular + reflect_diffuse);
        if (uniform_sample_reflect < reflect_specular)
            return propagate_at_specular_reflector(p, s);
        else
            return propagate_at_diffuse_reflector(p, s, rng);
    }
    else {
        p.history |= SURFACE_TRANSMIT;
        return PASS;
    }
} // propagate_at_wls


__noinline__ __device__ int
propagate_at_dichroic(Photon &p, State &s, curandState &rng, Surface *surface, bool use_weights=false)
{
    float incident_angle = get_theta(s.surface_normal, -p.direction);
    
    const DichroicProps *props = surface->dichroic_props;
    float idx = interp_idx(incident_angle,props->nangles,props->angles);
    unsigned int iidx = (int)idx;
    
    float reflect_prob_low = interp_property(surface, p.wavelength, props->dichroic_reflect[iidx]);
    float reflect_prob_high = interp_property(surface, p.wavelength, props->dichroic_reflect[iidx+1]);
    float transmit_prob_low = interp_property(surface, p.wavelength, props->dichroic_transmit[iidx]);
    float transmit_prob_high = interp_property(surface, p.wavelength, props->dichroic_transmit[iidx+1]);
    
    float reflect_prob = reflect_prob_low + (reflect_prob_high-reflect_prob_low)*(idx-iidx);
    float transmit_prob = transmit_prob_low + (transmit_prob_high-transmit_prob_low)*(idx-iidx);
    
    float uniform_sample = curand_uniform(&rng);
    if ((uniform_sample < reflect_prob)) {
        return propagate_at_specular_reflector(p, s);
    }
    else if (uniform_sample < transmit_prob+reflect_prob) {
        p.history |= SURFACE_TRANSMIT;
        return PASS;
    }
    else {
        p.history |= SURFACE_ABSORB;
        return BREAK;
    }

} // propagate_at_dichroic

__device__ int
propagate_at_angular(Photon &p, State &s, curandState &rng, Surface *surface, bool use_weights=false)
{
    float incident_angle = get_theta(s.surface_normal, -p.direction);
    
    const AngularProps *props = surface->angular_props;
    float idx = interp_idx(incident_angle, props->nangles, props->angles);
    unsigned int iidx = (int)idx;
    
    float t = idx - iidx;
    float transmit_prob = props->transmit[iidx] + t * (props->transmit[iidx+1] - props->transmit[iidx]);
    float reflect_spec_prob = props->reflect_specular[iidx] + t * (props->reflect_specular[iidx+1] - props->reflect_specular[iidx]);
    float reflect_diff_prob = props->reflect_diffuse[iidx] + t * (props->reflect_diffuse[iidx+1] - props->reflect_diffuse[iidx]);
    
    float absorb_prob = 1.0f - transmit_prob - reflect_spec_prob - reflect_diff_prob;
    
    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb_prob < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        float survive = 1.0f - absorb_prob;
        absorb_prob = 0.0f;
        p.weight *= survive;
        
        transmit_prob /= survive;
        reflect_spec_prob /= survive;
        reflect_diff_prob /= survive;
    }
    
    float uniform_sample = curand_uniform(&rng);
    
    if (uniform_sample < absorb_prob) {
        p.history |= SURFACE_ABSORB;
        return BREAK;
    }
    else if (uniform_sample < absorb_prob + transmit_prob) {
        p.history |= SURFACE_TRANSMIT;
        return PASS;
    }
    else if (uniform_sample < absorb_prob + transmit_prob + reflect_spec_prob) {
        return propagate_at_specular_reflector(p, s);
    }
    else {
        return propagate_at_diffuse_reflector(p, s, rng);
    }
} // propagate_at_angular

__device__ int
propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
                     bool use_weights=false)
{
    Surface *surface = geometry->surfaces[s.surface_index];

    if (surface->model == SURFACE_COMPLEX)
        return propagate_complex(p, s, rng, surface, use_weights);
    else if (surface->model == SURFACE_WLS)
        return propagate_at_wls(p, s, rng, surface, use_weights);
    else if (surface->model == SURFACE_DICHROIC)
        return propagate_at_dichroic(p, s, rng, surface, use_weights);
    else if (surface->model == SURFACE_ANGULAR)
        return propagate_at_angular(p, s, rng, surface, use_weights);
    else {
        // use default surface model: do a combination of specular and
        // diffuse reflection, detection, and absorption based on relative
        // probabilties

        // since the surface properties are interpolated linearly, we are
        // guaranteed that they still sum to 1.0.
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
        else if (uniform_sample < absorb + detect + reflect_diffuse)
            return propagate_at_diffuse_reflector(p, s, rng);
        else if (uniform_sample < absorb + detect + reflect_diffuse + reflect_specular)
            return propagate_at_specular_reflector(p, s);
        else
            return PASS;
    }

} // propagate_at_surface

#endif

