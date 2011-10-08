#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

struct Material
{
    float *refractive_index;
    float *absorption_length;
    float *scattering_length;
    unsigned int n;
    float step;
    float wavelength0;
};

struct Surface
{
    float *detect;
    float *absorb;
    float *reflect_diffuse;
    float *reflect_specular;
    unsigned int n;
    float step;
    float wavelength0;
};

struct Triangle
{
    float3 v0, v1, v2;
};

struct Geometry
{
    float3 *vertices;
    uint3 *triangles;
    unsigned int *material_codes;
    unsigned int *colors;
    float3 *lower_bounds;
    float3 *upper_bounds;
    unsigned int *node_map;
    unsigned int *node_map_end;
    Material **materials;
    Surface **surfaces;
    unsigned int start_node;
    unsigned int first_node;
};

__device__ Triangle
get_triangle(Geometry *geometry, const unsigned int &i)
{
    uint3 triangle_data = geometry->triangles[i];

    Triangle triangle;
    triangle.v0 = geometry->vertices[triangle_data.x];
    triangle.v1 = geometry->vertices[triangle_data.y];
    triangle.v2 = geometry->vertices[triangle_data.z];

    return triangle;
}

template <class T>
__device__ float
interp_property(T *m, const float &x, const float *fp)
{
    if (x < m->wavelength0)
	return fp[0];

    if (x > (m->wavelength0 + (m->n-1)*m->step))
	return fp[m->n-1];

    int jl = (x-m->wavelength0)/m->step;

    return fp[jl] + (x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step;
}

#endif
