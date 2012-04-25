#ifndef __GEOMETRY_TYPES_H__
#define __GEOMETRY_TYPES_H__

struct Material
{
    float *refractive_index;
    float *absorption_length;
    float *scattering_length;
    unsigned int n;
    float step;
    float wavelength0;
};

// surface models
enum {
    SURFACE_DEFAULT,  // specular + diffuse + absorption + detection
    SURFACE_SPECULAR, // perfect specular reflector
    SURFACE_DIFFUSE,  // perfect diffuse reflector
    SURFACE_COMPLEX,  // use complex index of refraction
    SURFACE_WLS       // wavelength-shifting reemission
};

struct Surface
{
    float *detect;
    float *absorb;
    float *reemit;
    float *reflect;
    float *reflect_diffuse;
    float *reflect_specular;
    float *eta;
    float *k;

    float *reemission_wavelength;
    float *reemission_cdf;

    unsigned int model;
    unsigned int n;
    unsigned int reemission_n;
    float step;
    float wavelength0;
    float thickness;
    bool transmissive;
};

struct Triangle
{
    float3 v0, v1, v2;
};

enum { INTERNAL_NODE, LEAF_NODE, PADDING_NODE };
const unsigned int CHILD_BITS = 28;
const unsigned int NCHILD_MASK = (0xFFFFu << CHILD_BITS);

struct Node
{
    float3 lower;
    float3 upper;
    unsigned int child;
    unsigned int nchild;
};

struct Geometry
{
    float3 *vertices;
    uint3 *triangles;
    unsigned int *material_codes;
    unsigned int *colors;
    uint4 *nodes;
    Material **materials;
    Surface **surfaces;
    float3 world_origin;
    float world_scale;
};

#endif
