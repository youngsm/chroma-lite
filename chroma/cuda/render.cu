//-*-c-*-

#include "linalg.h"
#include "intersect.h"
#include "mesh.h"
#include "sorting.h"
#include "geometry.h"

#include "stdio.h"

__device__ float4
get_color(const float3 &direction, const Triangle &t, unsigned int rgba)
{
    float3 v01 = t.v1 - t.v0;
    float3 v12 = t.v2 - t.v1;
    
    float3 surface_normal = normalize(cross(v01,v12));

    float cos_theta = dot(surface_normal,-direction);

    if (cos_theta < 0.0f)
	cos_theta = -cos_theta;

    unsigned int a0 = 0xff & (rgba >> 24);
    unsigned int r0 = 0xff & (rgba >> 16);
    unsigned int g0 = 0xff & (rgba >> 8);
    unsigned int b0 = 0xff & rgba;

    float alpha = (255 - a0)/255.0f;

    return make_float4(r0*cos_theta, g0*cos_theta, b0*cos_theta, alpha);
}

extern "C"
{

__global__ void
render(int nthreads, float3 *_origin, float3 *_direction, Geometry *g,
       unsigned int alpha_depth, unsigned int *pixels, float *_dx,
       unsigned int *dxlen, float4 *_color, unsigned int bg_color,
       const float *__restrict__ optix_distances,
       const int *__restrict__ optix_triangles,
       int use_optix)
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
	sg = *g;

    __syncthreads();

    int id = blockIdx.x*blockDim.x + threadIdx.x;
	
    if (id >= nthreads)
	return;
    
    g = &sg;

    float3 origin = _origin[id];
    float3 direction = _direction[id];
    unsigned int n = dxlen[id];

    float *dx = _dx + id*alpha_depth;
    float4 *color_a = _color + id*alpha_depth;
    unsigned int n = dxlen[id];

    if( use_optix )
    {
        unsigned int hits = 0;
        float3 norm_dir = normalize(direction);
        for( unsigned int depth = 0; depth < alpha_depth; ++depth )
        {
            float dist = optix_distances ? optix_distances[depth * nthreads + id] : -1.0f;
            int tri_idx = optix_triangles ? optix_triangles[depth * nthreads + id] : -1;
            if( dist < 0.0f || tri_idx < 0 )
                continue;

            Triangle t = get_triangle(g, tri_idx);
            dx[hits]   = dist;
            color_a[hits] = get_color(norm_dir, t, g->colors[tri_idx]);
            ++hits;
            if( hits >= alpha_depth )
                break;
        }

        dxlen[id] = hits;
        if( hits < 1 )
        {
            pixels[id] = bg_color;
            return;
        }
    }
    else
    {
        float distance;
        Node root = get_node(g, 0);

        float3 neg_origin_inv_dir = -origin / direction;
        float3 inv_dir = 1.0f / direction;

        if( n < 1 && !intersect_node( neg_origin_inv_dir, inv_dir, g, root ) )
        {
            pixels[id] = bg_color;
            return;
        }

        unsigned int child_ptr_stack[STACK_SIZE];
        unsigned int nchild_ptr_stack[STACK_SIZE];
        child_ptr_stack[0] = root.child;
        nchild_ptr_stack[0] = root.nchild;

        int curr = 0;

        float3 norm_dir = normalize(direction);

        while( curr >= 0 )
        {
            unsigned int first_child = child_ptr_stack[curr];
            unsigned int nchild      = nchild_ptr_stack[curr];
            curr--;

            for( unsigned int i = first_child; i < first_child + nchild; i++ )
            {
                Node node = get_node( g, i );

                if( intersect_node( neg_origin_inv_dir, inv_dir, g, node ) )
                {
                    if( node.nchild == 0 )
                    {
                        Triangle t = get_triangle( g, node.child );
                        if( intersect_triangle( origin, direction, t, distance ) )
                        {
                            unsigned int insert_index = ( n < 1 ) ? 0 : searchsorted( n, dx, distance );
                            if( insert_index <= alpha_depth - 1 )
                            {
                                insert( alpha_depth, dx, insert_index, distance );
                                unsigned int rgba = g->colors[node.child];
                                float4 color      = get_color( norm_dir, t, rgba );
                                insert( alpha_depth, color_a, insert_index, color );
                                if( n < alpha_depth )
                                    ++n;
                            }
                        }
                    }
                    else
                    {
                        curr++;
                        child_ptr_stack[curr]  = node.child;
                        nchild_ptr_stack[curr] = node.nchild;
                    }
                }
            }
        }

        if( n < 1 )
        {
            pixels[id] = bg_color;
            return;
        }
        dxlen[id] = n;
    }

    n = dxlen[id];

    float scale = 1.0f;
    float fr = 0.0;
    float fg = 0.0;
    float fb = 0.0;
    for (int i=0; i < n; i++) {
	float alpha = color_a[i].w;
	
	fr += scale*color_a[i].x*alpha;
	fg += scale*color_a[i].y*alpha;
	fb += scale*color_a[i].z*alpha;
	
	scale *= (1.0f-alpha);
    }
    float alpha = ((float)((bg_color & 0xFF000000) >> 24))/255.0;
    fr += scale*((float)((bg_color & 0xFF0000) >> 16))*alpha;
    fg += scale*((float)((bg_color & 0xFF00) >> 8))*alpha;
    fb += scale*((float)(bg_color & 0xFF))*alpha;
    scale *= (1.0f-alpha);
    
    unsigned int a;
    if (n < alpha_depth)
	a = floorf(255*(1.0f-scale));
    else
    	a = 255;
    	
    unsigned int red = floorf(fr/(1.0f-scale));
    unsigned int green = floorf(fg/(1.0f-scale));
    unsigned int blue = floorf(fb/(1.0f-scale));

    pixels[id] = a << 24 | red << 16 | green << 8 | blue;
}

} // extern "C"
