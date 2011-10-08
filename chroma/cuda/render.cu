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
       unsigned int *dxlen, float4 *_color)
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

    float distance;

    if (n < 1 && !intersect_node(origin, direction, g, g->start_node)) {
	pixels[id] = 0;
	return;
    }

    unsigned int stack[STACK_SIZE];

    unsigned int *head = &stack[0];
    unsigned int *node = &stack[1];
    unsigned int *tail = &stack[STACK_SIZE-1];
    *node = g->start_node;

    float *dx = _dx + id*alpha_depth;
    float4 *color_a = _color + id*alpha_depth;

    unsigned int i;

    do {
	unsigned int first_child = g->node_map[*node];
	unsigned int stop = g->node_map_end[*node];
	
	while (*node >= g->first_node && stop == first_child+1) {
	    *node = first_child;
	    first_child = g->node_map[*node];
	    stop = g->node_map_end[*node];
	}
		
	if (*node >= g->first_node) {
	    for (i=first_child; i < stop; i++) {
		if (intersect_node(origin, direction, g, i)) {
		    *node = i;
		    node++;
		}
	    }

	    node--;
	}
	else {
	    // node is a leaf
	    for (i=first_child; i < stop; i++) {
		Triangle t = get_triangle(g, i);
		
		if (intersect_triangle(origin, direction, t, distance)) {
		    if (n < 1) {
			dx[0] = distance;
			    
			unsigned int rgba = g->colors[i];
			float4 color = get_color(direction, t, rgba);

			color_a[0] = color;
		    }
		    else {
			unsigned long j = searchsorted(n, dx, distance);

			if (j <= alpha_depth-1) {
			    insert(alpha_depth, dx, j, distance);

			    unsigned int rgba = g->colors[i];
			    float4 color = get_color(direction, t, rgba);

			    insert(alpha_depth, color_a, j, color);
			}
		    }
					
		    if (n < alpha_depth)
			n++;
		}
				
	    } // triangle loop
	    
	    node--;
	    
	} // node is a leaf
	
    } // while loop
    while (node != head);

    if (n < 1) {
	pixels[id] = 0;
	return;
    }

    dxlen[id] = n;

    float scale = 1.0f;
    float fr = 0.0f;
    float fg = 0.0f;
    float fb = 0.0f;
    for (i=0; i < n; i++) {
	float alpha = color_a[i].w;

	fr += scale*color_a[i].x*alpha;
	fg += scale*color_a[i].y*alpha;
	fb += scale*color_a[i].z*alpha;
	
	scale *= (1.0f-alpha);
    }
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
