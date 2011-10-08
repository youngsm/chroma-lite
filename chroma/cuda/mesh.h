#ifndef __MESH_H__
#define __MESH_H__

#include "intersect.h"
#include "geometry.h"

#include "stdio.h"

#define STACK_SIZE 100

/* Tests the intersection between a ray and a node in the bounding volume
   hierarchy. If the ray intersects the bounding volume and `min_distance`
   is less than zero or the distance from `origin` to the intersection is
   less than `min_distance`, return true, else return false. */
__device__ bool
intersect_node(const float3 &origin, const float3 &direction,
	       Geometry *g, int i, float min_distance=-1.0f)
{
    /* assigning these to local variables is faster for some reason */
    float3 lower_bound = g->lower_bounds[i];
    float3 upper_bound = g->upper_bounds[i];

    float distance_to_box;

    if (intersect_box(origin, direction, lower_bound, upper_bound,
		      distance_to_box)) {
	if (min_distance < 0.0f)
	    return true;

	if (distance_to_box > min_distance)
	    return false;

	return true;
    }
    else {
	return false;
    }

}

/* Finds the intersection between a ray and `geometry`. If the ray does
   intersect the mesh and the index of the intersected triangle is not equal
   to `last_hit_triangle`, set `min_distance` to the distance from `origin` to
   the intersection and return the index of the triangle which the ray
   intersected, else return -1. */
__device__ int
intersect_mesh(const float3 &origin, const float3& direction, Geometry *g,
	       float &min_distance, int last_hit_triangle = -1)
{
    int triangle_index = -1;

    float distance;
    min_distance = -1.0f;

    if (!intersect_node(origin, direction, g, g->start_node, min_distance))
	return -1;

    unsigned int stack[STACK_SIZE];

    unsigned int *head = &stack[0];
    unsigned int *node = &stack[1];
    unsigned int *tail = &stack[STACK_SIZE-1];
    *node = g->start_node;

    unsigned int i;

    do
    {
	unsigned int first_child = g->node_map[*node];
	unsigned int stop = g->node_map_end[*node];

	while (*node >= g->first_node && stop == first_child+1) {
	    *node = first_child;
	    first_child = g->node_map[*node];
	    stop = g->node_map_end[*node];
	}
		
	if (*node >= g->first_node) {
	    for (i=first_child; i < stop; i++) {
		if (intersect_node(origin, direction, g, i, min_distance)) {
		    *node = i;
		    node++;
		}
	    }
	    
	    node--;
	}
	else {
	    // node is a leaf
	    for (i=first_child; i < stop; i++) {
		if (last_hit_triangle == i)
		    continue;

		Triangle t = get_triangle(g, i);

		if (intersect_triangle(origin, direction, t, distance)) {
		    if (triangle_index == -1) {
			triangle_index = i;
			min_distance = distance;
			continue;
		    }

		    if (distance < min_distance) {
			triangle_index = i;
			min_distance = distance;
		    }
		}
	    } // triangle loop

	    node--;

	} // node is a leaf

	if (node > tail) {
	    printf("warning: intersect_mesh() aborted; node > tail\n");
	    break;
	}

    } // while loop
    while (node != head);
    
    return triangle_index;
}

extern "C"
{

__global__ void
distance_to_mesh(int nthreads, float3 *_origin, float3 *_direction,
		 Geometry *g, float *_distance)
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
    direction /= norm(direction);

    float distance;

    int triangle_index = intersect_mesh(origin, direction, g, distance);

    if (triangle_index != -1)
	_distance[id] = distance;
}

__global__ void
color_solids(int first_triangle, int nthreads, int *solid_id_map,
	     bool *solid_hit, unsigned int *solid_colors, Geometry *g)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    int triangle_id = first_triangle + id;
    int solid_id = solid_id_map[triangle_id];
    if (solid_hit[solid_id])
	g->colors[triangle_id] = solid_colors[solid_id];
}

} // extern "C"

#endif
