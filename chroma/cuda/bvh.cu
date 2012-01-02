//-*-c++-*-
#include <cuda.h>

#include "linalg.h"

__device__ float3
fminf(const float3 &a, const float3 &b)
{
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ float3
fmaxf(const float3 &a, const float3 &b)
{
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ uint3
min(const uint3 &a, const uint3 &b)
{
  return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

__device__ uint3
max(const uint3 &a, const uint3 &b)
{
  return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}


__device__ uint3
operator+ (const uint3 &a, const unsigned int &b)
{
  return make_uint3(a.x + b, a.y + b, a.z + b);
}

// spread out the first 16 bits in x to occupy every 3rd slot in the return value
__device__ unsigned long long spread3_16(unsigned int input)
{
  // method from http://stackoverflow.com/a/4838734
  unsigned long long x = input;
  x = (x | (x << 16)) & 0x00000000FF0000FFul;
  x = (x | (x << 8)) & 0x000000F00F00F00Ful;
  x = (x | (x << 4)) & 0x00000C30C30C30C3ul;
  x = (x | (x << 2)) & 0X0000249249249249ul;
  
  return x;
}

__device__ unsigned int quantize(float v, float world_origin, float world_scale)
{
  // truncate!
  return (unsigned int) ((v - world_origin) / world_scale);
}

__device__ uint3 quantize3(float3 v, float3 world_origin, float3 world_scale)
{
  return make_uint3(quantize(v.x, world_origin.x, world_scale.x),
		    quantize(v.y, world_origin.y, world_scale.y),
		    quantize(v.z, world_origin.z, world_scale.z));
}

const unsigned int LEAF_BIT = (1U << 31);

extern "C"
{

  __global__ void
  make_leaves(unsigned int first_triangle,
	      unsigned int ntriangles, uint3 *triangles, float3 *vertices,
	      float3 world_origin, float3 world_scale,
	      uint4 *leaf_nodes, unsigned long long *morton_codes)

  {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= ntriangles)
      return;

    unsigned int triangle_id = first_triangle + thread_id;

    // Find bounding corners and centroid
    uint3 triangle = triangles[triangle_id];
    float3 lower = vertices[triangle.x];
    float3 centroid = lower;
    float3 upper = lower;
    float3 temp_vertex;

    temp_vertex = vertices[triangle.y];
    lower = fminf(lower, temp_vertex);
    upper = fmaxf(upper, temp_vertex);
    centroid  += temp_vertex;

    temp_vertex = vertices[triangle.z];
    lower = fminf(lower, temp_vertex);
    upper = fmaxf(upper, temp_vertex);
    centroid  += temp_vertex;

    centroid /= 3.0f;

    // Quantize bounding corners and centroid
    uint3 q_lower = quantize3(lower, world_origin, world_scale);
    uint3 q_upper = quantize3(upper, world_origin, world_scale) + 1;
    uint3 q_centroid = quantize3(centroid, world_origin, world_scale);

    // Compute Morton code from quantized centroid
    unsigned long long morton = 
      spread3_16(q_centroid.x) 
      | (spread3_16(q_centroid.y) << 1)
      | (spread3_16(q_centroid.z) << 2);
    
    // Write leaf and morton code
    uint4 leaf_node;
    leaf_node.x = q_lower.x | (q_upper.x << 16);
    leaf_node.y = q_lower.y | (q_upper.y << 16);
    leaf_node.z = q_lower.z | (q_upper.z << 16);
    leaf_node.w = triangle_id | LEAF_BIT;

    leaf_nodes[triangle_id] = leaf_node;
    morton_codes[triangle_id] = morton;
  }

  __global__ void
  reorder_leaves(unsigned int first_triangle,
		 unsigned int ntriangles,
		 uint4 *leaf_nodes_in, 
		 uint4 *leaf_nodes_out,
		 unsigned int *remap_order)

  {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= ntriangles)
      return;

    unsigned int dest_id = first_triangle + thread_id;
    unsigned int source_id = remap_order[dest_id];

    leaf_nodes_out[dest_id] = leaf_nodes_in[source_id];
  }

  __global__ void
  build_layer(unsigned int first_node,
	      unsigned int n_parent_nodes, 
	      unsigned int n_children_per_node,
	      uint4 *nodes,
	      unsigned int parent_layer_offset,
	      unsigned int child_layer_offset)
  {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= n_parent_nodes)
      return;

    unsigned int parent_id = first_node + thread_id;
    unsigned int first_child = child_layer_offset + parent_id * n_children_per_node;

    // Load first child
    uint4 parent_node = nodes[first_child];
    uint3 lower = make_uint3(parent_node.x & 0xFFFF, parent_node.y & 0xFFFF, parent_node.z & 0xFFFF);
    uint3 upper = make_uint3(parent_node.x >> 16, parent_node.y >> 16, parent_node.z >> 16);
    

    // Scan remaining children
    for (unsigned int i=1; i < n_children_per_node; i++) {
      uint4 child_node = nodes[first_child + i];
      
      if (child_node.x == 0)
	break;  // Hit first padding node in list of children

      uint3 child_lower = make_uint3(child_node.x & 0xFFFF, child_node.y & 0xFFFF, child_node.z & 0xFFFF);
      uint3 child_upper = make_uint3(child_node.x >> 16, child_node.y >> 16, child_node.z >> 16);

      lower = min(lower, child_lower);
      upper = max(upper, child_upper);
    }

    parent_node.w = first_child;
    parent_node.x = upper.x << 16 | lower.x;
    parent_node.y = upper.y << 16 | lower.y;
    parent_node.z = upper.z << 16 | lower.z;

    nodes[parent_layer_offset + parent_id] = parent_node;
  }

} // extern "C"
