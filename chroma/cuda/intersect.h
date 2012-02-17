//-*-c-*-

#ifndef __INTERSECT_H__
#define __INTERSECT_H__

#include "linalg.h"
#include "matrix.h"
#include "geometry.h"

#define EPSILON 0.0f

/* Tests the intersection between a ray and a triangle.
   If the ray intersects the triangle, set `distance` to the distance from
   `origin` to the intersection and return true, else return false.
   `direction` must be normalized to one. */
__device__ bool
intersect_triangle(const float3 &origin, const float3 &direction,
		   const Triangle &triangle, float &distance)
{
	float3 m1 = triangle.v1-triangle.v0;
	float3 m2 = triangle.v2-triangle.v0;
	float3 m3 = -direction;

	Matrix m = make_matrix(m1, m2, m3);
	
	float determinant = det(m);

	if (determinant == 0.0f)
		return false;

	float3 b = origin-triangle.v0;

	float u1 = ((m.a11*m.a22 - m.a12*m.a21)*b.x +
		    (m.a02*m.a21 - m.a01*m.a22)*b.y +
		    (m.a01*m.a12 - m.a02*m.a11)*b.z)/determinant;

	if (u1 < -EPSILON || u1 > 1.0f)
		return false;

	float u2 = ((m.a12*m.a20 - m.a10*m.a22)*b.x +
		    (m.a00*m.a22 - m.a02*m.a20)*b.y +
		    (m.a02*m.a10 - m.a00*m.a12)*b.z)/determinant;

	if (u2 < -EPSILON || u2 > 1.0f)
		return false;

	float u3 = ((m.a10*m.a21 - m.a11*m.a20)*b.x +
		    (m.a01*m.a20 - m.a00*m.a21)*b.y +
		    (m.a00*m.a11 - m.a01*m.a10)*b.z)/determinant;

	if (u3 <= 0.0f || (1.0f-u1-u2) < -EPSILON)
		return false;

	distance = u3;

	return true;
}

/* Tests the intersection between a ray and an axis-aligned box defined by
   an upper and lower bound. If the ray intersects the box, set
   `distance_to_box` to the distance from `origin` to the intersection and
   return true, else return false. `direction` must be normalized to one.

    Source: Optimizing ray tracing for CUDA by Hannu Saransaari
    https://wiki.aalto.fi/download/attachments/40023967/gpgpu.pdf
*/
__device__ bool
intersect_box(const float3 &neg_origin_inv_dir, const float3 &inv_dir,
	      const float3 &lower_bound, const float3 &upper_bound,
	      float& distance_to_box)
{
	float tmin = 0.0f, tmax = 1e30f;
	float t0, t1;

	// X
	t0 = lower_bound.x * inv_dir.x + neg_origin_inv_dir.x;
	t1 = upper_bound.x * inv_dir.x + neg_origin_inv_dir.x;
	
	tmin = max(tmin, min(t0, t1));
	tmax = min(tmax, max(t0, t1));
	
	// Y
	t0 = lower_bound.y * inv_dir.y + neg_origin_inv_dir.y;
	t1 = upper_bound.y * inv_dir.y + neg_origin_inv_dir.y;
	
	tmin = max(tmin, min(t0, t1));
	tmax = min(tmax, max(t0, t1));

	// Z
	t0 = lower_bound.z * inv_dir.z + neg_origin_inv_dir.z;
	t1 = upper_bound.z * inv_dir.z + neg_origin_inv_dir.z;
	
	tmin = max(tmin, min(t0, t1));
	tmax = min(tmax, max(t0, t1));

	if (tmin > tmax)
		return false;

	distance_to_box = tmin;

	return true;
}

#endif
