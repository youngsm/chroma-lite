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

   Source: "An Efficient and Robust Ray-Box Intersection Algorithm."
   by Williams, et. al. */
__device__ bool
intersect_box(const float3 &origin, const float3 &direction,
	      const float3 &lower_bound, const float3 &upper_bound,
	      float& distance_to_box)
{
	float kmin, kmax, kymin, kymax, kzmin, kzmax;

	float divx = 1.0f/direction.x;
	if (divx >= 0.0f)
	{
		kmin = (lower_bound.x - origin.x)*divx;
		kmax = (upper_bound.x - origin.x)*divx;
	}
	else
	{
		kmin = (upper_bound.x - origin.x)*divx;
		kmax = (lower_bound.x - origin.x)*divx;
	}

	if (kmax < 0.0f)
		return false;

	float divy = 1.0f/direction.y;
	if (divy >= 0.0f)
	{
		kymin = (lower_bound.y - origin.y)*divy;
		kymax = (upper_bound.y - origin.y)*divy;
	}
	else
	{
		kymin = (upper_bound.y - origin.y)*divy;
		kymax = (lower_bound.y - origin.y)*divy;
	}

	if (kymax < 0.0f)
		return false;

	if (kymin > kmin)
		kmin = kymin;

	if (kymax < kmax)
		kmax = kymax;

	if (kmin > kmax)
		return false;

	float divz = 1.0f/direction.z;
	if (divz >= 0.0f)
	{
		kzmin = (lower_bound.z - origin.z)*divz;
		kzmax = (upper_bound.z - origin.z)*divz;
	}
	else
	{
		kzmin = (upper_bound.z - origin.z)*divz;
		kzmax = (lower_bound.z - origin.z)*divz;
	}

	if (kzmax < 0.0f)
		return false;

	if (kzmin > kmin)
		kmin = kzmin;

	if (kzmax < kmax)
		kmax = kzmax;

	if (kmin > kmax)
		return false;

	distance_to_box = kmin;

	return true;
}

#endif
