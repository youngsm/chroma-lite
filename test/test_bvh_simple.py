import pycuda.autoinit
import unittest
from chroma.bvh import make_simple_bvh, BVH
import chroma.models

import numpy as np
#from numpy.testing import assert_array_max_ulp, assert_array_equal, \
#    assert_approx_equal

def test_simple_bvh():
    mesh = chroma.models.lionsolid()
    bvh = make_simple_bvh(mesh, degree=2)
    assert isinstance(bvh, BVH)

