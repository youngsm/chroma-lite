import numpy as np

def make_rotation_matrix(phi, n):
    """
    Make the rotation matrix to rotate points through an angle `phi`
    counter-clockwise around the axis `n` (when looking towards +infinity).

    Source: Weissten, Eric W. "Rotation Formula." Mathworld.

    Raises ValueError if n has zero magnitude
    """
    norm = np.linalg.norm(n)
    if norm == 0.0:
        raise ValueError('rotation axis has zero magnitude')
    n = np.asarray(n)/norm
    
    return np.cos(phi)*np.identity(3) + (1-np.cos(phi))*np.outer(n,n) + \
        np.sin(phi)*np.array([[0,n[2],-n[1]],[-n[2],0,n[0]],[n[1],-n[0],0]])

def rotate(x, phi, n):
    """
    Rotate an array of points `x` through an angle phi counter-clockwise
    around the axis `n` (when looking towards +infinity).
    """
    return np.inner(np.asarray(x),make_rotation_matrix(phi, n))

def normalize(x):
    "Returns unit vectors in the direction of `x`."
    x = np.asarray(x)

    if x.shape[-1] != 3:
        raise ValueError('dimension of last axis must be 3.')

    d = len(x.shape)

    if d == 1:
        norm = np.sqrt(x.dot(x))
    elif d == 2:
        norm = np.sqrt(np.sum(x*x, axis=1))[:,np.newaxis]
    else:
        raise ValueError('len(`x`.shape) must be zero or one.')

    return x/norm
