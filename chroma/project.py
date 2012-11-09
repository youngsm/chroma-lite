import numpy as np
from chroma.transform import normalize

def from_film(position=(0,0,0), axis1=(0,0,1), axis2=(1,0,0), size=(800,600),
              width=35.0, focal_length=18.0):
    """Project rays from a piece of film whose focal point is located at
    `position`. `axis1` and `axis2` specify the vectors pointing along the
    length and height of the film respectively.
    """

    height = width*(size[1]/float(size[0]))

    axis1 = normalize(axis1)
    axis2 = normalize(axis2)

    dx0 = width/size[0]
    dx1 = height/size[1]

    # for i in range(size[0]):
    #     for j in range(size[1]):
    #         grid[i*size[1]+j] = axis1*dx1*j - axis2*dx0*i

    x = np.arange(size[0])
    y = np.arange(size[1])

    yy, xx = np.meshgrid(y,x)

    n = size[0]*size[1]

    grid = -np.tile(axis2, (n,1))*xx.ravel()[:,np.newaxis]*dx0 + \
        np.tile(axis1, (n,1))*yy.ravel()[:,np.newaxis]*dx1

    grid += axis2*width/2 - axis1*height/2
    grid -= np.cross(axis1,axis2)*focal_length

    return np.tile(position,(n,1)), normalize(-grid)
