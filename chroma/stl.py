import numpy as np
import string
import struct
from chroma.geometry import Mesh
import bz2

def mesh_from_stl(filename):
    "Returns a `chroma.geometry.Mesh` from an STL file."
    if filename.endswith('.bz2'):
        f = bz2.BZ2File(filename)
    else:
        f = open(filename)
    buf = f.read(200)
    f.close()

    for char in buf:
        if char not in string.printable:
            return mesh_from_binary_stl(filename)

    return mesh_from_ascii_stl(filename)

def mesh_from_ascii_stl(filename):
    "Return a mesh from an ascii stl file."
    if filename.endswith('.bz2'):
        f = bz2.BZ2File(filename)
    else:
        f = open(filename)

    vertices = []
    triangles = []
    vertex_map = {}

    while True:
        line = f.readline()

        if line == '':
            break

        if not line.strip().startswith('vertex'):
            continue

        triangle = [None]*3
        for i in range(3):
            vertex = tuple([float(s) for s in line.strip().split()[1:]])

            if vertex not in vertex_map:
                vertices.append(vertex)
                vertex_map[vertex] = len(vertices) - 1

            triangle[i] = vertex_map[vertex]

            if i < 3:
                line = f.readline()

        triangles.append(triangle)

    f.close()

    return Mesh(np.array(vertices), np.array(triangles, dtype=np.uint32))

def mesh_from_binary_stl(filename):
    "Return a mesh from a binary stl file."
    if filename.endswith('.bz2'):
        f = bz2.BZ2File(filename)
    else:
        f = open(filename)

    vertices = []
    triangles = []
    vertex_map = {}

    f.read(80)
    ntriangles = struct.unpack('<I', f.read(4))[0]

    for i in range(ntriangles):
        normal = tuple(struct.unpack('<fff', f.read(12)))

        triangle = [None]*3
        for j in range(3):
            vertex = tuple(struct.unpack('<fff', f.read(12)))

            if vertex not in vertex_map:
                vertices.append(vertex)
                vertex_map[vertex] = len(vertices) - 1

            triangle[j] = vertex_map[vertex]

        triangles.append(triangle)

        f.read(2)

    f.close()

    return Mesh(np.array(vertices), np.array(triangles, dtype=np.uint32))
