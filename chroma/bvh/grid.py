from chroma.bvh.bvh import BVH, node_areas
from chroma.gpu.bvh import create_leaf_nodes

def make_recursive_grid_bvh(mesh, bits=11):
    '''Returns a binary tree BVH created using a 'recursive grid' method.

    This method is somewhat similar to the original Chroma BVH generator, 
    with a few adjustments:
      * It is a strict binary tree, with dummy nodes inserted when
        only one child is required on an inner node.
      * Every triangle is boxed individually by a leaf node in the BVH
      * Triangles are not rearranged in Morton order since the
        leaf nodes are sorted instead.
      * The GPU is used to assist in the calculation of the tree to
        speed things up.

    '''
    world_coords, leaf_nodes, morton_codes = create_leaf_nodes(mesh, bits)

    # rearrange in morton order
    argsort = morton_codes.argsort()
    leaf_nodes = leaf_nodes[argsort]
    morton_codes[argsort]

    
    print node_areas(leaf_nodes).sum() * world_coords.world_scale**2

    
