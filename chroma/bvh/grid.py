import numpy as np

from chroma.bvh.bvh import BVH, CHILD_BITS
from chroma.gpu.bvh import create_leaf_nodes, merge_nodes_detailed, concatenate_layers

MAX_CHILD = 2**(32 - CHILD_BITS) - 1

@profile
def make_recursive_grid_bvh(mesh, target_degree=3):
    '''Returns a BVH created using a 'recursive grid' method.

    This method is somewhat similar to the original Chroma BVH generator, 
    with a few adjustments:
      * Every triangle is boxed individually by a leaf node in the BVH
      * Triangles are not rearranged in Morton order since the
        leaf nodes are sorted instead.
      * The GPU is used to assist in the calculation of the tree to
        speed things up.

    '''
    world_coords, leaf_nodes, morton_codes = create_leaf_nodes(mesh)

    # rearrange in morton order
    argsort = morton_codes.argsort()
    leaf_nodes = leaf_nodes[argsort]
    morton_codes = morton_codes[argsort]

    # Create parent layers
    layers = [leaf_nodes]
    while len(layers[0]) > 1:        
        top_layer = layers[0]

        # Figure out how many bits to shift in morton code to achieve desired
        # degree
        nnodes = len(top_layer)
        nunique = len(np.unique(morton_codes))
        while nnodes / float(nunique) < target_degree and nunique > 1:
            morton_codes >>= 1
            nunique = len(np.unique(morton_codes))

        # Determine the grouping of child nodes into parents
        parent_morton_codes, first_child = np.unique(morton_codes, return_index=True)
        first_child = first_child.astype(np.int32)
        nchild = np.ediff1d(first_child, to_end=nnodes - first_child[-1])

        # Expand groups that have too many children
        print 'Expanding %d parent nodes' % (np.count_nonzero(nchild > MAX_CHILD))
        while nchild.max() > MAX_CHILD:
            index = nchild.argmax()
            new_first = np.arange(first_child[index], first_child[index]+nchild[index], MAX_CHILD)
            first_child = np.concatenate([first_child[:index],
                                          new_first,
                                          first_child[index+1:]])
            parent_morton_codes = np.concatenate([parent_morton_codes[:index],
                           np.repeat(parent_morton_codes[index], len(new_first)), 
                                                  parent_morton_codes[index+1:]])
            nchild = np.ediff1d(first_child, to_end=nnodes - first_child[-1])

        if nunique > 1: # Yes, I'm pedantic
            plural = 's'
        else:
            plural = ''
        print 'Merging %d nodes to %d parent%s' % (nnodes, nunique, plural)

        assert (nchild > 0).all()
        assert (nchild < 31).all()
        
        #print 'Max|Avg children: %d|%d' % (nchild.max(), nchild.mean())

        # Merge children
        parents = merge_nodes_detailed(top_layer, first_child, nchild)
        layers = [parents] + layers
        morton_codes = parent_morton_codes

    nodes, layer_bounds = concatenate_layers(layers)
    return BVH(world_coords, nodes, layer_bounds[:-1])
