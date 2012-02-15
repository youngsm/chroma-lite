import numpy as np

from chroma.bvh.bvh import BVH, CHILD_BITS
from chroma.gpu.bvh import create_leaf_nodes, merge_nodes_detailed, concatenate_layers

MAX_CHILD = 2**(32 - CHILD_BITS) - 1

def count_unique_in_sorted(a):
    return (np.ediff1d(a) > 0).sum() + 1

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
    morton_codes >>= 12
    # Create parent layers
    layers = [leaf_nodes]
    while len(layers[0]) > 1:        
        top_layer = layers[0]

        # Figure out how many bits to shift in morton code to achieve desired
        # degree
        nnodes = len(top_layer)
        
        #nunique = count_unique_in_sorted(morton_codes)
        #while nnodes / float(nunique) < target_degree and nunique > 1:
        #    morton_codes >>= 1
        #    nunique = count_unique_in_sorted(morton_codes)
        morton_codes >>= 3
        nunique = count_unique_in_sorted(morton_codes)

        # Determine the grouping of child nodes into parents
        parent_morton_codes, first_child = np.unique(morton_codes, return_index=True)
        #assert (np.sort(parent_morton_codes) == parent_morton_codes).all()
        first_child = first_child.astype(np.int32)
        nchild = np.ediff1d(first_child, to_end=nnodes - first_child[-1])

        # Expand groups that have too many children
        excess_children = np.argwhere(nchild > MAX_CHILD)
        if len(excess_children) > 0:
            print 'Expanding %d parent nodes' % len(excess_children)
            parent_morton_parts = np.split(parent_morton_codes, excess_children)
            first_child_parts = np.split(first_child, excess_children)
            nchild_parts = np.split(nchild, excess_children)

            new_parent_morton_parts = parent_morton_parts[:1]
            new_first_child_parts = first_child_parts[:1]

            for morton_part, first_part, nchild_part in zip(parent_morton_parts[1:], 
                                                            first_child_parts[1:],
                                                            nchild_parts[1:]):
                extra_first = np.arange(first_part[0], 
                                        first_part[0]+nchild_part[0], 
                                        MAX_CHILD)
                new_first_child_parts.extend([extra_first, first_part[1:]])
                new_parent_morton_parts.extend([np.repeat(parent_morton_codes[0], 
                                                          len(extra_first)), morton_part[1:]])

            parent_morton_codes = np.concatenate(new_parent_morton_parts)
            first_child = np.concatenate(new_first_child_parts)
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
