from chroma.bvh.bvh import BVH, node_area
from chroma.gpu.bvh import create_leaf_nodes, merge_nodes, concatenate_layers

def make_simple_bvh(mesh, degree):
    '''Returns a BVH tree created by simple grouping of Morton ordered nodes.
    '''
    world_coords, leaf_nodes, morton_codes = \
        create_leaf_nodes(mesh, round_to_multiple=degree)

    # rearrange in morton order. NOTE: morton_codes can be shorter than
    # leaf_nodes if dummy padding nodes were added at the end!
    argsort = morton_codes.argsort()
    leaf_nodes[:len(argsort)] = leaf_nodes[argsort]
    assert len(leaf_nodes) % degree == 0

    # Create parent layers
    layers = [leaf_nodes]
    while len(layers[0]) > 1:
        parent = merge_nodes(layers[0], degree=degree)
        if len(parent) > 1:
            assert len(parent) % degree == 0
        layers = [parent] + layers

    # How many nodes total?
    nodes, layer_bounds = concatenate_layers(layers)

    return BVH(degree, world_coords, nodes, layer_bounds[:-1])


    
