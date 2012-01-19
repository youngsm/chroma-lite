from chroma.bvh.bvh import BVH, node_area
from chroma.gpu.bvh import create_leaf_nodes, merge_nodes, concatenate_layers

def make_simple_bvh(mesh, degree):
    '''Returns a BVH tree created by simple grouping of Morton ordered nodes.
    '''
    world_coords, leaf_nodes, morton_codes = \
        create_leaf_nodes(mesh, round_to_multiple=degree)

    # rearrange in morton order
    argsort = morton_codes.argsort()
    leaf_nodes = leaf_nodes[argsort]
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

    for i, (layer_start, layer_end) in enumerate(zip(layer_bounds[:-1], 
                                                     layer_bounds[1:])):
        print i, node_area(nodes[layer_start:layer_end]) * world_coords.world_scale**2
    

    
