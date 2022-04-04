def cal_node_input(patch_size: tuple):
    """
    FUNCTION TO CALCULATE NUMBER OF NODE INPUTS BASED ON IMAGE PATCH SIZE

    Args:
        patch_size: Image patch size
    """
    n_dim = len(patch_size)
    node_input = 1

    for i in range(n_dim):
        node_input = node_input * patch_size[i]

    return node_input
