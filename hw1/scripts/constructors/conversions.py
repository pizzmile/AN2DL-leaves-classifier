def to_tuple(loader, node):
    seq = loader.construct_sequence(node)
    return tuple(seq)

def to_float(loader, node):
    scalar = loader.construct_scalar(node)
    return float(scalar)