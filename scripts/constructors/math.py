def divide(loader, node):
    seq = loader.construct_sequence(node)
    if len(seq) != 2:
        raise ValueError("Must enter 2 values")
    else:
        return seq[0] / seq[1]