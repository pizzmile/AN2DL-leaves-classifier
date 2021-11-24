import os


def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def join_path(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)
