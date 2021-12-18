import logging

logger = logging.getLogger(__name__)


def to_tuple(loader, node):
    seq = loader.construct_sequence(node)
    logger.debug('Yaml ("to_tuple") -> %s', str(seq))
    return tuple(seq)


def to_float(loader, node):
    scalar = loader.construct_scalar(node)
    logger.debug('Yaml ("to_float") -> %s', str(scalar))
    return float(scalar)


def map_to_list(loader, node):
    mapping = loader.construct_mapping(node)
    seq = [mapping[key] for key in mapping.keys()]
    logger.debug('Yaml ("map_to_list") -> %s', str(seq))
    return seq
