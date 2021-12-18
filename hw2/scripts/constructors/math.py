import logging

logger = logging.getLogger(__name__)


def divide(loader, node):
    seq = loader.construct_sequence(node)
    if len(seq) != 2:
        raise ValueError("Must enter 2 values")
    else:
        result = seq[0] / seq[1]
        logger.debug('Yaml ("divide") -> %s', str(result))
        return result
