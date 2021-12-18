import logging
import os

logger = logging.getLogger(__name__)


def join(loader, node):
    seq = loader.construct_sequence(node)
    logger.debug('Yaml ("join") -> %s', str(seq))
    return ''.join([str(i) for i in seq])


def join_path(loader, node):
    seq = loader.construct_sequence(node)
    logger.debug('Yaml ("join_path") -> %s', str(seq))
    return os.path.join(*seq)
