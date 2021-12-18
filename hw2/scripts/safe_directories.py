import os
import logging

logger = logging.getLogger(__name__)


def createDirectoryTree(tree):
    for directory in tree:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info("Created directory {directory}".format(directory=directory))
