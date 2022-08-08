import random
import os
import shutil

import numpy as np
import tensorflow as tf

def set_seed(seed):
    """
    Set relevant random seeds

    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clear_path(path):
    """
    Delete any files or directories from a path

    :param path: Path to remove
    :type path: str
    :return: None
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

def total_system_ram():
    """
    Get total system ram, in bytes

    :return:
    """
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')