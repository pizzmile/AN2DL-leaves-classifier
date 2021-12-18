import os


def create_dir(exps_dir):
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)
