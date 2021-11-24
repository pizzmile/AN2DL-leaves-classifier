import gc
import pprint

import yaml


from scripts import create_dir
from scripts import perform_job
from scripts import join, join_path, to_tuple, to_float, divide


# Setup
# -----
# Uncomment to use CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add yaml custom constructors
yaml.add_constructor('!join', join)
yaml.add_constructor('!joinPath', join_path)
yaml.add_constructor('!tuple', to_tuple)
yaml.add_constructor('!float', to_float)
yaml.add_constructor('!divide', divide)


# Clean ram
gc.collect()

# Load compiler configuration
with open(f'config.yaml') as file:
    compiler_config = yaml.load(file, Loader=yaml.FullLoader)
    directories = compiler_config['directories']
    file.close()

# Safe create directory tree
for key in directories.keys():
    exps_dir = directories[key]
    create_dir(exps_dir)


# Main
# ----
if __name__ == '__main__':
    # Load queue
    with open(f'queue.yaml') as file:
        job_queue = yaml.load(file, Loader=yaml.FullLoader)['jobs']
        file.close()
    # Complete jobs
    for work in job_queue:
        perform_job(work, directories, silent=False)
        gc.collect()
