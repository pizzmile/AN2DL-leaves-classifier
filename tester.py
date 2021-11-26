import csv
import gc
import os
import yaml
import pprint
import datetime
import tensorflow as tf


from scripts import build_generators, join, join_path, to_tuple, to_float, divide

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


def test_model(el):
    # Load and test models
    # --------------------
    # Load the model
    pprint.pprint(el)
    model = tf.keras.models.load_model(os.path.join(config['directories']['models_dir'], el))
    # Get model name
    name = el.split('@')[0]
    with open(os.path.join(config['directories']['configs_dir'], name, 'dataset.yaml')) as file:
        gen_config = yaml.load(file, Loader=yaml.FullLoader)
        file.close()

    # Build data generator
    _, _, test_gen = build_generators(gen_config)

    # Evaluate the model
    evaluation = model.evaluate(test_gen, return_dict=True)

    # Save the result
    with open(os.path.join(config['directories']['models_dir'], 'tests.csv'), 'a') as file:
        data = {**{'model': el}, **evaluation}

        writer = csv.DictWriter(file, fieldnames=[key for key in data.keys()])
        writer.writerow(data)
        file.close()

    gc.collect()


if __name__ == '__main__':
    # Load directory tree
    # -------------------
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        file.close()

    models_folders = next(os.walk(config['directories']['models_dir']))[1]
    pprint.pprint(models_folders)
    user_input = input('Enter a name from the list to test it or press ENTER to test all models: ')

    try:
        if user_input == 'all' or user_input == 'ALL':
            for elem in models_folders:
                test_model(elem)
        else:
            test_model(user_input)
    except FileNotFoundError as e:
        pprint.pprint("Unable to locate the model.")
        pprint.pprint(e)
