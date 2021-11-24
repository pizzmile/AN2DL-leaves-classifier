import os
import pprint

import telegram_send
import yaml
import random
import datetime
import numpy as np
import tensorflow as tf
import visualkeras


from .builders import build_generators, build_model
from .callbacks import create_output_callbacks


CALLBACKS_DICT = {
    'ReduceLROnPlateau': tf.keras.callbacks.ReduceLROnPlateau,
    'EarlyStopping': tf.keras.callbacks.EarlyStopping
}


def perform_job(model_name, directories, silent=True, tg_silent=True):
    configs_dir, logs_dir, models_dir = directories['configs_dir'], directories['logs_dir'], directories['models_dir']

    if not silent:
        pprint.pprint(model_name)
    if not tg_silent:
        telegram_send.send(messages=[u'\U0001f300' + f" <b>Starting work:</b> <i>{model_name}</i>"], parse_mode='HTML')

    # Setup model
    # -----------
    # Load model configuration
    with open(f'{configs_dir}\{model_name}\dataset.yaml') as file:
        gen_config = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    with open(f'{configs_dir}\{model_name}\model.yaml') as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    # Set random seed for reproducibility
    seed = gen_config['seed']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Build generators
    # ----------------
    train_gen, valid_gen, test_gen = build_generators(gen_config)

    # Build model
    # -----------
    model = build_model(model_config)
    # Get image
    image = visualkeras.layered_view(model, legend=True, spacing=20, scale_xy=5,
                                     to_file=f"{configs_dir}/{model_name}/diagram.png")
    if not silent:
        model.summary()
        image.show()
    if not tg_silent:
        with open(f"{configs_dir}/{model_name}/diagram.png", 'rb') as image_file:
            telegram_send.send(images=[image_file], silent=True)
            image_file.close()

    # Create callbacks
    # ----------------
    callbacks = create_output_callbacks(model_name=model_name, logs_dir=logs_dir)
    # Add optional callbacks
    if 'callbacks' in model_config.keys():
        if len(model_config['callbacks']) > 0:
            for callback in model_config['callbacks']:
                params = callback['params'] if callback['params'] is not None else {}
                callbacks.append(CALLBACKS_DICT[callback['type']](**params))

    # Train the model
    # ---------------
    model.fit(
        x=train_gen,
        epochs=model_config['epochs'],
        validation_data=valid_gen,
        callbacks=callbacks,
    )
    # Save best epoch models
    now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    model.save(f'{models_dir}/{model_name}{now}')

    # Evaluate the model
    # ------------------
    evaluation = model.evaluate(test_gen, return_dict=True)
    if not silent:
        pprint.pprint(evaluation)

