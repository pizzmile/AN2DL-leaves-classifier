import pprint

import tensorflow as tf

LAYERS_DICT = {
    'Input': tf.keras.layers.Input,
    'Conv2D': tf.keras.layers.Conv2D,
    'MaxPooling2D': tf.keras.layers.MaxPooling2D,
    'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'AveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'Flatten': tf.keras.layers.Flatten,
    'Dense': tf.keras.layers.Dense,
    'Dropout': tf.keras.layers.Dropout,
    'VGG16': tf.keras.applications.vgg16.VGG16,
    'InceptionV3': tf.keras.applications.InceptionV3,
    'ResNet50': tf.keras.applications.resnet50.ResNet50,
    # 'BatchNormalization': tf.keras.layers.BathcNormalization
}

INITIALIZERS_DICT = {
    'GlorotUniform': tf.keras.initializers.GlorotUniform
}

LOSSES_DICT = {
    'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy
}

OPTIMIZERS_DICT = {
    'Adam': tf.keras.optimizers.Adam,
    'SGD': tf.keras.optimizers.SGD
}


def parse_layer_params(raw_params):
    tmp_params = {}
    if raw_params is not None:
        tmp_params = raw_params.copy()
        if 'kernel_initializer' in tmp_params:
            tmp_params['kernel_initializer'] = \
                INITIALIZERS_DICT[tmp_params['kernel_initializer']['type']](tmp_params['kernel_initializer']['params'])
    return tmp_params


def build_layer(type, params):
    return LAYERS_DICT[type](**params)


def get_optimizer(config):
    params = {}
    if 'params' in config.keys():
        if config['params'] is not None:
            params = config['params']
    return OPTIMIZERS_DICT[config['type']](**params)


def get_loss(config):
    params = {}
    if 'params' in config.keys():
        if config['params'] is not None:
            params = config['params']
    return LOSSES_DICT[config['type']](**params)


def build_model(config):
    layer_params = parse_layer_params(config['input_layer']['params'])
    input_layer = build_layer(config['input_layer']['type'], layer_params)

    x = input_layer
    for layer in config['hidden_layers']:
        layer_params = parse_layer_params(layer['params'])
        l = build_layer(layer['type'], layer_params)
        if 'trainable_layers' in layer.keys():
            if layer['trainable_layers'] is not None:
                for i, sub_layer in enumerate(l.layers):
                    sub_layer.trainable = True if i > layer['trainable_layers'] else False
            else:
                for sub_layer in l.layers:
                    sub_layer.trainable = False
        x = l(x)

    layer_params = parse_layer_params(config['output_layer']['params'])
    output_layer = build_layer(config['output_layer']['type'], layer_params)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=config['name'])
    model.compile(loss=get_loss(config['loss']),
                  optimizer=get_optimizer(config['optimizer']),
                  metrics=config['metrics'])

    return model
