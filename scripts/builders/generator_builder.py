from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf


PREPROCESSING_FUN_DICT = {
    'VGG16': tf.keras.applications.vgg16.preprocess_input,
    'Inception': tf.keras.applications.inception_v3.preprocess_input,
    'ResNet50': tf.keras.applications.resnet50.preprocess_input
}


def parse_preprocessing_parameters(raw_params):
    tmp_params = {}
    if raw_params is not None:
        tmp_params = raw_params.copy()
        if 'preprocessing_function' in tmp_params.keys():
            tmp_params['preprocessing_function'] = PREPROCESSING_FUN_DICT[tmp_params['preprocessing_function']]
    return tmp_params


def build_generators(configuration):
    preprocessing_params = parse_preprocessing_parameters(configuration['generators']['training']['data_gen'])
    train_data_gen = ImageDataGenerator(
        **preprocessing_params)
    train_gen = train_data_gen.flow_from_directory(
        **configuration['generators']['training']['gen'])

    preprocessing_params = parse_preprocessing_parameters(configuration['generators']['validation']['data_gen'])
    valid_data_gen = ImageDataGenerator(
        **preprocessing_params)
    valid_gen = valid_data_gen.flow_from_directory(
        **configuration['generators']['validation']['gen'])

    preprocessing_params = parse_preprocessing_parameters(configuration['generators']['test']['data_gen'])
    test_data_gen = ImageDataGenerator(
        **preprocessing_params)
    test_gen = test_data_gen.flow_from_directory(
        **configuration['generators']['test']['gen'])

    return train_gen, valid_gen, test_gen
