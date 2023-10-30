# AN2DL - Project 1
The objective of this project is to create an image classification network specifically for leaf images. The dataset used comprises labeled images of leaves, each identified by its species. A Convolutional Neural Network (CNN) serves as the backbone of our approach.

As the project was set up as a competitive Kaggle challenge among POLIMI students, the final model has been excluded from the repository.

Included in the repository is a framework designed to simplify deep learning workflows, offering automated training and testing for a queue of models.

## Usage
To train and test the networks run "compiler.py"

The repository does not contains any dataset or actual model: you need to move in your own files like follows.

### Directories setup
- Dataset
  - data/MAIN_DATASET_DIR/train
  - data/MAIN_DATASET_DIR/valid
  - data/MAIN_DATASET_DIR/test

- Models architectures
  - models/MODEL_NAME/dataset.yaml
  - models/MODEL_NAME/model.yaml

NOTE: other empty directories will be filled automatically by the compiler.

### Set up the compiler
To apply your custom settings to the directory tree updates the value for the keys in the *"config.yaml"* file.

### Create a model
1. Create */settings/YOUR_MODEL_NAME*
2. Create configurations file inside the new folder: "dataset.yaml" and "model.yaml"
  a. Compile *"dataset.yaml"* to set the parameters for the ImageDataGenerators (preprocessing, data augmentation and so on)
  b. Compile *"model.yaml"* to set the parameters for the model and to define its architecture (input, output and hidden layers)

### Create a jobs queue
To run models enter their names (/settings/MODEL_NAME) under the key "jobs" in the *"queue.yaml"* file

### Comple the jobs queu
To compile the models in the jobs queue just run *"compiler.py"*

## Configuration samples
Here you can find some sample settings.

### dataset.yaml
```
# Directories
# location of the data set
root: &root data                                        # root directory for data
dataset_dir: &dataset_dir !joinPath [*root, dataset811] # folder that contains the data set
# Data
# generic information about the images in the data set
classes: &classes
  - apple
  - blueberry
  - cherry
  - corn
  - grape
  - orange
  - peach
  - pepper
  - potato
  - raspberry
  - soybean
  - squash
  - strawberry
  - tomato
target_size: &target_size !tuple [256, 256]
color_mode: &color_mode rgb
class_mode: &class_mode categorical
batch_size: &batch_size 32
# Reproducibility
seed: &seed 42
# Generators
# settings for the image data generators
generators:
  training:
    data_gen:                                       # settings for ImageDataGenerator (data augmentation and preprocessing)
      rotation_range: 10
      height_shift_range: 50
      width_shift_range: 50
      zoom_range: 0.3
      horizontal_flip: True
      vertical_flip: True
      fill_mode: reflect
      preprocessing_function: VGG16                 # preprocessing for transfer learning with VGG16 (each supernet has its own preprocessing function)
      # rescale: !divid [1, 255.]                     # uncomment when not using transfer learning
    gen:                                            # standard setup for generator
      directory: !joinPath [*dataset_dir, train]    # sub-directory for the train data set (inside main data set directory)
      target_size: *target_size
      classes: *classes
      seed: *seed
      color_mode: *color_mode
      class_mode: *class_mode
      batch_size: *batch_size
      shuffle: True                                 # set to True to use random sample from data set
  validation:                                       # standard setup for generator
    data_gen:                                       # settings for ImageDataGenerator (data augmentation and preprocessing)
      preprocessing_function: VGG16                 # preprocessing for transfer learning with VGG16 (each supernet has its own preprocessing function)
      # rescale: !divid [1, 255.]                     # uncomment when not using transfer learning
    gen:
      directory: !joinPath [*dataset_dir, val]      # sub-directory for the validation data set (inside main data set directory)
      target_size: *target_size
      classes: *classes
      seed: *seed
      color_mode: *color_mode
      class_mode: *class_mode
      batch_size: *batch_size
      shuffle: false                                # set to True to use random sample from data set
  test:
    data_gen:                                       # settings for ImageDataGenerator (data augmentation and preprocessing)
      preprocessing_function: VGG16                 # preprocessing for transfer learning with VGG16 (each supernet has its own preprocessing function)
      # rescale: !divid [1, 255.]                     # uncomment when not using transfer learning
    gen:                                            # standard setup for generator
      directory: !joinPath [*dataset_dir, test]     # sub-directory for the test data set (inside main data set directory)
      target_size: *target_size
      classes: *classes
      seed: *seed
      color_mode: *color_mode
      class_mode: *class_mode
      batch_size: *batch_size
      shuffle: false                                # set to True to use random sample from data set
```

### model.yaml
```
name: model                       # model's name
epochs: 200                         # number of max epochs for training
seed: 42
metrics:
  - accuracy
# Optimizer
optimizer:
  type: Adam
  params:
    lr: !float 1e-4                 # starting learning rate
# Loss function
loss:
  type: CategoricalCrossentropy
  params:
# Callbacks
callbacks:                          # add or remove callbacks
  - type: ReduceLROnPlateau         # callbacks 1
    params:
      monitor: val_loss
      patience: 5
      min_lr: !float 1e-7
      factor: 0.2
  - type: EarlyStopping             # callbacks 2
    params:
      monitor: val_loss
      patience: 10
      restore_best_weights: True
                                    # callbacks N ...
# Input
input_layer:                        # settings for the input layer
  type: Input
  params:
    shape: !tuple [256, 256, 3]
    name: Input
# Output
output_layer:                       # settings for the output layer
  type: Dense
  params:
    units: 14
    activation: softmax
    name: Output
# Hidden
hidden_layers:                      # list of hidden layers (and their settings)
  # supernet
  - type: VGG16
    trainable_layers: 14            # specify the index of the first layer where to start unlocking for training
    params:
      include_top: False
      weights: imagenet
      input_shape: !tuple [256, 256, 3]
  # conv blocks
  - type: Conv2D
    params:
      filters: 32
      kernel_size: !tuple [3, 3]
      strides: !tuple [ 1, 1]
      padding: same
      activation: relu
      name: conv1
  - type: MaxPooling2D
    params:
      pool_size: !tuple [ 2, 2]
      name: pool1
  - type: Conv2D
    params:
      filters: 64
      kernel_size: !tuple [ 3, 3]
      strides: !tuple [ 1, 1]
      padding: same
      activation: relu
      name: conv2
  - type: MaxPooling2D
    params:
      pool_size: !tuple [ 2, 2]
      name: pool2
  - type: Conv2D
    params:
      filters: 128
      kernel_size: !tuple [ 3, 3]
      strides: !tuple [ 1, 1]
      padding: same
      activation: relu
      name: conv3
  - type: MaxPooling2D
    params:
      pool_size: !tuple [ 2, 2]
      name: pool3
  # flattening
  - type: Flatten
    params:
      name: flatten
  - type: Dropout
    params:
      name: drop1
      rate: 0.3
      seed: 42
  # classification
  - type: Dense
    params:
      units: 256
      activation: relu
      name: class
  - type: Dropout
    params:
      rate: 0.3
      seed: 42
      name: drop2
```

### queue.yaml
```
jobs:
  - model41
```

