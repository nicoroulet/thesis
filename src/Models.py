""" This file contains the different segmentation models. """

import os

import Metrics

import numpy as np

# import tensorflow as tf

from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, \
                         concatenate, Add, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint


def build_unet(n_classes, depth=4, base_filters=32, n_channels=1):
  """ Build a UNet neural network and returns the input and output layers.

  Args:
      n_classes (int): number of output classes (segmentation labels).
      depth (int, optional): number of UNet levels. This is not the number
          of layers of the neural network, a UNet layer is the combination of
          convolution layers and maxpooling/upconvolution layers.
      base_filters (int, optional): number of filters on the initial layers.
          The number of filters is doubled on each maxpooling and halved on
          each upconvolution.
  Return:
    inputs, outputs: input and output layers.
  """
  conv_params = {
      'kernel_size': (3, 3, 3),
      'strides': (1, 1, 1),
      'padding': 'same',  # TODO: consider 'valid' (no padding),
                          # as suggested by paper.
      'activation': 'relu',
      # TODO: experiment with regularizers and initializers.
      'kernel_initializer': 'he_normal'
      # 'kernel_regularizer': keras.regularizers.l2(.0001)
  }

  inputs = Input((None, None, None, n_channels))
  x = inputs

  n_filters = base_filters

  # Layers tat will be used in up-convolution
  layer_outputs = []
  # TODO: try adding batch normalization layers.

  # Convolution layers
  for layer in range(depth - 1):
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)
    layer_outputs.append(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    n_filters *= 2

  # Bottom layers
  x = Dropout(.3)(x)
  x = Conv3D(filters=n_filters, **conv_params)(x)
  x = Conv3D(filters=n_filters, **conv_params)(x)

  # Transposed Convolution layers (up-convolution)
  for layer in reversed(range(depth - 1)):
    n_filters //= 2
    x = Conv3DTranspose(filters=n_filters, kernel_size=(2, 2, 2),
                        strides=(2, 2, 2))(x)
    x = Add()([x, layer_outputs.pop()])
    # x = concatenate([x, layer_outputs.pop()])
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)

  # Final layer
  x = Conv3D(filters=n_classes,
             kernel_size=(1, 1, 1),
             padding='same',
             activation='softmax')(x)

  return inputs, x


class UNet(Model):
  """UNet model, taken from https://arxiv.org/abs/1505.04597."""

  def __init__(self, n_classes, depth=4, base_filters=32, n_channels=1):
    """Create UNet model.

    Args:
        n_classes (int): number of output classes (segmentation labels).
        depth (int, optional): number of UNet levels. This is not the number
            of layers of the neural network, a UNet layer is the combination of
            convolution layers and maxpooling/upconvolution layers.
        base_filters (int, optional): number of filters on the initial layers.
            The number of filters is doubled on each maxpooling and halved on
            each upconvolution.
    """
    inputs, outputs = build_unet(n_classes,
                                 depth=depth,
                                 base_filters=base_filters,
                                 n_channels=n_channels)
    super(UNet, self).__init__(inputs, outputs)
    self.n_classes = n_classes
    self.depth = depth
    self.base_filters = base_filters
    self.patch_multiplicity = 2 ** depth

  def predict_generator(self, generator, steps=1):
    """Generate predictions for full volume images generator.

    Args:
        generator (BatchGenerator): Full volume batch generator.
        steps (int): number of images to evaluate

    Yields:
        tuple: (ground truth, prediction), both categorically encoded.
    """
    for i in range(steps):
      x, y = next(generator)
      xmin, xmax, ymin, ymax, zmin, zmax = generator.get_bounding_box(x)
      x_cropped = x[:, xmin:xmax, ymin:ymax, zmin:zmax, :]
      y_pred_cropped = self.predict(x_cropped)
      y = np.squeeze(y, axis=-1)
      y_pred = np.zeros_like(y)
      y_pred[:, xmin:xmax, ymin:ymax, zmin:zmax] = np.argmax(y_pred_cropped, axis=-1)
      yield y, y_pred


class MultiUNet:
  """Muliti task segmentation model based on multiple UNets.

  An independent UNet is trained per each task. Upon evaluation, predictions
  from each net are merged by an order of priorities.

  Attributes:
      callbacks (dict): stores callbacks for each network.
      labels (list): concatenation of labels from all tasks, in priority order.
      nets (dict): maps task names to the corresponding network.
      savefiles (dict): maps task names to the savefiles used to store weights.
      task_names (list): task names, in priority order.
  """

  def __init__(self, tasks, net_depth=4):
    """ Build a model with one UNet per task.

    Args:
        tasks: list of dicts storing: {name, labels}
               Tasks should be given in order of priority. Class 0 of all tasks
               is assumed to be background (thus, identical between tasks).
               The rest of the labels are given in the field labels, in order.
               Example, if possible labels are bg, gray matter, white matter,
               then labels should be ["gray matter", "white matter"]

    Example:
      tumor_tasks = [
         {"name": "tumor",
          "labels": ["necrosis",
                     "edema",
                     "nonenhancing tumor",
                     "enhancing tumor"]},
         {"name": "anatomical",
          "labels": ["CSF",
                     "White matter",
                     "Gray matter"]}
        ]
    """
    self.task_names = [task['name'] for task in tasks]
    self.nets = {}
    self.savefiles = {}
    self.callbacks = {}
    self.labels = ["background"]
    for task in tasks:
      name = task["name"]
      n_classes = len(task["labels"]) + 1
      self.labels.append(task["labels"])
      self.nets[name] = UNet.UNet(n_classes)
      self.nets[name].compile(
                              loss=Metrics.continuous.dice_loss,
                              # loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy',
                                       Metrics.continuous.dice_coef,
                                       Metrics.continuous.mean_dice_coef()])

      savedir = "weights"
      if not os.path.exists(savedir):
        os.mkdir(savedir)
      savefile = savedir + "/multiunet_%s.h5" % name.lower().replace(' ', '_')
      self.savefiles[name] = savefile
      if os.path.exists(savefile):
        self.nets[name].load_weights(savefile)

      model_checkpoint = ModelCheckpoint(savefile,
                                         monitor='loss',
                                         save_best_only=False)

      self.callbacks[name] = [model_checkpoint]

  def fit_generator(self, task_name, *args, **kwargs):
    """ Call the fit_generator corresponding to task_name. """
    print("Fitting task: %s..." % task_name)
    kwargs['callbacks'] = (kwargs.get('callbacks', []) +
                           self.callbacks[task_name])
    return self.nets[task_name].fit_generator(*args, **kwargs)

  def merge_segmentations(self, y_per_task):
    """Merge segmentations from the multiple nets.

    Args:
        y_per_task (dict): dictionary (task_name: Y). Dictionary that contains
            predictions for each task. Y is given in sparse notation (one label
            per pixel, not the scores).

    Returns:
        Numpy array: resulting segmentation.
    """

    # TODO: manage overlapping label definitions (Ys should be disjoint except
    # for background)
    merge = np.zeros_like(next(iter(y_per_task.values())))
    label_offset = 0
    for task_name in self.task_names:
      Y = y_per_task[task_name]
      merge += (Y + (Y != 0) * label_offset) * (merge == 0)
      label_offset += self.nets[task_name].n_classes
    return merge

  def evaluate_generator(self, generator, metrics, steps=5):
    """Evaluate the model on the given generator.

    Args:
        generator: Batch generator used for evaluation. Important: labels used
            in the segmentation yielded by the generator should match those in
            self.labels, in the same order.
        metrics (list): list of metrics to evaluate on. Each metric is a
            function that takes y_true, y_pred and returns a value
        steps (5, optional): number of evaluation steps.

    Returns:
        numpy array: averaged metrics for all evaluated images.
    """
    metric_values = []
    for i, (x, y_true) in zip(range(steps), generator):
      y_true = y_true.reshape(y_true.shape[:-1])
      y_pred = self.predict(x)
      metric_values.append([metric(y_true, y_pred) for metric in metrics])
    return np.mean(metric_values, axis=0)

  def predict(self, x):
    """ Return a prediction for the given input.

    Args:
        X (Numpy array): input 3d image.
    """
    predictions = {}
    for task_name in self.task_names:
      predictions[task_name] = np.argmax(self.nets[task_name].predict(x),
                                         axis=-1)
    return self.merge_segmentations(predictions)
