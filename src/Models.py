"""This file contains the different segmentation models."""

import os

import Metrics
import Tools
import Logger
import Tools

import numpy as np
import fractions
import functools

# import tensorflow as tf

import keras
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, \
                         Add, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint


def lcm(l):
  """Least common multiplier of elements in a list.

  Args:
      l (list): The list

  Returns:
      int: the lcm of the list

  """
  def _lcm(a, b):
    return a * b / fractions.gcd(a, b)
  return functools.reduce(_lcm, l)


def build_unet(n_classes, depth=4, base_filters=32, n_channels=1):
  """Build a UNet neural network and returns the input and output layers.

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
      # 'activation': 'relu',
      # TODO: experiment with regularizers and initializers.
      'kernel_initializer': 'he_normal',
      'kernel_regularizer': keras.regularizers.l2(.005)
  }

  inputs = Input((None, None, None, n_channels))
  x = inputs

  n_filters = base_filters

  # Layers tat will be used in up-convolution
  layer_outputs = []
  # TODO: try adding batch normalization layers.

  # Convolution layers
  for layer in range(depth - 1):
    # x = Dropout(.1)(x)
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    layer_outputs.append(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    n_filters *= 2

  # Bottom layers
  x = Dropout(0.3)(x)
  x = Conv3D(filters=n_filters, **conv_params)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv3D(filters=n_filters, **conv_params)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # Transposed Convolution layers (up-convolution)
  for layer in reversed(range(depth - 1)):
    n_filters //= 2
    # x = Dropout(.1)(x)
    x = Conv3DTranspose(filters=n_filters, kernel_size=(2, 2, 2),
                        strides=(2, 2, 2))(x)
    x = Add()([x, layer_outputs.pop()])
    # x = concatenate([x, layer_outputs.pop()])
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

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
        tuple: (ground truth, prediction), both segmentations categorically encoded.

    """
    for (i, (x, y)) in zip(range(steps), generator):
      xmin, xmax, ymin, ymax, zmin, zmax = Tools.get_bounding_box(x, generator.patch_multiplicity)
      x_cropped = x[:, xmin:xmax, ymin:ymax, zmin:zmax, :]
      y_pred_cropped = self.predict(x_cropped)
      y = np.squeeze(y, axis=-1)
      y_pred = np.zeros_like(y)
      y_pred[:, xmin:xmax, ymin:ymax, zmin:zmax] = np.argmax(y_pred_cropped, axis=-1)
      yield x, y, y_pred


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

  def __init__(self, datasets, depth=4, loss=None):
    """Build a model with one UNet per task.

    TODO: rewrite docstring.
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
    # TODO: manage passing different losses to each dataset.
    Logger.info('Building MultiUNet from datasets:', *(d.name for d in datasets))
    Logger.debug('Segmentation labels:', *('%d - %s,' % (i + 1, label) for (i, label) in enumerate(
                                                       l for d in datasets for l in d.classes[1:])))
    self.datasets = datasets
    self.nets = {}
    self.savefiles = {}
    self.callbacks = {}
    for dataset in datasets:
      name = dataset.name
      self.nets[name] = UNet(dataset.n_classes, depth=depth, n_channels=dataset.n_modalities)
      self.nets[name].compile(
                              # loss=Metrics.continuous.dice_loss,
                              loss='sparse_categorical_crossentropy',
                              optimizer=keras.optimizers.Adam(lr=0.0003),#'adam',
                              metrics=['accuracy',
                                       Metrics.dice_coef,
                                       Metrics.mean_dice_coef])

      savedir = Tools.get_dataset_savedir(dataset, loss)
      Tools.ensure_dir(savedir)
      savefile = savedir + "/best_weights.h5"
      secondary_savefile = savedir + "/weights.h5"
      if os.path.exists(savefile):
        self.savefiles[name] = savefile
      elif os.path.exists(secondary_savefile):
        self.savefiles[name] = secondary_savefile
      else:
        Logger.warning('WARNING: weights file not found at %s nor %s.' % (savefile,
                                                                          secondary_savefile))
      # TODO: add epoch count, metrics, tensorboard.
      if name in self.savefiles:
        Logger.info('Loading weights for dataset', name, 'from', self.savefiles[name])
        self.nets[name].load_weights(self.savefiles[name])

      model_checkpoint = ModelCheckpoint(savefile,
                                         monitor='val_loss',
                                         save_best_only=True)

      self.callbacks[name] = [model_checkpoint]

    # self.task_names = [task['name'] for task in tasks]
    # self.nets = {}
    # self.savefiles = {}
    # self.callbacks = {}
    # self.labels = ["background"]
    # for task in tasks:
    #   name = task["name"]
    #   n_classes = len(task["labels"]) + 1
    #   self.labels.append(task["labels"])
    #   self.nets[name] = UNet.UNet(n_classes)
    #   self.nets[name].compile(
    #                           loss=Metrics.continuous.dice_loss,
    #                           # loss='sparse_categorical_crossentropy',
    #                           optimizer='adam',
    #                           metrics=['accuracy',
    #                                    Metrics.continuous.dice_coef,
    #                                    Metrics.continuous.mean_dice_coef)

    #   savedir = "weights"
    #   if not os.path.exists(savedir):
    #     os.mkdir(savedir)
    #   savefile = savedir + "/multiunet_%s.h5" % name.lower().replace(' ', '_')
    #   self.savefiles[name] = savefile
    #   if os.path.exists(savefile):
    #     self.nets[name].load_weights(savefile)

    #   model_checkpoint = ModelCheckpoint(savefile,
    #                                      monitor='loss',
    #                                      save_best_only=False)

    #   self.callbacks[name] = [model_checkpoint]

  def fit_generator(self, task_name, *args, **kwargs):
    """Call the fit_generator corresponding to task_name."""
    Logger.info("Fitting task: %s..." % task_name)
    kwargs['callbacks'] = (kwargs.get('callbacks', []) + self.callbacks[task_name])
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
    for dataset in self.datasets:
      Logger.debug("Multiunet: applying predictions on", dataset.name)
      Y = y_per_task[dataset.name]
      merge += (Y + (Y != 0) * (label_offset)) * (merge == 0)
      label_offset += self.nets[dataset.name].n_classes - 1
    return merge

  def evaluate_generator(self, generator, metrics, steps=5, modalities=None):
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
    for y_true, y_pred in self.predict_generator(generator, steps, modalities):
      metric_values.append([metric(y_true, y_pred) for metric in metrics])
    return np.mean(metric_values, axis=0)

  def predict(self, x, modalities=None):
    """Return a prediction for the given input.

    Args:
        x (Numpy array): input image.
        modalities (list, optional): modalities encoded in the las dimension of x. If `None`,
            then modalities from x are assumed to match modalities on the training datasets.

    Returns:
        numpy array: predictions, categorically encoded.

    """
    predictions = {}
    for dataset in self.datasets:
      if modalities is not None:
        x_filtered = Tools.filter_modalities(modalities, dataset.modalities, x)
      else:
        x_filtered = x
      Logger.debug('predicting x with shape = ', x_filtered.shape)
      predictions[dataset.name] = np.argmax(self.nets[dataset.name].predict(x_filtered), axis=-1)
    return self.merge_segmentations(predictions)

  def predict_generator(self, generator, steps=5, modalities=None):
    """Generate predictions from a given generator.

    Args:
        generator (BatchGenerator): image generator
        steps (int, optional): number of images to predict.
        modalities (list, optional): list of modalities for on the given generator.

    Yields:
        tuple: (y_true, y_pred) - ground truth and predicted labels.

    """
    for i, (x, y_true) in zip(range(steps), generator):

      xmin, xmax, ymin, ymax, zmin, zmax = Tools.get_bounding_box(x, generator.patch_multiplicity)
      x_cropped = x[:, xmin:xmax, ymin:ymax, zmin:zmax, :]
      y_pred_cropped = self.predict(x_cropped, modalities)
      y_true = np.squeeze(y_true, axis=-1)
      y_pred = np.zeros_like(y_true)
      y_pred[:, xmin:xmax, ymin:ymax, zmin:zmax] = y_pred_cropped
      yield x, y_true, y_pred

  @property
  def patch_multiplicity(self):
    """Joint patch multiplicity for all nets."""
    return int(lcm([net.patch_multiplicity for net in self.nets.values()]))

  @property
  def n_classes(self):
    """Number of classes of multiunet, which is the union of the classes from the different nets."""
    return sum(dataset.n_classes - 1 for dataset in self.datasets) + 1
