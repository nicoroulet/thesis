""" This file contains the different segmentation models. """

import os

import Metrics

import numpy as np

import UNet


from keras.callbacks import ModelCheckpoint


class MultiUNet:

  def __init__(self, tasks, net_depth=4):
    """ Builds a model with one UNet per task.

    Args:
        tasks: list of dicts storing: {name, labels}
               Tasks should be given in order of priority. Class 0 of all tasks
               is assumed to be background (thus, identical between tasks).
               The rest of the labels are given in the field labels, in order.
               Example, if possible labels are bg, gray matter, white matter,
               then labels should be ["gray matter", "white matter"]

    Example:
      net = MultiUnet({"tumor": ["non-enhancing tumor",  # label 1 of 1st net
                                 "enhancing tumor"],     # label 2 of 1st net
                       "anatomical": ["CSF",             # label 1 of 2nd net
                                      "White matter",    # label 2 of 2nd net
                                      "Gray matter"]})   # label 3 of 2nd net
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
                    loss=Metrics.continuous.sparse_dice_loss,
                    # loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy',
                             Metrics.continuous.sparse_dice_coef,
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
                                         save_best_only=True)

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
    """Evaluates the model on the given generator.

    Args:
        generator: Batch generator used for evaluation. Important: labels used
            in the segmentation yielded by the generator should match those in
            self.labels, in the same order.
        metrics (list): list of metrics to uvaluate on. Each metric is a
            function that takes y_true, y_pred and returns a value
        steps (5, optional): number of evaluation steps.
    """
    metrics = []
    for i, (x, y_true) in zip(range(steps), generator):
      y_pred = self.predict(x)
      metrics.append([metric(y_true, y_pred) for metric in metrics])
    return np.mean(metrics, axis=0)

  def predict(self, x):
    """ Returns a prediction for the given input.

    Args:
        X (Numpy array): input 3d image.
    """
    predictions = {}
    for task_name in self.task_names:
      predictions[task_name] = self.nets[task_name].predict(x)
    return self.merge_segmentations(predictions)
