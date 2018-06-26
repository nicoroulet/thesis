"""Collection of metrics to compare segmentations."""

import keras
import keras.backend as K
from keras.callbacks import Callback

import numpy as np

import tensorflow as tf

# from medpy.metrics.binary import

import importlib.util
spec = importlib.util.spec_from_file_location("MetricsMonitor",
                      "../../blast/blast/cnn/MetricsMonitor.py")
MetricsMonitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MetricsMonitor)

################################################################################
############################## CONTINUOUS METRICS ##############################
################################################################################

# Continuous metrics operate on the class scores.


class ContinuousMetrics:
  """Continuous metrics, tensor-based implementation.

  Can be used for training because gradients can be computed for backpropagation
  """

  @staticmethod
  def dice_coef(y_true, y_pred):
    """Compute the dice coefficient between two labelings.

    Dice coefficient is defined as 2 * tp / (2 * tp + fp + fn), where
    tp = true positives, fp = false positives, fn = false negatives.
    In a multiclass problem we consider all nonzero labels to be positive
    (0 is background).

    Args:
        y_true (tensor): ground truth segmentation, in sparse notation (one
            category label per voxel)
        y_pred (tensor): predicted segmentation in categorical notation (a
            per-class score for each voxel)

    Returns:
        tensor: coefficient value
    """
    # shape notation: b = batch index, d = depth, w = width, h = height
    # y_true shape: (b, w, h, d, 1)
    # y_pred shape: (b, w, h, d, n_classes)
    n_classes = K.shape(y_pred)[-1]
    # n_classes = K.print_tensor(n_classes, message="n_classes = ")

    # y_true_int shape: (b, w, h, d)
    y_true_int = K.cast(K.squeeze(y_true, axis=-1), 'int32')

    # positive_mask shape: (b, w, h, d, 1)
    positive_mask = K.clip(y_true, 0, 1)
    # correct_mask shape: (b, w, h, d, n_classes)
    correct_mask = K.one_hot(y_true_int, n_classes)
    # correct_scores shape: (b, w, h, d, n_classes)
    correct_scores = correct_mask * y_pred
    # Scores given to the correct labels
    # correct_scores_sum shape: (b, w, h, d, 1)
    correct_scores_sum = K.sum(correct_scores, axis=-1, keepdims=True)
    # True Positive: sum of correct scores assigned to positive label
    tp = K.sum(positive_mask * correct_scores_sum)

    bg_scores = y_pred[..., 0:1]
    # Positive scores given to the background labels
    fp = K.sum((1 - positive_mask) * (1 - bg_scores))

    # Background scores given to the positive labels.
    fn = K.sum(bg_scores * positive_mask)

    num = 2 * tp
    den = num + fn + fp + 1e-4
    # num = K.print_tensor(num, message="num = ")
    # den = K.print_tensor(den, message="den = ")
    return num / den

  @staticmethod
  def dice_loss(y_true, y_pred):
    """Dice loss, same as dice coefficient but decreasing."""
    return 1 - ContinuousMetrics.dice_coef(y_true, y_pred)

  @staticmethod
  def mean_dice_coef(ignore_background=True):
    """Metric that computes the averaged one-versus-all dice coefficient.

    The coefficient between two labelings is computed by averaging the dice
    coefficient of each label against the rest.

    Args:
        ignore_background (optional, bool): if True, dice for the background
            label (0) is not averaged.

    Returns:
        tensor: coefficient value
    """
    def _mean_dice_coef(y_true, y_pred):
      # y_true is in sparse notation (one category number per voxel), while
      # y_pred is in categorical notation (a probability distribution for each
      #                                    voxel)
      n_classes = K.int_shape(y_pred)[-1]
      labels = range(ignore_background, n_classes)
      n_labels = n_classes - ignore_background
      mean = None
      for label in labels:
        # label is the positive label.

        positive_mask = K.equal(y_true, label)
        positive_mask = K.cast(positive_mask, 'float')
        negative_mask = 1 - positive_mask
        # True Positive: sum of correct scores assigned to positive label
        tp = K.sum(y_pred[..., label:label + 1] * positive_mask)
        # / (K.sum(positive_mask) + 1e-4)
        # False Positive: sum of positive scores assigned to negative labels
        fp = K.sum(y_pred[..., label:label + 1] * (negative_mask))
        # / (K.sum(negative_mask) + 1e-4)
        # False Negative: sum of negative scores assigned to positive label
        # This assumes that the sum of scores is 1 (output from softmax)
        fn = K.sum((1 - y_pred[..., label:label + 1]) * positive_mask)
        # / (K.sum(positive_mask) + 1e-4)
        num = 2 * tp
        den = num + fp + fn + 1e-5
        coef = num / den

        if mean is None:
          mean = coef
        else:
          mean += coef
      mean *= 1 / n_labels
      return mean

    return _mean_dice_coef

  @staticmethod
  def mean_dice_loss(ignore_background=True):
    """Mean dice loss, same as mean dice coefficient but decreasing."""

    def _mean_dice_loss(y_true, y_pred):
      return 1 - ContinuousMetrics.mean_dice_coef(ignore_background)(y_true, y_pred)

    return _mean_dice_loss

################################################################################
############################## DISCRETE METRICS ################################
################################################################################

# Discrete metrics operate on the predicted classes.


class DiscreteMetrics:
  """Discrete metrics, implemented with numpy arrays.

  Cannot be used as training loss. Discrete means it operates on the predicted
  labels, not the predicted scores.
  """

  # @staticmethod
  # def dice_coef(y_true, y_pred):
  #   """ Compute the dice coefficient between two labelings.

  #   Dice coefficient is defined as 2 * tp / (2 * tp + fp + fn), where
  #   tp = true positives, fp = false positives, fn = false negatives.
  #   In a multiclass problem we consider all nonzero labels to be positive
  #   (0 is background).

  #   Args:
  #     y_true (Numpy array): ground truth segmentation, in sparse notation (one
  #       category label per voxel)
  #       y_pred (Numpy array): predicted segmentation, categorical notation
  #   """

  #   positive_mask = K.
  #   positive_mask = (y_true != 0)

  #   tp = np.sum(positive_mask * (y_true == y_pred))

  #   fp_fn = np.sum(y_true != y_pred)

  #   num = 2 * tp
  #   den = num + fp_fn + 1e-4
  #   return num / den

  # @staticmethod
  # def dice_loss(y_true, y_pred):
  #   """Dice loss, same as dice coefficient but decreasing."""
  #   return 1 - DiscreteMetrics.dice_coef(y_true, y_pred)

  @staticmethod
  def mean_dice_coef(n_classes, ignore_background=True, verbose=False):
    """Metric that computes the averaged one-versus-all dice coefficient.

    The coefficient between two labelings is computed by averaging the dice
    coefficient of each label against the rest.

    Args:
        n_classes: number of possible distinct labels.
        ignore_background: if True, dice for the background label (0) is not
                           averaged.
    """
    def _mean_dice_coef(y_true, y_pred):
      # y_true is in sparse notation (one category number per voxel)
      # y_pred is in sparse notation
      labels = range(ignore_background, n_classes)
      n_labels = n_classes - ignore_background
      mean = 0
      for label in labels:
        # label is the positive label.
        true_mask = K.cast(K.equal(y_true, label), 'float')
        pred_mask = K.cast(K.equal(y_pred, label), 'float')

        tp = K.sum(true_mask * pred_mask)
        fp = K.sum((1 - true_mask) * pred_mask)
        fn = K.sum(true_mask * (1 - pred_mask))
        if verbose:
          tp = K.print_tensor(tp, message='tp = ')
          fp = K.print_tensor(fp, message='fp = ')
          fn = K.print_tensor(fn, message='fn = ')
        num = 2 * tp
        den = num + fp + fn + 1e-4
        coef = num / den

        mean += coef
      mean *= 1 / n_labels
      return mean

    return _mean_dice_coef

  @staticmethod
  def mean_dice_loss(n_classes, ignore_background=True):
    """Mean dice loss, same as mean dice coefficient but decreasing."""

    def _mean_dice_loss(y_true, y_pred):
      return 1 - DiscreteMetrics.mean_dice_coef(n_classes, ignore_background)(y_true, y_pred)

    return _mean_dice_loss


################################################################################
################################ NUMPY METRICS #################################
################################################################################

# Numpy metrics receive numpy arrays instead of tensors and operate on the
# predicted classes.


class NumpyMetrics:
  """Discrete metrics, implemented with numpy arrays.

  Cannot be used as training loss. Discrete means it operates on the predicted
  labels, not the predicted scores.
  """

  @staticmethod
  def accuracy(y_true, y_pred):
    """Accuracy metric.

    Args:
        y_true (numpy.array): ground truth labels
        y_pred (numpy.array): predicted labels

    Returns:
      float: accuracy
    """

    assert(y_true.shape == y_pred.shape)
    return np.sum(y_true == y_pred) / y_true.size

  @staticmethod
  def dice_coef(y_true, y_pred):
    """ Compute the dice coefficient between two labelings.

    Dice coefficient is defined as 2 * tp / (2 * tp + fp + fn), where
    tp = true positives, fp = false positives, fn = false negatives.
    In a multiclass problem we consider all nonzero labels to be positive
    (0 is background).

    Args:
        y_true (Numpy array): ground truth segmentation, in sparse notation (one
        category label per voxel)
        y_pred (Numpy array): predicted segmentation, categorical notation
    """

    positive_mask = (y_true != 0)

    tp = np.sum(positive_mask * (y_true == y_pred))

    fp_fn = np.sum(y_true != y_pred)

    num = 2 * tp
    den = num + fp_fn + 1e-4
    return num / den

  @staticmethod
  def dice_loss(y_true, y_pred):
    """Dice loss, same as dice coefficient but decreasing."""
    return 1 - NumpyMetrics.dice_coef(y_true, y_pred)

  @staticmethod
  def mean_dice_coef(n_classes, ignore_background=True):
    """Metric that computes the averaged one-versus-all dice coefficient.

    The coefficient between two labelings is computed by averaging the dice
    coefficient of each label against the rest.

    Args:
        n_classes: number of possible distinct labels.
        ignore_background: if True, dice for the background label (0) is not
                           averaged.
    """
    def _mean_dice_coef(y_true, y_pred):
      # y_true is in sparse notation (one category number per voxel)
      # y_pred is in sparse notation
      labels = range(ignore_background, n_classes)
      n_labels = n_classes - ignore_background
      mean = 0
      for label in labels:
        # label is the positive label.
        positive_mask = (y_true == label)
        correct_mask = (y_pred == y_true)
        tp = np.sum(positive_mask * correct_mask)

        fp_fn = np.sum(positive_mask != correct_mask)

        num = 2 * tp
        den = num + fp_fn + 1e-4
        coef = num / den

        mean += coef
      mean *= 1 / n_labels
      return mean

    return _mean_dice_coef

  @staticmethod
  def mean_dice_loss(n_classes, ignore_background=True):
    """Mean dice loss, same as mean dice coefficient but decreasing."""

    def _mean_dice_loss(y_true, y_pred):
      return 1 - NumpyMetrics.mean_dice_coef(n_classes, ignore_background)(y_true, y_pred)

    return _mean_dice_loss


class FullVolumeValidationCallback(Callback):
  """Validation callback. Performs full-volume validation every n epochs.

  Validation is performed on CPU to avoid running out of memory.
  """

  def __init__(self, model, val_generator, metrics_savefile,
               train_generator=None, validate_every_n_epochs=20,
               steps=1):
    """Callback initialization.

    Args:
        model (TYPE): Description
        val_generator (TYPE): Description
        validate_every_n_epochs (int, optional): Description
    Attributes:
        generator (BatchGenerator): Validation batches generator.
        model: The model being trained.
        validate_every_n_epochs (int): as CPU validation is slow, full-volume
            validation is performed every n epochs.
    """
    self.original_model = model
    with tf.device('/cpu:0'):
      self.validation_model = keras.models.clone_model(model)
    self.labels = list(range(1, model.n_classes))
    self.generators = {'val': val_generator}
    if train_generator is not None:
      self.generators['train'] = train_generator
    self.validate_every_n_epochs = validate_every_n_epochs
    self.steps = steps
    self.history = {}
    self.metrics_savefile = metrics_savefile

  def on_epoch_end(self, epoch, logs={}):
    """Perform full-volume validation every n epochs.

    Args:
        epoch (int): Epoch number.
        logs (dict, optional): logs include `acc` and `loss`, and optionally
            include `val_loss` (if validation is enabled in `fit`), and
            `val_acc` (if validation and accuracy monitoring are enabled).
    """
    if epoch % self.validate_every_n_epochs:
      return
    # Copy weights from trained model.
    self.validation_model.set_weights(self.original_model.get_weights())
    for key in self.generators:
      metrics = []
      for i in range(self.steps):
        # Evaluate model in full volume.
        X, Y = next(self.generators[key])
        xmin, xmax, ymin, ymax, zmin, zmax = self.generators[key].get_bounding_box(X)
        X_cropped = X[:, xmin:xmax, ymin:ymax, zmin:zmax, :]
        print("Shape:::::", X_cropped.shape)
        Y_pred_cropped = self.validation_model.predict(X_cropped)
        Y = np.squeeze(Y, axis=-1)
        Y_pred = np.zeros_like(Y)
        print("shapes:", Y_pred_cropped.shape, Y_pred.shape, Y.shape)
        Y_pred[:, xmin:xmax, ymin:ymax, zmin:zmax] = np.argmax(Y_pred_cropped, axis=-1)
        metric = MetricsMonitor.MetricsMonitor.getMetricsForWholeSegmentation(Y, Y_pred,
                                                                              labels=self.labels)[0]
        print(metric)
        metrics.append(metric)
      if key not in self.history:
        self.history[key] = []
      self.history[key].append(np.mean(metrics, axis=0))

  def on_train_end(self, epoch, logs=None):
    """Append new metrics to previous ones.

    Args:
        epoch (int): epoch number
        logs (dict, optional): Dictionary of logs (unused)
    """
    try:
      previous_metrics = np.load(self.metrics_savefile + '.npz')
      metrics = {key: np.append(previous_metrics[key], new_metrics)
                for (key, new_metrics) in self.history.items()}
      # metrics = np.append(previous_metrics, self.history, axis=0)
    except FileNotFoundError:
      print('Previous metrics not found in %s, recording only new metrics.')
      metrics = {key: np.stack(history) for (key, history) in self.history.items()}
    print('Metrics history:\n', metrics)
    np.savez(self.metrics_savefile, **metrics)
