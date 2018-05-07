import keras
import keras.backend as K

import tensorflow as tf

# from medpy.metrics.binary import

################################################################################
############################## CONTINUOUS METRICS ##############################
################################################################################

# Continuous metrics operate on the class scores.

class continuous:

  @staticmethod
  def sparse_dice_coef(y_true, y_pred):
    """ Computes the dice coefficient between two labelings.

    Args:
        y_true: ground truth segmentation, in sparse notation (one category
                label per voxel)
        y_pred: predicted segmentation in categorical notation (a per-class
                score for each voxel)

      Dice coefficient is defined as 2 * tp / (2 * tp + fp + fn), where
      tp = true positives, fp = false positives, fn = false negatives.
      In a multiclass problem we consider all nonzero labels to be positive
      (0 is background).
    """
    # shape notation: b = batch index, d = depth, w = width, h = height
    # y_true shape: (b, d, w, h, 1)
    # y_pred shape: (b, d, w, h, n_classes)
    n_classes = K.shape(y_pred)[-1]
    # y_true_int shape: (b, d, w, h)
    y_true_int = K.cast(K.squeeze(y_true, axis=-1), 'int32')

    # positive_mask shape: (b, d, w, h, 1)
    positive_mask = K.clip(y_true, 0, 1)
    # correct_mask shape: (b, d, w, h, n_classes)
    correct_mask = K.one_hot(y_true_int, n_classes)
    # correct_scores shape: (b, d, w, h, n_classes)
    correct_scores = correct_mask * y_pred
    # Scores given to the correct labels
    # correct_scores_sum shape: (b, d, w, h, 1)
    correct_scores_sum = K.sum(correct_scores, axis=-1, keepdims=True)
    # True Positive: sum of correct scores assigned to positive label
    tp = K.sum(positive_mask * correct_scores_sum)

    bg_scores = y_pred[...,0:1]
    # Positive scores given to the background labels
    fp = K.sum((1 - positive_mask) * (1 - bg_scores))

    # Background scores given to the positive labels.
    fn = K.sum(bg_scores * positive_mask)

    num = 2 * tp
    den = num + fn + fp + 1e-4
    return num / den

  @staticmethod
  def sparse_dice_loss(y_true, y_pred):
    return 1 - sparse_dice_coef(y_true, y_pred)


  @staticmethod
  def mean_dice_coef(ignore_background=True):
    """ Returns a metric that computes the dice coefficient between two
    labelings by averaging the dice coefficient of each label against the rest.
    Args:
        ignore_background: if True, dice for the background label (0) is not
                           averaged.
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

        # label_mask = K.ones(K.shape(y_true), dtype='float', name='ones') * label
        positive_mask = K.equal(y_true, label)
        positive_mask = K.cast(positive_mask, 'float')
        negative_mask = 1 - positive_mask
        # True Positive: sum of correct scores assigned to positive label
        tp = K.sum(y_pred[...,label:label+1] * positive_mask)# / (K.sum(positive_mask) + 1e-4)
        # False Positive: sum of positive scores assigned to negative labels
        fp = K.sum(y_pred[...,label:label+1] * (negative_mask))# / (K.sum(negative_mask) + 1e-4)
        # False Negative: sum of negative scores assigned to positive label
        # This assumes that the sum of scores is 1 (output from softmax)
        fn = K.sum((1 - y_pred[...,label:label+1]) * positive_mask)# / (K.sum(positive_mask) + 1e-4)
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

    def _mean_dice_loss(y_true, y_pred):
      return 1 - mean_dice_coef(ignore_background)(y_true, y_pred)

    return _mean_dice_loss

################################################################################
############################### DISCRETE METRICS ###############################
################################################################################

# Discrete metrics operate on the predicted classes.

class discrete:

  @staticmethod
  def sparse_dice_coef(y_true, y_pred):
    """ Computes the dice coefficient between two labelings.

    Args:
        y_true: ground truth segmentation, in sparse notation (one category
                label per voxel)
        y_pred: predicted segmentation, in sparse notation

        Dice coefficient is defined as 2 * tp / (2 * tp + fp + fn), where
        tp = true positives, fp = false positives, fn = false negatives.
        In a multiclass problem we consider all nonzero labels to be positive
        (0 is background).
    """

    positive_mask = K.clip(y_true, 0, 1)

    tp = positive_mask * K.cast(K.equal(y_true, y_pred), 'float')# / K.sum(positive_mask)

    fp_fn = K.cast(K.not_equal(y_true, y_pred), 'float')

    num = 2 * tp
    den = num + fp_fn + 1e-4
    return num / den

  @staticmethod
  def sparse_dice_loss(y_true, y_pred):
    return 1 - sparse_dice_coef(y_true, y_pred)



  @staticmethod
  def mean_dice_coef(n_classes, ignore_background=True):
    """ Returns a metric that computes the dice coefficient between two
    labelings by averaging the dice coefficient of each label against the rest.

    Args:
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


        # label_mask = K.ones(K.shape(y_true), dtype='float', name='ones') * label
        positive_mask = K.cast(K.equal(y_true, label), 'float')
        correct_mask = K.cast(K.equal(y_pred, label), 'float')
        tp = K.sum(positive_mask * correct_mask)

        fp_fn = K.sum(K.cast(K.not_equal(positive_mask, correct_mask), 'float'))

        num = 2 * tp
        den = num + fp_fn + 1e-4
        coef = num / den

        mean += coef
      mean *= 1 / n_labels
      return mean

    return _mean_dice_coef

  @staticmethod
  def mean_dice_loss(ignore_background=True):

    def _mean_dice_loss(y_true, y_pred):
      return 1 - mean_dice_coef(ignore_background)(y_true, y_pred)

    return _mean_dice_loss

  @staticmethod
  def to_continuous(loss):
    def loss_wrapper(y_true, y_pred):
      # y_true is in sparse notation
      # y_pred is in categorical notation
      y_pred_sparse = K.cast(K.expand_dims(K.argmax(y_pred, axis=-1), axis=-1), 'float')
      return loss(y_true, y_pred_sparse)

    return loss_wrapper



























# def sparse_sum_diff(y_true, y_pred):
#   # return 1 - sparse_dice_coef(y_true, y_pred)
#   return K.sum(y_true == K.argmax(y_pred)

# def dice_coef(y_true, y_pred):
#   smooth=1
#   y_true_f = keras.backend.flatten(y_true)
#   y_pred_f = keras.backend.flatten(y_pred)
#   intersection = keras.backend.sum(y_true_f * y_pred_f)
#   return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#   return 1 - dice_coef(y_true, y_pred)

# def sparse_mean_dice_loss(y_true, y_pred):


def dice_loss(y_true,
              y_pred,
              num_classes=2,
              smooth=1e-5,
              include_background=True,
              only_present=False):
    """Calculates a smooth Dice coefficient loss from sparse labels.
    Args:
        logits (tf.Tensor): logits prediction for which to calculate
            crossentropy error
        labels (tf.Tensor): sparse labels used for crossentropy error
            calculation
        num_classes (int): number of class labels to evaluate on
        smooth (float): smoothing coefficient for the loss computation
        include_background (bool): flag to include a loss on the background
            label or not
        only_present (bool): flag to include only labels present in the
            inputs or not
    Returns:
        tf.Tensor: Tensor scalar representing the loss
    """

    # Get a softmax probability of the logits predictions and a one hot
    # encoding of the labels tensor
    probs = y_pred
    onehot_labels = tf.one_hot(
        indices=y_true,
        depth=num_classes,
        dtype=tf.float32,
        name='onehot_labels')

    # Compute the Dice similarity coefficient
    label_sum = tf.reduce_sum(onehot_labels, axis=[1, 2, 3], name='label_sum')
    pred_sum = tf.reduce_sum(probs, axis=[1, 2, 3], name='pred_sum')
    intersection = tf.reduce_sum(onehot_labels * probs, axis=[1, 2, 3],
                                 name='intersection')

    per_sample_per_class_dice = (2. * intersection + smooth)
    per_sample_per_class_dice /= (label_sum + pred_sum + smooth)

    # Include or exclude the background label for the computation
    if include_background:
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice, (-1, ))
        flat_label = tf.reshape(label_sum, (-1, ))
    else:
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice[:, 1:], (-1, ))
        flat_label = tf.reshape(label_sum[:, 1:], (-1, ))

    # Include or exclude non-present labels for the computation
    if only_present:
        masked_dice = tf.boolean_mask(flat_per_sample_per_class_dice,
                                      tf.logical_not(tf.equal(flat_label, 0)))
    else:
        masked_dice = tf.boolean_mask(
            flat_per_sample_per_class_dice,
            tf.logical_not(tf.is_nan(flat_per_sample_per_class_dice)))

    dice = tf.reduce_mean(masked_dice)
    loss = 1. - dice

    return loss