import keras
import keras.backend as K

import tensorflow as tf

# from medpy.metrics.binary import

def sparse_dice_coef(y_true, y_pred):
  """ Computes the dice coefficient between two labelings.
      y_true is in sparse notation (one category number per voxel), while
      y_pred is in categorical notation (a probability distribution for each
                                         voxel)

      This comment might be obsolete:
      Dice coefficient is defined as 2 * tp / (2 * tp + fp + fn), where
      tp = true positives, fp = false positives, fn = false negatives.
      In a multiclass problem we consider all nonzero labels to be positive
      (0 is background). This distinction is useful for computing tp, for
      fp + fn, we just compute all the incorrect labelings.
  """
  # y_true_f = K.flatten(y_true)
  n_classes = K.shape(y_pred)[-1]
  y_true_int = K.cast(K.sum(y_true, axis=-1), 'int32')

  positive_mask = K.clip(y_true, 0, 1)
  correct_mask = K.one_hot(y_true_int, n_classes)
  # y_pred_for_correct = K.gather(y_pred, y_true_int)
  # Scores given to the correct labels
  tp = K.sum(positive_mask * K.sum(correct_mask * y_pred))

  bg_scores = y_pred[...,0:1]
  # Positive scores given to the background labels
  fp = K.sum((1 - positive_mask) * (1 - bg_scores))

  # Background scores given to the positive labels.
  # K.gather(y_pred, K.zeros_like(y_true_int))
  fn = K.sum(bg_scores * positive_mask)

  num = 2 * tp
  den = num + fn + fp + 1e-4
  # assert(den > 0)
  # if den == 0:
  #   return 0
  return num / den

  # y_pred_f = K.cast(K.flatten(K.gather(y_pred, axis=-1)), dtype='float32')
  # # y_pred_f =


  # y_true_f = K.flatten(y_true)
  # y_pred_f = K.cast(K.flatten(K.argmax(y_pred, axis=-1)), dtype='float32')
  # corrects = K.cast(K.equal(y_true_f, y_pred_f), dtype='float32')
  # # 2 * tp
  # tp_dbl = 2 * K.sum(corrects * K.clip(y_true_f, 0, 1))
  # # fp + fn
  # fp_fn = K.sum(-corrects + 1)
  # return tp_dbl / (tp_dbl + fp_fn)

# def sparse_sum_diff(y_true, y_pred):
#   # return 1 - sparse_dice_coef(y_true, y_pred)
#   return K.sum(y_true == K.argmax(y_pred)

def sparse_dice_loss(y_true, y_pred):
  return 1 - sparse_dice_coef(y_true, y_pred)

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