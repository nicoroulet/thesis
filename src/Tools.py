"""Collection of Tools."""

import numpy as np
import random


def get_voxel_of_rand_label(Y, sub_volume):
  """Random voxel from the given index, with balanced label probabilities.

  Args:
      Y (Numpy array): Image from which to pick the voxel.
      sub_volume (tuple): box x1, x2, y1, y2, z1, z2 from which to
          sample the voxel.

  Returns:
      Numpy array: coordinates of the chosen voxel.

  """
  labels = range(int(np.max(Y)) + 1)
  while (True):
    label = np.random.choice(labels)
    x1, x2, y1, y2, z1, z2 = sub_volume
    Y_cropped = Y[x1:x2, y1:y2, z1:z2]
    try:
      voxel = random.choice(np.argwhere(Y_cropped == label))[:-1]
      return voxel + [x1, y1, z1]
    except IndexError:
      pass


def generate_cuboid_centered(cuboid_shape, volume_shape, center_voxel):
  """Generate a cuboid to crop a patch, centered on a given voxel.

  Args:
      cuboid_shape (iterable): shape of returned cuboid.
      volume_shape (iterable): tuple width, height, depth. Volume that
          contains the returned cuboid.
      center_voxel (iterable): 3D point x, y, z that will be centered in
          the returned cuboid.

  Returns:
      tuple: cuboid (x1, x2, y1, y2, z1, z2) that contains `center_voxel`
          and is fully contained by `volume_shape`. The generated cuboid is,
          as much as possible, centered on `center_voxel`.

  """
  x1, y1, z1 = v = np.minimum(np.maximum(0, np.array(center_voxel) -
                                 np.array(cuboid_shape, dtype='int') // 2),
                          np.array(volume_shape) - cuboid_shape)
  x2, y2, z2 = v + cuboid_shape
  return x1, x2, y1, y2, z1, z2


def generate_cuboid_containing(cuboid_shape, volume_shape, contained_voxel):
  """Generate a cuboid to crop a patch, containing a given voxel.

  Args:
      cuboid_shape (iterable): shape of returned cuboid.
      volume_shape (iterable): tuple width, height, depth. Volume that
          contains the returned cuboid.
      contained_voxel (iterable): 3D point x, y, z that will be contained in
          the returned cuboid.

  Returns:
      tuple: cuboid (x1, x2, y1, y2, z1, z2) that contains `contained_voxel`
          and is fully contained by `volume_shape`.

  """
  cuboid_width, cuboid_height, cuboid_depth = cuboid_shape
  width, height, depth = volume_shape
  vx, vy, vz = contained_voxel
  x1 = np.random.randint(max(0, vx - cuboid_width),
                         min(vx + 1, width - cuboid_width))
  y1 = np.random.randint(max(0, vy - cuboid_height),
                         min(vy + 1, height - cuboid_height))
  z1 = np.random.randint(max(0, vz - cuboid_depth),
                         min(vz + 1, depth - cuboid_depth))
  x1 = random.randrange(width - cuboid_width)
  x2 = x1 + cuboid_width
  y2 = y1 + cuboid_height
  z2 = z1 + cuboid_depth
  return x1, x2, y1, y2, z1, z2


def filter_modalities(all_modalities, target_modalities, x):
  """Filter channels from x based on the given modalities.

  Modalities are represented on the last dimension of `x` and are the different types of images
  (t1, t2, flair, etc.). This is used to feed a dataset with extra modalities to a net that has
  been trained on a subset of them.

  Args:
      all_modalities (list): modalities of x.
      target_modalities (list): desired modalities.
      x (numpy array): image or batch of images to filter.

  Returns:
      numpy array: filtered x

  """
  # TODO: this is inefficient. Furthermore, it may be innecessarily recomputed on repeated calls.
  target_indexes = [i for (i, modality) in enumerate(all_modalities)
                    if modality in target_modalities]

  return x[..., target_indexes]


def get_dataset_savedir(dataset, loss=None):
  """Figure out savedir from a given dataset and loss function.

  Args:
      dataset (Dataset): the Dataset
      loss (string or function, optional): Dataset loss. Default is
          `sparse_categorical_crossentropy`

  """
  savedir = 'checkpoints/unet_%s' % (dataset.name)
  if loss is not None and loss != 'sparse_categorical_crossentropy':
    savedir += '_' + (loss if isinstance(loss, str) else loss.__name__)
  return savedir
