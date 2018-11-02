"""Collection of Tools."""

import numpy as np
import random
import os

def get_label_index(Y, bbox):
  x1, x2, y1, y2, z1, z2 = bbox
  Y_cropped = Y[x1:x2, y1:y2, z1:z2]
  labels = range(int(np.max(Y_cropped)) + 1)
  label_index = {}
  for label in labels:
    label_index[label] = np.argwhere(Y_cropped == label)
  return label_index


def get_voxel_of_rand_label(Y, bbox, label_index, ignore_bg=False):
  """Random voxel from the given index, with balanced label probabilities.

  Args:
      Y (Numpy array): Image from which to pick the voxel.
      bbox (tuple): bounding box x1, x2, y1, y2, z1, z2 from which to
          sample the voxel.

  Returns:
      Numpy array: coordinates of the chosen voxel.

  """
  labels = range(ignore_bg, int(np.max(Y)) + 1)
  x1, x2, y1, y2, z1, z2 = bbox
  Y_cropped = Y[x1:x2, y1:y2, z1:z2]
  while (True):
    label = np.random.choice(labels)
    try:
      voxel = random.choice(label_index[label])[:-1]
      return voxel + np.array([x1, y1, z1])
    except IndexError:
      pass


def get_bounding_box(X, patch_multiplicity):
  """Get the bounding box of an image.

  The bounding box is the smallest box that contains all nonzero elements of
  the volume. The multiplicity defined by the generator is enforced by
  enlarging the box if needed.

  Args:
    X (numpy array): image volume from which to calculate the box
    patch_multiplicity (int): multiplicity enforced to dimensions of bounding box.

  Returns:
    tuple: xmin, xmax, ymin, ymax, zmin, zmax; 3D bounding box
  """
  try:
    X = np.squeeze(X, axis=0)
  except ValueError:
    pass  # axis 0 is not single-dimensional
  # Clear possible interpolation artifacts around actual brain.
  mask = X != bg_value
  # X = X * np.abs(X) > 0.0001
  out = []
  for ax in ((1, 2), (0, 2), (0, 1)):
    collapsed_mask = np.any(mask, axis=ax)

    vmin, vmax = np.where(collapsed_mask)[0][[0, -1]]
    max_size = collapsed_mask.shape[0]
    size = vmax - vmin
    # FIXME: if size % patch_multiplicity == 0, this adds innecesary size.
    new_size = size + (patch_multiplicity - size % patch_multiplicity)
    diff = new_size - size
    # Expand the box to enforce multiplicity, without exceeding the [0, max_size) interval.
    new_vmin = max(0, min(vmin - diff // 2, max_size - new_size))
    new_vmax = min(max_size, new_vmin + new_size)
    out.extend([new_vmin, new_vmax])
  return tuple(out)


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

bg_value = -4

_model_subdir = ''

def set_model_subdir(subdir):
  global _model_subdir
  _model_subdir = subdir

def get_dataset_savedir(dataset, loss=None):
  """Figure out savedir from a given dataset and loss function.

  Args:
      dataset (Dataset): the Dataset.
      loss (string or function, optional): Dataset loss. Default is
          `sparse_categorical_crossentropy`.

  """
  savedir = '../models/%s/unet_%s' % (_model_subdir, dataset.name)
  if loss is not None and loss != 'sparse_categorical_crossentropy':
    savedir += '_' + (loss if isinstance(loss, str) else loss.__name__)
  return savedir

def ensure_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
