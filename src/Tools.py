"""Collection of Tools."""

import numpy as np
import random


class Image:
  """Image cropping and processing tools."""

  @staticmethod
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
      Y = Y[x1:x2, y1:y2, z1:z2]
      try:
        voxel = random.choice(np.argwhere(Y == label))[:-1]
        return voxel + [x1, y1, z1]
      except IndexError:
        pass

  @staticmethod
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

  @staticmethod
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
