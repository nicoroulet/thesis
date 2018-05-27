"""Collection of batch generators.

TODO: add description of each.
"""

import itertools
import numpy as np
import queue
import random
import threading
from sys import stdout
import keras.backend as K


class Transformations:
  """Enumerate of patch transformations, arranged as a bit set.

  Attributes:
      ALL: apply all transformations
      CROP: crop a patch of patch_shape from the original image
      FLIP: randomply perform a horizontal flip
      NOISE: add gaussian noise
      NONE: none of the above
  """

  NONE = 0
  CROP = 1
  FLIP = 1 << 1
  NOISE = 1 << 2
  ALL = (1 << 3) - 1


class BatchGenerator(object):
  """Basic batch generator. Uses a dataset that is loaded into memory.

  Attributes:
      dataset (Dataset): The dataset from which to generate batches.
      patch_multiplicity (int): multiplicity forced to patch dimensions.
          Useful for validation patches (without cropping)
      patch_shape (tuple): shape of the sampled patches
      transformations (Transformation): transformations to apply to generate
          patches.
  """

  def __init__(self,
               dataset,
               patch_shape,
               transformations=Transformations.ALL,
               patch_multiplicity=1):
    """ Initialize batch generator.

    Args:
        dataset: object that contains X_train, Y_train, X_val, Y_val
        patch_shape: patch depth, width, height
    """
    self.dataset = dataset
    self.patch_shape = patch_shape
    self.transformations = transformations
    self.patch_multiplicity = patch_multiplicity

  @staticmethod
  def get_voxel_of_rand_label(Y):
    """Random voxel from the given index, with balanced label probabilities.

    Args:
        Y (Numpy array): Image from which to pick the voxel.

    Returns:
        Numpy array: coordinates of the chosen voxel.
    """
    labels = range(int(np.max(Y)) + 1)
    while (True):
      label = np.random.choice(labels)
      try:
        return random.choice(np.argwhere(Y == label))[:-1]
      except IndexError:
        pass

  def generate_cuboid(self, volume_shape, contained_voxel):
    """Generate a cuboid to crop a patch.

    The generated cuboid has dimensions `self.patch size`, containing the given
        voxel, inside the given volume.

    Args:
        volume_shape (tuple): tuple depth, width, height. Volume that contains
            the returned cuboid.
        contained_voxel (tuple): 3D point z, x, y that will be contained in the
            returned cuboid

    Returns:
        tuple: cuboid (x1, x2, y1, y2, z1, z2) that contains `contained_voxel`
            and is fully contained by `volume_shape`
    """
    depth, width, height = volume_shape
    patch_depth, patch_width, patch_height = self.patch_shape
    vz, vx, vy = contained_voxel
    x1 = np.random.randint(max(0, vx - patch_width),
                           min(vx + 1, width - patch_width))
    y1 = np.random.randint(max(0, vy - patch_height),
                           min(vy + 1, height - patch_height))
    z1 = np.random.randint(max(0, vz - patch_depth),
                           min(vz + 1, depth - patch_depth))
    x1 = random.randrange(width - patch_width)
    x2 = x1 + patch_width
    y2 = y1 + patch_height
    z2 = z1 + patch_depth
    return x1, x2, y1, y2, z1, z2

  def crop(self, X, Y):
    """Crop a patch from image and segmentation volumes.

    If cropping is enabled in `self.transformations`, then a voxel of a random
    segmentation label is chosen and a box of patch_size around it is cropped
    from the data and segmentation.
    Otherwise, the original data and segmentation are returned, only enforcing
    the the multiplicity set in `self.patch_multiplicity`.

    Args:
        X (numpy array): data
        Y (numpy array): segmentation

    Returns:
        TYPE: Description
    """
    if self.transformations & Transformations.CROP:
      contained_voxel = self.get_voxel_of_rand_label(Y)
      x1, x2, y1, y2, z1, z2 = self.generate_cuboid(X.shape[:-1],
                                                    contained_voxel)
      return (X[z1:z2, x1:x2, y1:y2, :],
              Y[z1:z2, x1:x2, y1:y2, :])
    if self.patch_multiplicity > 1:
      dwh = np.array(X.shape[:-1]).astype('int32')
      dwh_cropped = (dwh // self.patch_multiplicity) * self.patch_multiplicity
      zxy1 = (dwh - dwh_cropped) // 2
      zxy2 = zxy1 + dwh_cropped
      z1, x1, y1 = zxy1
      z2, x2, y2 = zxy2
      return (X[z1:z2, x1:x2, y1:y2, :],
              Y[z1:z2, x1:x2, y1:y2, :])
    return X, Y

  def add_gaussian_noise(self, patch, sigma=.01):
    """Add gaussian noise to a given patch.

    This is performed only if `Transformations.NOISE` is enabled in
    `self.transformations`.

    Args:
        patch (tuple): (X, Y), data and segmentation patch to which the gaussian
            noise will be applied. X and Y are numpy arrays.
        sigma (float, optional): standard deviation of the applied noise

    Returns:
        tuple: patch with added noise.
    """
    if not (self.transformations & Transformations.NOISE):
      return patch
    X, Y = patch
    X += np.random.normal(0, sigma, X.shape)
    return X, Y

  def maybe_flip(self, patch):
    """ Apply horizontal flip on the patch with 50% chance. """
    if not (self.transformations & Transformations.FLIP):
      return patch
    if np.random.randint(1):
      X, Y = patch
      X[:] = np.flip(X, axis=3)
      Y[:] = np.flip(Y, axis=3)
    return patch

  def get_bounding_box(self, X):
    """Get the bounding box of an image.

    The bounding box is the smallest box that contains all nonzero elements of
    the volume. The multiplicity defined by the generator is enforced by
    enlarging the box if needed.

    Args:
        X (numpy array): image volume from which to calculate the box

    Returns:
        tuple: xmin, xmax, ymin, ymax, zmin, zmax, 3D bounding box
    """

    X = X[0, ...]
    print("Input shape: ", X.shape)

    # FIXME: this probably gives the [..] interval instead of the [..).
    out = []
    for ax in ((1, 2), (0, 2), (0, 1)):
      collapsed_X = np.any(X, axis=ax)
      vmin, vmax = np.where(collapsed_X)[0][[0, -1]]
      print("vmin, vmax: ", vmin, vmax)
      print("Collapsed: ", collapsed_X.shape)
      max_size = collapsed_X.shape[0]
      size = vmax - vmin
      new_size = size + (self.patch_multiplicity -
                         size % self.patch_multiplicity)
      diff = new_size - size
      # Expand the box to enforce multiplicity, without exceeding the
      # [0, max_size) interval.
      new_vmin = max(0, min(vmin - diff // 2, max_size - new_size))
      new_vmax = min(max_size, new_vmin + new_size)
      out.extend([new_vmin, new_vmax])
    print("Bounding box: ", out)
    print("Multiplicity: ", self.patch_multiplicity)
    return tuple(out)

  def generate_batches(self, batch_size=5):
    gen = self.generate_patches()
    if self.patch_shape is None:
      for X, Y in gen:
        yield (X.reshape(1, *X.shape),
               Y.reshape(1, *Y.shape))
    while True:
      X, Y = next(gen)
      batch = (np.empty((batch_size, *X.shape)),
               np.empty((batch_size, *Y.shape)))
      batch[0][0,...] = X
      batch[1][0,...] = Y

      for (X, Y), i in zip(gen, range(1, batch_size)):
        batch[0][i,...] = X
        batch[1][i,...] = Y
      yield batch

  def generate_patches(self):
    """ Generates a batch of patches from the dataset. """
    n = len(self.dataset.X_train)
    while (1):
      idx = np.random.randint(n)
      X = self.dataset.X_train[idx]
      Y = self.dataset.Y_train[idx]
      yield self.maybe_flip(
              self.add_gaussian_noise(
                self.crop(X, Y)))


class FetcherThread(threading.Thread):

  def __init__(self, loader_function, paths, queue):
    """ Initialize the Thread that fetches images.

    Args:
      loader_function: function that receives a path and returns a tuple
                       (data, segmentation), both as numpy arrays.
      paths: the list of paths to all elements of the dataset.
      queue: the queue where the data is stored.
    """
    super(FetcherThread, self).__init__()
    self.loader_function = loader_function
    self.paths = paths
    self.queue = queue
    self.daemon = True

  def run(self):
    while (True):
      self.queue.put(self.loader_function(np.random.choice(self.paths)))


class AsyncBatchGenerator(BatchGenerator):
  """ Asynchronous batch generator.
  Does not require the dataset to be loaded in memory.
  Uses a separate thread to load images from disk, and keeps them in a queue.
  """

  def __init__(self, patch_shape, paths, loader_function, max_queue_size=10,
               pool_size=5, transformations=Transformations.ALL,
               patch_multiplicity=1, n_classes=2):
    """ Initialize the thread that fetches objects in the queue.

    Args:
      patch_shape: shape of the patches to be generated.
      loader_function: see FetcherThread.__init__ args.
      paths: see FetcherThread.__init__ args.
      max_queue_size: the maximum size of the queue, the secondary thread
                      will block once the queue is full. Should be at least the
                      batch size.
      pool_size: number of images kept simultaneously.
      is_validation: if True, images will be yielded with no cropping or
                     augmentation
      transformations: defines which transformations will be applied to images
                       in order to generate images.
    """
    self.patch_shape = patch_shape
    self.queue = queue.Queue(maxsize=max_queue_size)
    self.thread = FetcherThread(loader_function, paths, self.queue)
    self.thread.start()
    self.pool = []
    self.pool_size = pool_size
    # for _ in range(pool_size):
    self.pool.append(self.queue.get())
    self.queue.task_done()
    # print('pool', len(self.pool))
    self.cycle = itertools.cycle(range(pool_size))
    self.transformations = transformations
    self.patch_multiplicity = patch_multiplicity
    self.n_classes = n_classes

  def generate_patches(self):
    while (True):
      for _ in range(20):  # TODO: parameterize
        X, Y = random.choice(self.pool)
        yield self.maybe_flip(
                self.add_gaussian_noise(
                  self.crop(X, Y)))
      # Replace one element from the pool
      idx = next(self.cycle)
      self.pool[idx] = self.queue.get()
      self.queue.task_done()
      if len(self.pool) < self.pool_size:
        self.pool.append(self.queue.get())
        self.queue.task_done()


class GeneratorThread(threading.Thread):

  def __init__(self, batch_generator, queue):
    """ Eagerly generates batches and stores them in a queue.

    Args:
      batch_generator: generator used to produce the batches.
    """
    super(GeneratorThread, self).__init__()
    self.batch_generator = batch_generator
    self.queue = queue
    self.daemon = True

  def run(self):
    while (True):
      self.queue.put(next(self.batch_generator))

class FetcherGenerator:

  def __init__(self, batch_size, *args, **kwargs):
    generator = AsyncBatchGenerator(*args, **kwargs).generate_batches(batch_size)
    self.queue = queue.Queue(maxsize=10)
    self.thread = GeneratorThread(generator, self.queue)
    self.thread.start()

  def generate_batches(self, *args, **kwargs):
    while (True):
      batch = self.queue.get()
      self.queue.task_done()
      yield batch


if __name__ == '__main__':
  import Datasets
  ibsr = Datasets.TrainableDataset(Datasets.IBSR())
  bg = BatchGenerator(ibsr, 100, 100, 2)
  patch = next(bg.generate_batch(2))
  import matplotlib.pyplot as plt
  print(patch[0].shape)
  plt.imshow(patch[0][0] + patch[1][0])
  plt.show()
