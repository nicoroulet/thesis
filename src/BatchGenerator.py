"""Collection of batch generators.

TODO: add description of each.
"""

import Tools
import Logger

import itertools
import numpy as np
import queue
import random
import scipy
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


class FetcherThread(threading.Thread):
  """Thread that fetches full images from disk and puts them in a queue.

  Images on disk should be normalized.
  TODO: make this part more robust to non-normalized images.
  """

  def __init__(self, loader_function, paths, queue, infinite=True):
    """ Initialize the Thread that fetches images.

    Args:
      loader_function: function that receives a path and returns a tuple
                       (data, segmentation), both as numpy arrays.
      paths: the list of paths to all elements of the dataset.
      queue: the queue where the data is stored.
      infinite: if True, then paths are infinitely iterated in a random way.
                if False, then paths are iterated only once, sorted lexicographically.
    """
    super(FetcherThread, self).__init__()
    self.loader_function = loader_function
    self.paths = sorted(paths)
    if infinite:
      # self.paths = itertools.cycle(paths)
      # Infinite generator of random choices of paths. `iter(int, 1)` is an infinite iterable
      self.paths_iter = (random.choice(self.paths) for _ in iter(int, 1))
    else:
      Logger.debug('Paths', self.paths)
      # Note: this behaviour is assumed by Experiment.py::validate_unet
      self.paths_iter = iter(self.paths)
    self.queue = queue
    self.daemon = True

  def run(self):
    """Thread execution routine."""
    while (True):
      try:
        self.queue.put(self.loader_function(next(self.paths_iter)))
      except StopIteration:
        break


class BatchGenerator:
  """ Asynchronous batch generator.

  Does not require the dataset to be loaded in memory.
  Uses a separate thread to load images from disk, and keeps them in a queue.
  """

  def __init__(self, patch_shape, paths, loader_function, max_queue_size=10,
               pool_size=5, pool_refresh_period=20,
               transformations=Transformations.ALL,
               patch_multiplicity=1,
               batch_size=5,
               infinite=True,
               sample_bg=True):
    """Initialize the thread that fetches objects in the queue.

    Args:
        patch_shape: shape of the patches to be generated.
        paths: see FetcherThread.__init__ args.
        loader_function: see FetcherThread.__init__ args.
        max_queue_size: the maximum size of the queue, the secondary thread
                        will block once the queue is full.
        pool_size: number of images kept simultaneously.
        transformations: defines which transformations will be applied to images
                         in order to generate images.
        patch_multiplicity (int, optional): multiplicity forced to patch dims.
    """
    self.patch_shape = patch_shape
    self.queue = queue.Queue(maxsize=max_queue_size)
    self.thread = FetcherThread(loader_function, paths, self.queue, infinite=infinite)
    self.thread.start()
    self.pool = []
    self.pool_size = pool_size
    self.pool_refresh_period = pool_refresh_period
    self.pool.append(self.queue.get())
    self.queue.task_done()
    self.cycle = itertools.cycle(range(pool_size))
    self.transformations = transformations
    self.patch_multiplicity = patch_multiplicity
    self.patch_generator = self.generate_patches()
    self.batch_size = batch_size
    self.bboxes = [Tools.get_bounding_box(self.pool[0][0], self.patch_multiplicity)]
    self.label_maps = [Tools.get_label_index(self.pool[0][1], self.bboxes[0])]
    self.ignore_bg = not sample_bg

  @property
  def paths(self):
    return self.thread.paths

  def crop(self, X, Y, bbox, label_index):
    """Crop a patch from image and segmentation volumes.

    If cropping is enabled in `self.transformations`, then a voxel of a random
    segmentation label is chosen and a box of patch_size around it is cropped
    from the data and segmentation.
    Otherwise, the original data and segmentation are returned. `self.patch_multiplicity` is not
    enforced, it should be enforced in the bounding box.

    Args:
        X (numpy array): data
        Y (numpy array): segmentation

    Returns:
        TYPE: Description
    """
    if self.transformations & Transformations.CROP:
      contained_voxel = Tools.get_voxel_of_rand_label(Y, bbox, label_index, ignore_bg=self.ignore_bg)
      x1, x2, y1, y2, z1, z2 = Tools.generate_cuboid_containing(
                                self.patch_shape, X.shape[:-1], contained_voxel)
      return (X[x1:x2, y1:y2, z1:z2, :],
              Y[x1:x2, y1:y2, z1:z2, :])
    # if self.patch_multiplicity > 1:
    #   whd = np.array(X.shape[:-1]).astype('int32')
    #   whd_cropped = (whd // self.patch_multiplicity) * self.patch_multiplicity
    #   xyz1 = (whd - whd_cropped) // 2
    #   xyz2 = xyz1 + whd_cropped
    #   x1, y1, z1 = xyz1
    #   x2, y2, z2 = xyz2
    #   # print(x1, x2, y1, y2, z1, z2)
    #   return (X[x1:x2, y1:y2, z1:z2, :],
    #           Y[x1:x2, y1:y2, z1:z2, :])
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
      X[:] = np.flip(X, axis=0)
      Y[:] = np.flip(Y, axis=0)
    return patch

  def __next__(self):
    """Generate a batch of given size.

    Batches are filled with the generator given by `generate_patches`

    Yields:
        tuple: (x, y), data and segmentation batches.
    """
    if self.patch_shape is None:
      X, Y = next(self.patch_generator)
      return (X.reshape(1, *X.shape),
             Y.reshape(1, *Y.shape))
    else:
      X, Y = next(self.patch_generator)
      batch = (np.empty((self.batch_size, *X.shape)),
               np.empty((self.batch_size, *Y.shape)))
      batch[0][0, ...] = X
      batch[1][0, ...] = Y

      for i, (X, Y) in zip(range(1, self.batch_size), self.patch_generator):

        batch[0][i, ...] = X
        batch[1][i, ...] = Y
      return batch

  def __iter__(self):
    return self

  def generate_patches(self):
    idx = next(self.cycle)
    while (True):
      for _ in range(self.pool_refresh_period):
        i = np.random.randint(len(self.pool))
        X, Y = self.pool[i]
        yield self.maybe_flip(
                self.add_gaussian_noise(
                    self.crop(X, Y, self.bboxes[i], self.label_maps[i])))
      # Replace one element from the pool
      try:
        if len(self.pool) < self.pool_size:
          self.pool.append(self.queue.get(timeout=5))
          self.bboxes.append(Tools.get_bounding_box(self.pool[-1][0], self.patch_multiplicity))
          self.label_maps.append(Tools.get_label_index(self.pool[-1][1], self.bboxes[-1]))
        else:
          self.pool[idx] = self.queue.get(timeout=5)
          self.bboxes[idx] = Tools.get_bounding_box(self.pool[idx][0], self.patch_multiplicity)
          self.label_maps[idx] = Tools.get_label_index(self.pool[idx][1], self.bboxes[idx])
          idx = next(self.cycle)
        self.queue.task_done()
      except queue.Empty:  # On timeout, the queue is assumed to be depleted.
        # TODO: gradually deplete pool. For now, this is only used in full volume with pool_size = 1
        # so this is fine.
        return


class ModalityFilter:
  """Batch Generator wrapper that drops modalities.

  This enables to feed nets that are trained on a subset of the modalities.
  """

  def __init__(self, batch_generator, all_modalities, kept_modalities):
    """Build ModalityFilter.

    Args:
        batch_generator (BatchGenerator): generator to filter.
        all_modalities (list): list of modalities yielded by `batch_generator`.
        kept_modalities (list): list of modalities that will be kept.
    """
    self.batch_generator = batch_generator
    self.kept_modalities = [i for (i, modality) in enumerate(all_modalities)
                            if modality in kept_modalities]

  def __next__(self):
    """Get a batch from the generator and filter unwanted modalities.

    Returns:
        tuple: (x, y), filtered batch.
    """
    x, y = next(self.batch_generator)
    x_filtered = x[..., self.kept_modalities]
    return x_filtered, y

  def __iter__(self):
    """Return an iterable (the wrapper itself)."""
    return self

  def __getattr__(self, name):
    """Forward every other method to the batch generator."""
    return getattr(self.batch_generator, name)


class BackgroundFilter:
  """Batch Generator wrapper that sets background labels to -1.

  This allows the use of a loss function that ignores the -1.
  """

  def __init__(self, batch_generator):
    """Build BackgroundFilter.

    Args:
        batch_generator (BatchGenerator): generator to filter.

    """
    self.batch_generator = batch_generator

  def __next__(self):
    """Get a batch from the generator and set all zeros to -1.

    Returns:
        tuple: (x, y), filtered batch.
    """
    x, y = next(self.batch_generator)
    x_filtered = x
    x_filtered[x_filtered == 0] = -1
    return x_filtered, y

  def __iter__(self):
    """Return an iterable (the wrapper itself)."""
    return self

  def __getattr__(self, name):
    """Forward every other method to the batch generator."""
    return getattr(self.batch_generator, name)


# if __name__ == '__main__':
#   import Datasets
#   dataset = Datasets.BraTS()
#   gen = dataset.get_val_generator(patch_multiplicity=1)
#   X, Y = next(gen)
#   print('First px', X[0,0,0,0,0])
#   # X -= X[0, 0, 0, 0, 0]

#   a,b,c,d,e,f = gen.get_bounding_box(X)
#   print(a,b,c,d,e,f)
#   # X = np.log(np.log(X + np.min(X) + 1) + 2)
#   # X = (0 < X) & (X < .2)

#   Xc = X[0,a:b,c:d,e:f,:]
#   Yc = Y[0,a:b,c:d,e:f,:]
#   # Xc = X[0,...]
#   # Yc = Y[0,...]
#   # Yc = Xc != 0
#   np.save('cropped', [Xc, Yc])
