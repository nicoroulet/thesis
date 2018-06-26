"""Collection of batch generators.

TODO: add description of each.
"""

import Tools

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
    """Thread execution routine."""
    while (True):
      self.queue.put(self.loader_function(np.random.choice(self.paths)))


class BatchGenerator:
  """ Asynchronous batch generator.

  Does not require the dataset to be loaded in memory.
  Uses a separate thread to load images from disk, and keeps them in a queue.
  """

  def __init__(self, patch_shape, paths, loader_function, max_queue_size=10,
               pool_size=5, pool_refresh_period=20,
               transformations=Transformations.ALL,
               patch_multiplicity=1,
               batch_size=5):
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
    self.thread = FetcherThread(loader_function, paths, self.queue)
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
      bbox = self.get_bounding_box(X)
      contained_voxel = Tools.Image.get_voxel_of_rand_label(Y, bbox)
      x1, x2, y1, y2, z1, z2 = Tools.Image.generate_cuboid_containing(
                                self.patch_shape, X.shape[:-1], contained_voxel)
      return (X[x1:x2, y1:y2, z1:z2, :],
              Y[x1:x2, y1:y2, z1:z2, :])
    if self.patch_multiplicity > 1:
      whd = np.array(X.shape[:-1]).astype('int32')
      whd_cropped = (whd // self.patch_multiplicity) * self.patch_multiplicity
      xyz1 = (whd - whd_cropped) // 2
      xyz2 = xyz1 + whd_cropped
      x1, y1, z1 = xyz1
      x2, y2, z2 = xyz2
      return (X[x1:x2, y1:y2, z1:z2, :],
              Y[x1:x2, y1:y2, z1:z2, :])
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

  def get_bounding_box(self, X):
    """Get the bounding box of an image.

    The bounding box is the smallest box that contains all nonzero elements of
    the volume. The multiplicity defined by the generator is enforced by
    enlarging the box if needed.

    Args:
        X (numpy array): image volume from which to calculate the box

    Returns:
        tuple: xmin, xmax, ymin, ymax, zmin, zmax; 3D bounding box
    """

    try:
      X = np.squeeze(X, axis=0)
    except ValueError:
      pass  # axis 0 is not single-dimensional
    # Clear possible interpolation artifacts around actual brain.
    X = X * np.abs(X) > 0.0001
    out = []
    for ax in ((1, 2), (0, 2), (0, 1)):
      collapsed_X = np.any(X, axis=ax)

      vmin, vmax = np.where(collapsed_X)[0][[0, -1]]
      max_size = collapsed_X.shape[0]
      size = vmax - vmin
      # FIXME: if size % patch_multiplicity == 0, this adds innecesary size.
      new_size = size + (self.patch_multiplicity -
                         size % self.patch_multiplicity)
      diff = new_size - size
      # Expand the box to enforce multiplicity, without exceeding the [0, max_size) interval.
      new_vmin = max(0, min(vmin - diff // 2, max_size - new_size))
      new_vmax = min(max_size, new_vmin + new_size)
      out.extend([new_vmin, new_vmax])
    return tuple(out)

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
    while (True):
      for i in range(self.pool_refresh_period):
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


class ModalityFilter:
  """Batch Generator wrapper thar drops modalities.

  This enables to feed nets that are trained on a subset of the modalities.
  """

  def __init__(self, batch_generator, kept_modalities):
    """Build ModalityFilter.

    Args:
        batch_generator (BatchGenerator): generator to filter
        kept_modalities (list): indexes of modalities that will be kept
    """
    self.batch_generator = batch_generator
    self.kept_modalities = kept_modalities

  def __next__(self):
    """Get a batch from the generator and filter unwanted modalities.

    Returns:
        tuple: (x, y), filtered batch.
    """
    x, y = next(self.batch_generator)
    x_filtered = x[..., self.kept_modalities]
    return x_filtered, y

  def __iter__(self):
    return self

  def __getattr__(self, name):
    return self.batch_generator.__dict__[name]


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
