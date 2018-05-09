import itertools
import numpy as np
import queue
import random
import threading
from sys import stdout
import keras.backend as K

class Transformations:
  NONE  = 0
  CROP  = 1
  FLIP  = 1 << 1
  NOISE = 1 << 2
  ALL   = (1 << 3) - 1

class BatchGenerator(object):

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
  def get_labels(Y):
    return [y for y in range(np.max(Y)) if np.any(Y == y)]

  @staticmethod
  def get_voxel_of_rand_label(Y):
    # labels = list(set(Y.flat))
    labels = range(int(np.max(Y)))
    while (True):
      label = np.random.choice(labels)
      if np.any(Y == label):
        return random.choice(np.argwhere(Y == label))[:-1]
    # index = possible_indexes[np.random.randint(possible_indexes.shape[0])]
    # return index

  def generate_cuboid(self, volume_shape, contained_voxel):
    """ Generates a cuboid of patch size containing the given voxel, inside
        the given volume. """
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
    if not (self.transformations & Transformations.CROP):
      if self.patch_multiplicity > 1:
        dwh = np.array(X.shape[:-1]).astype('int32')
        dwh_cropped = (dwh // self.patch_multiplicity) * self.patch_multiplicity
        zxy1 = (dwh - dwh_cropped) // 2
        zxy2 = zxy1 + dwh_cropped
        z1,x1,y1 = zxy1
        z2,x2,y2 = zxy2
        return (X[z1:z2, x1:x2, y1:y2, :],
                Y[z1:z2, x1:x2, y1:y2, :])
      return X, Y
    contained_voxel = self.get_voxel_of_rand_label(Y)
    x1, x2, y1, y2, z1, z2 = self.generate_cuboid(X.shape[:-1], contained_voxel)
    return (X[z1:z2, x1:x2, y1:y2, :],
            Y[z1:z2, x1:x2, y1:y2, :])

  def add_gaussian_noise(self, patch, sigma=.01):
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

  def generate_batches(self, batch_size=5):
    gen = self.generate_patches()
    if self.patch_shape is None:
      for X, Y in gen:
        yield (X.reshape(1, *X.shape),
               Y.reshape(1, *Y.shape))
    while True:
      batch = (np.zeros((batch_size, *self.patch_shape, 1)),
               np.zeros((batch_size, *self.patch_shape, 1)))

      for (X, Y), i in zip(gen, range(batch_size)):
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
               pool_size=10, transformations=Transformations.ALL,
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
      for _ in range(10):  # TODO: parameterize
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



if __name__ == '__main__':
  import Datasets
  ibsr = Datasets.TrainableDataset(Datasets.IBSR())
  bg = BatchGenerator(ibsr, 100, 100, 2)
  patch = next(bg.generate_batch(2))
  import matplotlib.pyplot as plt
  print(patch[0].shape)
  plt.imshow(patch[0][0] + patch[1][0])
  plt.show()
