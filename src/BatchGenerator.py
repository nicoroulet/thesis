import itertools
import numpy as np
import queue
import random
import threading
from sys import stdout
class BatchGenerator(object):

  def __init__(self,
               trainable_dataset,
               patch_shape):

    self.dataset = trainable_dataset
    self.patch_depth , self.patch_width, self.patch_height = patch_shape

  @staticmethod
  def get_voxel_of_rand_label(Y):
    labels = list(set(Y.flat))
    label = np.random.choice(labels)
    return random.choice(np.argwhere(Y == label))[:-1]
    # index = possible_indexes[np.random.randint(possible_indexes.shape[0])]
    # return index

  def generate_cuboid(self, volume_shape, contained_voxel):
    """ Generates a cuboid of patch size containing the given voxel, inside
        the given volume. """
    depth, width, height = volume_shape
    vz, vx, vy = contained_voxel
    x1 = np.random.randint(max(0, vx - self.patch_width),
                           min(vx + 1, width - self.patch_width))
    y1 = np.random.randint(max(0, vy - self.patch_height),
                           min(vy + 1, height - self.patch_height))
    z1 = np.random.randint(max(0, vz - self.patch_depth),
                           min(vz + 1, depth - self.patch_depth))
    x1 = random.randrange(width - self.patch_width)
    x2 = x1 + self.patch_width
    y2 = y1 + self.patch_height
    z2 = z1 + self.patch_depth
    return x1, x2, y1, y2, z1, z2

  @staticmethod
  def crop(X, Y, cuboid):
    x1, x2, y1, y2, z1, z2 = cuboid
    return (X[z1:z2, x1:x2, y1:y2, :],
            Y[z1:z2, x1:x2, y1:y2, :])

  @staticmethod
  def add_gaussian_noise(patch, sigma=.01):
    X, Y = patch
    return X + np.random.normal(0, sigma, X.shape), Y

  @staticmethod
  def maybe_flip(patch):
    """ Apply horizontal flip on the patch with 50% chance. """
    X, Y = patch
    if np.random.choice([0,1]) == 1:
      return (np.flip(X, axis=3),
              np.flip(Y, axis=3))
    return patch

  def generate_batches(self, batch_size=5):
    gen = self.generate_patches()
    while 1:
      batch = (np.zeros((batch_size, self.patch_depth, self.patch_width,
                        self.patch_height, 1)),
               np.zeros((batch_size, self.patch_depth, self.patch_width,
                        self.patch_height, 1)))

      for (X, Y), i in zip(gen, range(batch_size)):
        batch[0][i,...] = X
        batch[1][i,...] = Y
      yield batch

  def generate_patches(self):
    """ Generates a batch of patches from the dataset. """

    while (1):
      idx = np.random.randint(len(self.dataset.X_train))
      X = self.dataset.X_train[idx]
      Y = self.dataset.Y_train[idx]
      contained_voxel = get_voxel_of_rand_label(Y)
      crop_region = self.generate_cuboid(X.shape[:-1], contained_voxel)
      yield self.maybe_flip(
              self.add_gaussian_noise(
                self.crop(X, Y, crop_region)))


class AsyncBatchGenerator(BatchGenerator):
  """ Asynchronous batch generator.
  Does not require the dataset to be loaded in memory.
  Uses a separate thread to load images from disk, and keeps them in a queue.
  """

  class FetcherThread(threading.Thread):

    def __init__(self, loader_function, paths, queue):
      """ Initialize the Thread that fetches images.

      Args:
        loader_function: function that receives a path and returns a tuple
                         (data, segmentation), both as numpy arrays.
        paths: the list of paths to all elements of the dataset.
        queue: the queue where the data is stored.
      """
      super(AsyncBatchGenerator.FetcherThread,self).__init__()
      self.loader_function = loader_function
      self.paths = paths
      self.queue = queue
      self.daemon = True

    def run(self):
      while (True):
        self.queue.put(self.loader_function(np.random.choice(self.paths)))


  def __init__(self, patch_shape, paths, loader_function, max_queue_size=10,
               pool_size=10):
    """ Initialize the thread that fetches objects in the queue.

    Args:
      patch_shape: shape of the patches to be generated.
      loader_function: see FetcherThread.__init__ args.
      paths: see FetcherThread.__init__ args.
      max_queue_size: the maximum size of the queue, the secondary thread
                      will block once the queue is full. Should be at least the
                      batch size.
      pool_size: number of images kept simultaneously.
    """
    self.patch_depth, self.patch_width, self.patch_height = patch_shape
    self.queue = queue.Queue(maxsize=max_queue_size)
    self.thread = AsyncBatchGenerator.FetcherThread(loader_function, paths,
                                                    self.queue)
    self.thread.start()
    self.pool = []
    for _ in range(pool_size):
      self.pool.append(self.queue.get())
      self.queue.task_done()
    # print('pool', len(self.pool))
    self.cycle = itertools.cycle(range(pool_size))

  def generate_patches(self):
    while (True):
      for _ in range(10):  # TODO: parameterize
        X, Y = random.choice(self.pool)
        contained_voxel = self.get_voxel_of_rand_label(Y)
        crop_region = self.generate_cuboid(X.shape[:-1], contained_voxel)
        yield self.maybe_flip(
                self.add_gaussian_noise(
                  self.crop(X, Y, crop_region)))
      # Replace one element from the pool
      idx = next(self.cycle)
      self.pool[idx] = self.queue.get()
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