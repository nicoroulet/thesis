import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def write_gif(img3d, filename, fps=24):
  """ Write the image to the given paths as a GIF, one frame per layer. """
  import array2gif
  to_rgb = [np.array([layer, layer, layer]).astype(int) // 100
            for layer in img3d]
  array2gif.write_gif(to_rgb, filename, fps=fps)


class Timeline:

  def __init__(self):
    self.train_acc = []
    self.val_acc = []

  def update(self, history):
    self.train_acc += history.history['acc']
    if 'val_acc' in history.history:
      self.val_acc += history.history['val_acc']

  def plot(self, show=True, label_prefix=''):
    plt.plot(self.train_acc, label='%s: train_acc' % label_prefix)
    if self.val_acc:
      plt.plot(self.val_acc, label='%s: val_acc' % label_prefix)
    if show:
      plt.show()


class TimelinePool:

  def __init__(self):
    self.timelines = {}

  def new_timeline(self, label):
    self.timelines[label] = Timeline()
    return self.timelines[label]

  def plot(self, labels=[]):
    if not labels:
      labels = self.timelines.keys()
    for label, timeline in self.timelines.items():
      if label in labels:
        timeline.plot(show=False, label_prefix=label)
    plt.legend()
    plt.show()

  def rename(self, label, new_label):
    if label in self.timelines:
      self.timelines[new_label] = self.timelines.pop(label)


def visualize_predictions(x, y_true, y_pred):
  print(y_true.shape, y_pred.shape)
  assert(y_true.shape == y_pred.shape)

  ax = plt.subplot(121)
  ax.hist(y_true.flat)
  ax.set_yscale('log')
  ax.set_title('Ground truth hist')
  ax = plt.subplot(122)
  ax.hist(y_pred.flat)
  ax.set_yscale('log')
  ax.set_title('Ground truth hist')
  plt.show()

  x /= np.max(x)
  y_true = y_true / (np.max(y_true) + 1e-04)
  y_pred = y_pred / (np.max(y_pred) + 1e-04)

  img = np.concatenate([x, y_true, y_pred], axis=2)
  vmin = np.min(img)
  vmax = np.max(img)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  def update(i):
    ax.clear()
    ax.imshow(img[i], vmin=vmin, vmax=vmax)

  A = anim.FuncAnimation(fig, update, frames=img.shape[0], repeat=True, interval=100)
  plt.show()
