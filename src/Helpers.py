import array2gif
import numpy as np
import matplotlib.pyplot as plt

def write_gif(img3d, filename, fps=24):
  """ Writes the image to the given paths as a GIF, one frame per layer. """
  to_rgb = [np.array([layer, layer, layer]).astype(int) // 100 for layer in img3d]
  array2gif.write_gif(to_rgb, filename, fps=fps)

class Timeline(object):

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

class TimelinePool(object):

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


def plot_example(model, X, Y):
  """ Plot a patch, its segmentation and the prediction given by the model.
  The middle layer of the model is displayed. """

  def crop_channel(a):
      return a.reshape(a.shape[:-1])

  def decategorize(a):
      # return np.rint(np.sum(a * np.array([0,1,2,3]), axis=-1))
      return np.argmax(a, axis=-1)

  plt.figure(figsize=(15,5*3))

  Y2 = model.predict(X.reshape(1, *X.shape))

  plt.subplot(131)
  plt.imshow(crop_channel(X[5]))
  plt.title('Input image')

  plt.subplot(132)
  plt.imshow(crop_channel(Y[5]) * 50)
  plt.title('Segmentation')

  plt.subplot(133)
  plt.imshow(decategorize(Y2[0][5]) * 50)
  plt.title('Prediction')

  plt.show()

def plot_random_example(model, batch_generator):
  """ Plots a random patch generated by the given generator. """
  X, Y = next(batch_generator.generate_patches())
  plot_example(model, X, Y)