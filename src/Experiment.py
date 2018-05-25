"""Execution routines."""

import Datasets
import Helpers
import Metrics
import Models

import glob
import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.metrics import sparse_categorical_accuracy

atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
mrbrains = Datasets.MRBrainS()
tasks = [
         {"name": "tumor",
          "labels": ["necrosis",
                     "edema",
                     "nonenhancing tumor",
                     "enhancing tumor"]},
         {"name": "anatomical",
          "labels": ["CSF",
                     "White matter",
                     "Gray matter"]}
        ]


def train_unet(dataset, epochs=1, steps_per_epoch=20, batch_size=5,
               patch_shape=(32, 32, 32), net_depth=4, n_channels=1):
  """Build UNet, load the weights (if any), train, save weights."""
  savedir = 'checkpoints/unet_%s_%d' % (dataset.name, net_depth)
  weights_file = '%s/weights.h5' % savedir
  epoch_file = '%s/last_epoch.txt' % savedir

  if os.path.isfile(epoch_file):
    initial_epoch = int(open(epoch_file, 'r').readline())
  else:
    initial_epoch = 0

  epochs += initial_epoch

  # tr_gen, val_gen = dataset.get_generators(patch_shape,
  #                                        patch_multiplicity=2**net_depth)
  tr_gen = dataset.get_train_generator(patch_shape)

  n_classes = dataset.n_classes

  model = Models.UNet(n_classes, depth=net_depth, n_channels=n_channels)

  model.compile(
                # loss='sparse_categorical_crossentropy',
                loss=Metrics.ContinuousMetrics.mean_dice_loss(),
                optimizer='adam',
                # optimizer=keras.optimizers.Adam(lr=.005),
                metrics=[sparse_categorical_accuracy,
                         Metrics.ContinuousMetrics.dice_coef,
                         Metrics.ContinuousMetrics.mean_dice_coef(),
                         Metrics.DiscreteMetrics.mean_dice_coef(n_classes)
                         ])

  print(model.summary(line_length=150, positions=[.25, .55, .67, 1.]))

  if os.path.exists(weights_file):
    model.load_weights(weights_file)

  if not os.path.exists(savedir):
    os.makedirs(savedir)
  model_checkpoint = ModelCheckpoint(weights_file,
                                     monitor='loss',
                                     save_best_only=True)
  # model_checkpoint = Callback()

  for file in glob.glob('tensorboard/*'):
    os.remove(file)
  tensorboard = TensorBoard(log_dir='./tensorboard',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)

  model.fit_generator(tr_gen.generate_batches(batch_size=batch_size),
                      steps_per_epoch=steps_per_epoch,
                      initial_epoch=initial_epoch,
                      epochs=epochs,
                      # validation_data=val_gen.generate_batches(batch_size=1),
                      # validation_steps=5,
                      callbacks=[model_checkpoint, tensorboard])

  open(epoch_file, 'w').write(str(epochs))
  print("Done")


def visualize_unet(dataset, net_depth=4, n_channels=1):
  """Compute one MultiUNet prediction and visualize against ground truth."""
  import numpy as np
  generator = dataset.get_train_generator((128, 128, 128))
  generator = generator.generate_batches(batch_size=1)

  n_classes = dataset.n_classes

  model = Models.UNet(n_classes, depth=net_depth, n_channels=n_channels)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd')

  savedir = 'checkpoints/unet_%s_%d' % (dataset.name, net_depth)
  weights_file = '%s/weights.h5' % savedir
  if os.path.exists(weights_file):
    model.load_weights(weights_file)

  x, y = next(generator)
  while (np.sum(y) < 500):
    x, y = next(generator)
  y_pred = np.argmax(model.predict(x), axis=-1)
  print(set(y_pred.flat))
  y = y.reshape(y.shape[:-1])
  Helpers.visualize_predictions(x[0, ..., 0], y[0], y_pred[0])


def train_multiunet(epochs=1, steps_per_epoch=20, batch_size=3,
                    patch_shape=(64, 64, 64)):
  """Build a MultiUNet, load weights (if any), train and save weights.

  Args:
      epochs (int, optional): number of training epochs
      steps_per_epoch (int, optional): number of batches per epoch
      batch_size (int, optional): size of training batches
  """
  tumor_gen = brats.get_train_generator(patch_shape)

  # anatomical_gen = mrbrains.get_train_generator(patch_shape)

  model = Models.MultiUNet(tasks)

  model.fit_generator("tumor",
                      tumor_gen.generate_batches(batch_size=batch_size),
                      steps_per_epoch=steps_per_epoch,
                      epochs=epochs)
  # model.fit_generator("anatomical",
  #                     anatomical_gen.generate_batches(batch_size=batch_size),
  #                     steps_per_epoch=steps_per_epoch,
  #                     epochs=epochs)


def test_multiunet():
  """Build a MultiUNet, load weights and evaluate on generator."""
  # Using the train generator to fit into memory
  anatomical_gen = mrbrains.get_train_generator((64, 64, 64)).generate_batches()

  model = Models.MultiUNet(tasks)

  print(model.evaluate_generator(anatomical_gen,
                                 [Metrics.discrete.accuracy,
                                  Metrics.discrete.dice_coef]))


def visualize_multiunet():
  """Compute one MultiUNet prediction and visualize against ground truth."""
  import numpy as np
  generator = brats.get_train_generator((128, 128, 128))
  generator = generator.generate_batches(batch_size=1)

  model = Models.MultiUNet(tasks)

  x, y = next(generator)
  while (np.sum(y) == 0):
    x, y = next(generator)
  y_pred = model.predict(x)
  print(set(y_pred.flat))
  y_pred[y_pred == 8] = 0
  y_pred[y_pred == 6] = 0
  y_pred[y_pred == 7] = 0
  y = y.reshape(y.shape[:-1])
  Helpers.visualize_predictions(x[0, ..., 0], y[0], y_pred[0])

# train_unet(mrbrains, epochs=10, steps_per_epoch=10)
# visualize_unet(mrbrains)
# train_unet(brats, epochs=40, steps_per_epoch=50, n_channels=4)
# visualize_unet(brats, n_channels=4)

# train_multiunet(epochs=200, steps_per_epoch=50)
# test_multiunet()
# visualize_multiunet()
