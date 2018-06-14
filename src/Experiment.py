"""Execution routines."""

import BatchGenerator
import Datasets
import Helpers
import Metrics
import Models

import glob
import os

import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.metrics import sparse_categorical_accuracy

atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
mrbrains = Datasets.MRBrainS()
ibsr = Datasets.IBSR()

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
  metrics_file = '%s/metrics' % savedir
  tensorboard_dir = '%s/tensorboard' % savedir

  if os.path.isfile(epoch_file):
    initial_epoch = int(open(epoch_file, 'r').readline())
  else:
    initial_epoch = 0

  epochs += initial_epoch

  n_classes = dataset.n_classes

  model = Models.UNet(n_classes, depth=net_depth, n_channels=n_channels)

  print('patch_multiplicity', model.patch_multiplicity)
  patch_tr_gen, patch_val_gen = dataset.get_patch_generators(patch_shape)
  full_tr_gen, full_val_gen = dataset.get_full_volume_generators(
                                                      model.patch_multiplicity)

  model.compile(
                loss='sparse_categorical_crossentropy',
                # loss=Metrics.ContinuousMetrics.dice_loss,
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
                                     monitor='val_loss',
                                     save_best_only=True)
  # model_checkpoint = Callback()

  for file in glob.glob('tensorboard/*'):
    os.remove(file)
  tensorboard = TensorBoard(log_dir=tensorboard_dir,
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)

  def sched(epoch, lr):
    return lr * .95
  lr_sched = LearningRateScheduler(sched, verbose=1)

  full_volume_validation = Metrics.FullVolumeValidationCallback(model,
      full_val_gen, metrics_savefile=metrics_file, validate_every_n_epochs=1)

  model.fit_generator(patch_tr_gen.generate_batches(batch_size=batch_size),
                   steps_per_epoch=steps_per_epoch,
                   initial_epoch=initial_epoch,
                   epochs=epochs,
                   validation_data=patch_val_gen.generate_batches(
                                                        batch_size=batch_size),
                   validation_steps=10,
                   callbacks=[model_checkpoint,
                              tensorboard,
                              lr_sched,
                              full_volume_validation
                              ])

  open(epoch_file, 'w').write(str(epochs))
  print("Done")


def validate_unet(dataset, net_depth=4, n_channels=1, val_steps=5):
  """Run full volume CPU validation on both training and validation."""
  import numpy as np
  import tensorflow as tf

  import importlib.util
  spec = importlib.util.spec_from_file_location("MetricsMonitor",
                        "../../blast/blast/cnn/MetricsMonitor.py")
  MetricsMonitor = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(MetricsMonitor)

  # tr_gen = tr_gen.generate_batches(batch_size=1)
  # val_gen = val_gen.generate_batches(batch_size=1)

  n_classes = dataset.n_classes

  with tf.device('/cpu:0'):
    model = Models.UNet(n_classes, depth=net_depth, n_channels=n_channels)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd')

  tr_gen, val_gen = dataset.get_full_volume_generators(patch_multiplicity=model.patch_multiplicity)

  savedir = 'checkpoints/unet_%s_%d' % (dataset.name, net_depth)
  weights_file = '%s/weights.h5' % savedir
  # if os.path.exists(weights_file):
  print('Loading weights...')
  model.load_weights(weights_file)

  metrics = []
  for generator in (tr_gen, val_gen):
    gen_metrics = []
    for y_true, y_pred in model.predict_generator(generator, steps=val_steps):
      gen_metrics.append(MetricsMonitor.MetricsMonitor.getMetricsForWholeSegmentation(y_true,
                                                            y_pred,
                                                            labels=range(1, model.n_classes))[0])
    print(gen_metrics)
    metrics.append(np.mean(gen_metrics, axis=0))
  print('Train:\n', metrics[0])
  print('Val:\n', metrics[1])


def visualize_unet(dataset, net_depth=4, n_channels=1):
  """Compute one MultiUNet prediction and visualize against ground truth."""
  import numpy as np
  generator = dataset.get_val_generator((128, 128, 128),
                                        transformations=BatchGenerator.Transformations.CROP)
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

train_unet(ibsr, epochs=10, steps_per_epoch=2000, batch_size=10)
# validate_unet(ibsr, val_steps=10)
# visualize_unet(ibsr)

# train_unet(brats, epochs=1, steps_per_epoch=5000, n_channels=4, batch_size=2)
# validate_unet(brats, val_steps=10, n_channels=4)
# visualize_unet(brats, n_channels=4)

# train_unet(atlas, epochs=4, steps_per_epoch=10000, patch_shape=(32, 32, 32), batch_size=10)
# validate_unet(atlas, val_steps=10)
# visualize_unet(atlas)

# train_multiunet(epochs=200, steps_per_epoch=50)
# test_multiunet()
# visualize_multiunet()
