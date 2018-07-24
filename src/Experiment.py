"""Execution routines."""

import BatchGenerator
import Datasets
import Helpers
import Metrics
import Models
import Tools

import csv
import glob
import numpy as np
import os

from keras.callbacks import ModelCheckpoint  # , LearningRateScheduler
from keras.metrics import sparse_categorical_accuracy

np.set_printoptions(suppress=True)

atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
mrbrains13 = Datasets.MRBrainS13()
mrbrains13_val = Datasets.MRBrainS13(validation_portion=1)
ibsr = Datasets.IBSR()
mrbrains17 = Datasets.MRBrainS17()
mrbrains18 = Datasets.MRBrainS18(validation_portion=1)
mrbrains17_ibsr = Datasets.MultiDataset([mrbrains17, ibsr], mrbrains18)
mrbrains17_ibsr_ignore_bg = Datasets.MultiDataset([mrbrains17, ibsr], mrbrains18,
                                                  ignore_backgrounds=True)
mrbrains17_13 = Datasets.MultiDataset([mrbrains17, mrbrains13], mrbrains18)
mrbrains17_13_ignore_bg = Datasets.MultiDataset([mrbrains17, mrbrains13], mrbrains18,
                                                ignore_backgrounds=True)

tumor_tasks = [
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

wmh_tasks = [
         {"name": "wmh",
          "labels": ["wmh",
                     "other"]},
         {"name": "anatomical",
          "labels": ["CSF",
                     "White matter",
                     "Gray matter"]}
        ]


def train_unet(dataset, epochs=1, steps_per_epoch=200, batch_size=7,
               patch_shape=(32, 32, 32), net_depth=4, loss=None):
  """Build UNet, load the weights (if any), train, save weights."""
  savedir = Tools.get_dataset_savedir(dataset, loss)
  weights_file = '%s/weights.h5' % savedir
  epoch_file = '%s/last_epoch.txt' % savedir
  metrics_file = '%s/metrics.csv' % savedir
  full_volume_metrics_file = '%s/full_volume_metrics' % savedir
  tensorboard_dir = '%s/tensorboard' % savedir

  if os.path.isfile(epoch_file):
    initial_epoch = int(open(epoch_file, 'r').readline())
  else:
    initial_epoch = 0

  epochs += initial_epoch

  n_classes = dataset.n_classes
  n_channels = dataset.n_modalities

  if loss is None:
    loss = 'sparse_categorical_crossentropy'

  model = Models.UNet(n_classes, depth=net_depth, n_channels=n_channels)

  # print('patch_multiplicity', model.patch_multiplicity)
  # patch_tr_gen = dataset.get_train_generator(patch_shape, batch_size=batch_size)
  # patch_val_gen = val_dataset.get_val_generator(patch_shape=(128, 128, 128))

  patch_tr_gen, patch_val_gen = dataset.get_patch_generators(patch_shape, batch_size=batch_size)
  # full_tr_gen, full_val_gen = dataset.get_full_volume_generators(model.patch_multiplicity)


  model.compile(loss=loss,
                optimizer='adam',
                metrics=[sparse_categorical_accuracy,
                         Metrics.dice_coef,
                         Metrics.mean_dice_coef,
                         # Metrics.DiscreteMetrics.mean_dice_coef(n_classes)
                         ])

  print(model.summary(line_length=150, positions=[.25, .55, .67, 1.]))

  if os.path.exists(weights_file):
    model.load_weights(weights_file)
  else:
    print('WARNING: no weights found. Initializing a new net.')

  if not os.path.exists(savedir):
    os.makedirs(savedir)
  model_checkpoint = ModelCheckpoint(weights_file,
                                     monitor='val_loss',
                                     save_best_only=False)
  # model_checkpoint = Callback()

  for file in glob.glob('tensorboard/*'):
    os.remove(file)
  tensorboard = Metrics.TrainValTensorBoard(log_dir=tensorboard_dir,
                                            histogram_freq=0,
                                            write_graph=True,
                                            write_images=True)

  # def sched(epoch, lr):
  #   return lr * .99
  # lr_sched = LearningRateScheduler(sched, verbose=1)

  # full_volume_validation = Metrics.FullVolumeValidationCallback(model,
  #     full_val_gen, metrics_savefile=full_volume_metrics_file, validate_every_n_epochs=10)

  h = model.fit_generator(patch_tr_gen,
                          steps_per_epoch=steps_per_epoch,
                          initial_epoch=initial_epoch,
                          epochs=epochs,
                          validation_data=patch_val_gen,
                          validation_steps=10,
                          callbacks=[model_checkpoint,
                                     tensorboard,
                                     # lr_sched,
                                    #  full_volume_validation
                                     ])

  # Write metrics to a csv.
  keys = sorted(h.history.keys())
  if not os.path.exists(metrics_file):
    metrics_f = open(metrics_file, 'w')
    metrics_writer = csv.writer(metrics_f)
    metrics_writer.writerow(keys)
  else:
    metrics_f = open(metrics_file, 'a')
    metrics_writer = csv.writer(metrics_f)
  metrics_writer.writerows(zip(*[h.history[key] for key in keys]))

  open(epoch_file, 'w').write(str(epochs))
  print("Done")


def validate_unet(train_dataset, val_dataset=None, net_depth=4, val_steps=100, loss=None):
  """Run full volume CPU validation on both training and validation."""
  import numpy as np
  import tensorflow as tf

  import importlib.util
  spec = importlib.util.spec_from_file_location("MetricsMonitor",
                        "../../blast/blast/cnn/MetricsMonitor.py")
  MetricsMonitor = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(MetricsMonitor)

  if val_dataset is None:
    val_dataset = train_dataset
  with tf.device('/cpu:0'):
    model = Models.UNet(train_dataset.n_classes,
                        depth=net_depth,
                        n_channels=train_dataset.n_modalities)

  loaddir = Tools.get_dataset_savedir(train_dataset, loss)
  weights_file = '%s/weights.h5' % loaddir

  if loss is None:
    loss = 'sparse_categorical_crossentropy'

  model.compile(loss=loss, optimizer='sgd')

# tr_gen, val_gen = dataset.get_full_volume_generators(patch_multiplicity=model.patch_multiplicity,
  #                                                      infinite=False)
  val_gen = val_dataset.get_val_generator(patch_multiplicity=model.patch_multiplicity,
                                          infinite=False)

  if val_dataset is not train_dataset:
    # TODO this is copied from Tools.filter_modalities, refactor into ModalityFilter.
    val_gen = BatchGenerator.ModalityFilter(val_gen,
                                            val_dataset.modalities,
                                            train_dataset.modalities)

  # if os.path.exists(weights_file):
  print('Loading weights from %s...' % weights_file)
  model.load_weights(weights_file)

  metrics = []

  # for generator in [val_gen]:
  generator = val_gen

  gen_metrics = []
  for y_true, y_pred in model.predict_generator(generator, steps=val_steps):
    ignore_mask = y_true == -1
    print('y_pred', set(y_pred.flat))
    print('y_true', set(y_true.flat))
    y_true[ignore_mask] = 0
    y_pred[ignore_mask] = 0

    new_metrics = MetricsMonitor.MetricsMonitor.getMetricsForWholeSegmentation(y_true, y_pred,
                                                          labels=range(1, model.n_classes))
    new_metrics = np.nan_to_num(np.squeeze(new_metrics, axis=0))
    print(new_metrics)
    gen_metrics.append(new_metrics)
  # print(gen_metrics)
  metrics.append(np.mean(gen_metrics, axis=0))
  # print('Train:\n', metrics[0])
  print('Val:\n', metrics[0])

  metrics_file = '%s/validation_metrics' % loaddir
  if val_dataset is not train_dataset:
    metrics_file += '_' + val_dataset.name
  np.save(metrics_file, np.array(gen_metrics))


def visualize_unet(train_dataset, val_dataset=None, net_depth=4, loss=None):
  """Compute one MultiUNet prediction and visualize against ground truth."""
  if val_dataset is None:
    val_dataset = train_dataset
  generator = val_dataset.get_val_generator((128, 128, 128),
                                            transformations=BatchGenerator.Transformations.CROP,
                                            batch_size=1)
  if val_dataset is not train_dataset:
    generator = BatchGenerator.ModalityFilter(generator,
                                              val_dataset.modalities,
                                              train_dataset.modalities)

  model = Models.UNet(train_dataset.n_classes, depth=net_depth,
                      n_channels=train_dataset.n_modalities)

  savedir = Tools.get_dataset_savedir(train_dataset, loss)
  weights_file = '%s/weights.h5' % savedir

  if loss is None:
    loss = 'sparse_categorical_crossentropy'

  model.compile(loss=loss, optimizer='sgd')

  print('Loading weights from %s...' % weights_file)
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
  tumor_gen = brats.get_train_generator(patch_shape, batch_size=batch_size)

  # anatomical_gen = mrbrains.get_train_generator(patch_shape, batch_size=batch_size)

  model = Models.MultiUNet(tumor_tasks)

  model.fit_generator("tumor",
                      tumor_gen,
                      steps_per_epoch=steps_per_epoch,
                      epochs=epochs)
  # model.fit_generator("anatomical",
  #                     anatomical_gen,
  #                     steps_per_epoch=steps_per_epoch,
  #                     epochs=epochs)


def validate_multiunet(multiunet_datasets, val_dataset, net_depth=4, loss=None):
  """Run MultiUNet full volume CPU validation on a combine dataset (containing all labels)."""
  import numpy as np
  import tensorflow as tf
  # import matplotlib.pyplot as plt

  import importlib.util
  spec = importlib.util.spec_from_file_location("MetricsMonitor",
                        "../../blast/blast/cnn/MetricsMonitor.py")
  MetricsMonitor = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(MetricsMonitor)

  np.set_printoptions(suppress=True)

  with tf.device('/cpu:0'):
    model = Models.MultiUNet(multiunet_datasets, depth=net_depth)

  val_gen = val_dataset.get_val_generator(patch_multiplicity=model.patch_multiplicity,
                                          infinite=False)

  metrics = []
  for y_true, y_pred in model.predict_generator(val_gen, steps=100,
                                                modalities=val_dataset.modalities):
    ignore_mask = y_true == -1
    y_true[ignore_mask] = 0
    y_pred[ignore_mask] = 0

    # import pdb
    # pdb.set_trace()

    metric = MetricsMonitor.MetricsMonitor.getMetricsForWholeSegmentation(y_true, y_pred,
                                                          labels=range(1, model.n_classes))[0]
    print(metric)
    metrics.append(metric)
  print('Average metrics:')
  print(np.mean(metrics, axis=0))

  savedir = Tools.get_dataset_savedir(val_dataset, loss)
  metrics_file = '%s/validation_metrics' % savedir
  np.save(metrics_file, np.array(metrics))


def test_multiunet():
  """Build a MultiUNet, load weights and evaluate on generator."""
  # Using the train generator to fit into memory
  anatomical_gen = mrbrains13.get_train_generator((64, 64, 64))

  model = Models.MultiUNet(tumor_tasks)

  print(model.evaluate_generator(anatomical_gen,
                                 [Metrics.discrete.accuracy,
                                  Metrics.discrete.dice_coef]))


def visualize_multiunet():
  """Compute one MultiUNet prediction and visualize against ground truth."""
  import numpy as np
  generator = brats.get_train_generator((128, 128, 128), batch_size=1)

  model = Models.MultiUNet(tumor_tasks)

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


if __name__ == '__main__':
  # --------------------------------- MRBrainS13 ---------------------------------------------------
  # train_unet(mrbrains13, epochs=50)
  # train_unet(mrbrains13, epochs=30, loss=Metrics.Wmean_dice_loss)

  # validate_unet(mrbrains13)
  # validate_unet(mrbrains13, loss=Metrics.dice_loss)
  # validate_unet(mrbrains13, loss=Metrics.mean_dice_loss)

  # visualize_unet(mrbrains13)

  # --------------------------------- IBSR ---------------------------------------------------------
  # train_unet(ibsr, epochs=100)
  # train_unet(ibsr, epochs=100, loss=Metrics.dice_loss)
  # train_unet(ibsr, epochs=50, loss=Metrics.mean_dice_loss)

  # validate_unet(ibsr)
  # validate_unet(ibsr, loss=Metrics.dice_loss)
  # validate_unet(ibsr, loss=Metrics.mean_dice_loss)
  # validate_unet(ibsr, mrbrains13_val)

  # visualize_unet(ibsr)

  # ---------------------------------- BraTS -------------------------------------------------------
  # train_unet(brats, epochs=1, steps_per_epoch=5000, batch_size=2)
  # validate_unet(brats, val_steps=10)
  # visualize_unet(brats)

  # ---------------------------------- ATLAS -------------------------------------------------------
  # train_unet(atlas, epochs=4, steps_per_epoch=10000, batch_size=10)
  # validate_unet(atlas, val_steps=10)
  # visualize_unet(atlas)

  # --------------------------------- MRBrainS17 ---------------------------------------------------
  # train_unet(mrbrains17, epochs=100)
  # train_unet(mrbrains17, epochs=100, loss=Metrics.dice_loss)
  # train_unet(mrbrains17, epochs=50, loss=Metrics.mean_dice_loss)

  # validate_unet(mrbrains17)
  # validate_unet(mrbrains17, loss=Metrics.dice_loss)
  # validate_unet(mrbrains17, loss=Metrics.mean_dice_loss)

  # visualize_unet(mrbrains17)

  # --------------------------------- MRBrainS17_IBSR ----------------------------------------------
  # train_unet(mrbrains17_ibsr, epochs=50)
  # train_unet(mrbrains17_ibsr, epochs=50, loss=Metrics.dice_loss)
  # train_unet(mrbrains17_ibsr, epochs=50, loss=Metrics.mean_dice_loss)
  # train_unet(mrbrains17_ibsr, epochs=200, loss=Metrics.selective_dice_loss)
  # train_unet(mrbrains17_ibsr_ignore_bg, epochs=50,
  #            loss=Metrics.selective_sparse_categorical_crossentropy)
  train_unet(mrbrains17_ibsr_ignore_bg, epochs=100,
             loss=Metrics.improved_selective_sparse_categorical_crossentropy)


  # validate_unet(mrbrains17_ibsr)
  # validate_unet(mrbrains17_ibsr, mrbrains18)
  # validate_unet(mrbrains17_ibsr, mrbrains18, loss=Metrics.dice_loss)
  # validate_unet(mrbrains17_ibsr, mrbrains18, loss=Metrics.mean_dice_loss)
  validate_unet(mrbrains17_ibsr, mrbrains18, loss=Metrics.selective_dice_loss)
  validate_unet(mrbrains17_ibsr_ignore_bg, mrbrains18,
                loss=Metrics.selective_sparse_categorical_crossentropy)
  validate_unet(mrbrains17_ibsr_ignore_bg, mrbrains18,
                loss=Metrics.improved_selective_sparse_categorical_crossentropy)

  # visualize_unet(mrbrains17_ibsr, loss=Metrics.dice_loss)
  # visualize_unet(mrbrains17_ibsr, mrbrains18)

  # ---------------------------------- MRBrainS17_13 -----------------------------------------------
  # print(mrbrains17_13.train_paths)
  # train_unet(mrbrains17_13, epochs=50)
  # train_unet(mrbrains17_13, epochs=50, loss=Metrics.dice_loss)
  # train_unet(mrbrains17_13, epochs=50, loss=Metrics.mean_dice_loss)
  train_unet(mrbrains17_13, epochs=100, loss=Metrics.selective_dice_loss)
  # train_unet(mrbrains17_13_ignore_bg, epochs=100,
  #            loss=Metrics.selective_sparse_categorical_crossentropy)
  train_unet(mrbrains17_13_ignore_bg, epochs=100,
             loss=Metrics.improved_selective_sparse_categorical_crossentropy)

  # validate_unet(mrbrains17_13, mrbrains18)
  # validate_unet(mrbrains17_13, mrbrains18, loss=Metrics.dice_loss)
  # validate_unet(mrbrains17_13, mrbrains18, loss=Metrics.mean_dice_loss)
  validate_unet(mrbrains17_13, mrbrains18, loss=Metrics.selective_dice_loss)
  # validate_unet(mrbrains17_13_ignore_bg, mrbrains18,
  #               loss=Metrics.selective_sparse_categorical_crossentropy)
  validate_unet(mrbrains17_13_ignore_bg, mrbrains18,
                loss=Metrics.improved_selective_sparse_categorical_crossentropy)

  # visualize_unet(mrbrains17_13, mrbrains18, loss=Metrics.mean_dice_loss)
  # visualize_unet(mrbrains17_13, mrbrains18)

  # ---------------------------------- MultiUNet ---------------------------------------------------
  # validate_multiunet([mrbrains17, ibsr], mrbrains18)
  # validate_multiunet([mrbrains17, mrbrains13], mrbrains18)

  # train_multiunet(epochs=200, steps_per_epoch=50)
  # test_multiunet()
  # visualize_multiunet()
