"""Execution routines."""

import BatchGenerator
import Datasets
import Helpers
import Logger
import Metrics
import Models
import Tools

import csv
import glob
import numpy as np
import os
import pandas as pd

from keras.callbacks import ModelCheckpoint  # , LearningRateScheduler
from keras.metrics import sparse_categorical_accuracy

np.set_printoptions(suppress=True)

brats12 = Datasets.BraTS12()
brainweb = Datasets.BrainWeb()
tumorsim = Datasets.TumorSim(validation_portion=1)
brats12_brainweb = Datasets.MultiDataset([brats12, brainweb])
brats12_ignore_bg_brainweb = Datasets.MultiDataset([brats12, brainweb],
                                                   ignore_backgrounds=True)

mrbrains13 = Datasets.MRBrainS13()
mrbrains13_val = Datasets.MRBrainS13(validation_portion=1)
ibsr = Datasets.IBSR()
mrbrains17 = Datasets.MRBrainS17()
mrbrains18 = Datasets.MRBrainS18(validation_portion=1)
mrbrains17_ibsr = Datasets.MultiDataset([mrbrains17, ibsr])
mrbrains17_ibsr_ignore_bg = Datasets.MultiDataset([mrbrains17, ibsr],
                                                  ignore_backgrounds=True)
# mrbrains17_ibsr_balance_labels = Datasets.MultiDataset([mrbrains17, ibsr],
#                                                        balance='labels')
# mrbrains17_ibsr_ignore_bg_balance_labels = Datasets.MultiDataset([mrbrains17, ibsr],
#                                                 ignore_backgrounds=True, balance='labels')

mrbrains17_13 = Datasets.MultiDataset([mrbrains17, mrbrains13])
mrbrains17_13_ignore_bg = Datasets.MultiDataset([mrbrains17, mrbrains13],
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

def load_weights(model, primary_file, secondary_file=None):
  """Load model weights from primary_file if it exists, otherwise try with secondary_file."""
  if os.path.exists(primary_file):
    Logger.info("Loading weights from %s..." % primary_file)
    model.load_weights(primary_file)
  elif secondary_file is not None and os.path.exists(secondary_file):
    Logger.info("Loading weights from %s..." % secondary_file)
    model.load_weights(secondary_file)
  else:
    Logger.warning('No weights found. Initializing a new net at %s.' % primary_file)


def train_unet(dataset, epochs=1, steps_per_epoch=200, batch_size=7,
               patch_shape=(32, 32, 32), net_depth=4, loss=None):
  """Build UNet, load the weights (if any), train, save weights."""
  savedir = Tools.get_dataset_savedir(dataset, loss)
  weights_file = '%s/weights.h5' % savedir
  best_weights_file = '%s/best_weights.h5' % savedir
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
                         Metrics.mean_dice_coef,
                         ])

  print(model.summary(line_length=150, positions=[.25, .55, .67, 1.]))

  load_weights(model, weights_file)

  if not os.path.exists(savedir):
    os.makedirs(savedir)
  model_checkpoint = ModelCheckpoint(weights_file,
                                     monitor='val_loss',
                                     save_best_only=False)
  best_model_checkpoint = ModelCheckpoint(best_weights_file,
                                     monitor='val_loss',
                                     save_best_only=True)

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
                                     best_model_checkpoint,
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
  Logger.info("Done")


def validate_unet(train_dataset, val_dataset=None, net_depth=4, val_steps=100, loss=None, tag=''):
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
  weights_file = '%s/best_weights.h5' % loaddir
  secondary_weights_file = '%s/weights.h5' % loaddir

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

  load_weights(model, weights_file, secondary_weights_file)


  # for generator in [val_gen]:
  generator = val_gen

  Logger.info('Running validation on %s, trained with %s' %(val_dataset.name, train_dataset.name))
  metrics = []
  for y_true, y_pred in model.predict_generator(generator, steps=val_steps):
    ignore_mask = y_true == -1
    Logger.debug('y_pred labels:', set(y_pred.flat), '- y_true labels:', set(y_true.flat))
    y_true[ignore_mask] = 0
    y_pred[ignore_mask] = 0

    new_metrics = MetricsMonitor.MetricsMonitor.getMetricsForWholeSegmentation(y_true, y_pred,
                                                          labels=range(1, model.n_classes))
    new_metrics = np.nan_to_num(np.squeeze(new_metrics, axis=0))
    metrics.append(new_metrics)
  metrics = np.array(metrics)
  metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice', 'Jaccard']
  df = pd.DataFrame()
  for i, clss in enumerate(val_dataset.classes[1:]):
    tmp_df = pd.DataFrame(metrics[:, i, :], columns=metric_labels)
    tmp_df['Class'] = clss
    df = df.append(tmp_df)
  if tag:
    df['Loss function'] = tag
  Logger.debug('Validation metrics:\n', df.groupby(['Loss function', 'Class']).mean())

  metrics_file = '%s/validation_metrics' % loaddir
  if val_dataset is not train_dataset:
    metrics_file += '_' + val_dataset.name
  metrics_file += '.csv'
  Logger.info('Saving validation metrics to', metrics_file)
  # np.save(metrics_file, np.array(gen_metrics))
  df.to_csv(metrics_file)


def visualize_unet(train_dataset, val_dataset=None, net_depth=4, loss=None, savefile=''):
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
  weights_file = '%s/best_weights.h5' % savedir
  secondary_weights_file = '%s/weights.h5' % savedir

  if loss is None:
    loss = 'sparse_categorical_crossentropy'

  model.compile(loss=loss, optimizer='sgd')

  print(model.summary(line_length=150, positions=[.25, .55, .67, 1.]))

  load_weights(model, weights_file, secondary_weights_file)

  x, y = next(generator)
  while (np.sum(y) < 500):
    x, y = next(generator)
  y_pred = np.argmax(model.predict(x), axis=-1)
  print(set(y_pred.flat))
  y = y.reshape(y.shape[:-1])
  Helpers.visualize_predictions(x[0, ..., 0], y[0], y_pred[0], savefile=savefile)


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


def validate_multiunet(multiunet_datasets, val_dataset, net_depth=4, loss=None, tag='MultiUNet'):
  """Run MultiUNet full volume CPU validation on a combined dataset (containing all labels)."""
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
  Logger.info('Running validation on %s, MultiUNet trained with %s' % (val_dataset.name,
              ', '.join(d.name for d in multiunet_datasets)))

  metrics = []
  for y_true, y_pred in model.predict_generator(val_gen, steps=100,
                                                modalities=val_dataset.modalities):
    ignore_mask = y_true == -1
    y_true[ignore_mask] = 0
    y_pred[ignore_mask] = 0
    Logger.debug('y_pred labels:', set(y_pred.flat), '- y_true labels:', set(y_true.flat))
    metric = MetricsMonitor.MetricsMonitor.getMetricsForWholeSegmentation(y_true, y_pred,
                                                          labels=range(1, model.n_classes))[0]
    Logger.debug('Metrics: (classes =', val_dataset.classes, ')\n', metric)
    # Helpers.visualize_predictions(y_true[0], y_true[0], y_pred[0], savefile='validation')

    metrics.append(metric)

  metrics = np.array(metrics)
  metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice', 'Jaccard']
  df = pd.DataFrame()
  for i, clss in enumerate(val_dataset.classes[1:]):
    tmp_df = pd.DataFrame(metrics[:, i, :], columns=metric_labels)
    tmp_df['Class'] = clss
    df = df.append(tmp_df)
  if tag:
    df['Loss function'] = tag
  Logger.debug('Validation metrics:\n', df.groupby(['Loss function', 'Class']).mean())


  savedir = Tools.get_dataset_savedir(val_dataset, loss)
  if not os.path.exists(savedir):
    os.makedirs(savedir)
  metrics_file = '%s/validation_metrics.csv' % savedir
  Logger.info('Saving validation metrics to', metrics_file)
  # np.save(metrics_file, np.array(metrics))
  df.to_csv(metrics_file)


def test_multiunet():
  """Build a MultiUNet, load weights and evaluate on generator."""
  # Using the train generator to fit into memory
  anatomical_gen = mrbrains13.get_train_generator((64, 64, 64))

  model = Models.MultiUNet(tumor_tasks)

  print(model.evaluate_generator(anatomical_gen,
                                 [Metrics.discrete.accuracy,
                                  Metrics.discrete.dice_coef]))


def visualize_multiunet(train_datasets, val_dataset, net_depth=4, loss=None, savefile=''):
  """Compute one MultiUNet prediction and visualize against ground truth."""
  Logger.info("Visualizing multiunet predictions.")
  import numpy as np
  generator = val_dataset.get_val_generator((128, 128, 128),
                                            transformations=BatchGenerator.Transformations.CROP,
                                            batch_size=1)
  model = Models.MultiUNet(train_datasets, depth=net_depth, loss=loss)

  x, y = next(generator)
  while (np.sum(y) == 0):
    x, y = next(generator)
  y_pred = model.predict(x, modalities=val_dataset.modalities)
  Logger.debug("Prediction labels:", set(y_pred.flat))
  y = y.reshape(y.shape[:-1])

  Helpers.visualize_predictions(x[0, ..., 0], y[0], y_pred[0], savefile=savefile)


if __name__ == '__main__':
  Tools.set_model_subdir('no_vessels_metrics')

  "--------------------------------- MRBrainS13 ---------------------------------------------------"
  # train_unet(mrbrains13, epochs=50)
  # train_unet(mrbrains13, epochs=30, loss=Metrics.mean_dice_loss)

  # validate_unet(mrbrains13, tag='Single task')
  # validate_unet(mrbrains13, loss=Metrics.dice_loss)
  # validate_unet(mrbrains13, loss=Metrics.mean_dice_loss)

  # visualize_unet(mrbrains13)

  "--------------------------------- IBSR ---------------------------------------------------------"
  # train_unet(ibsr, epochs=50)
  # train_unet(ibsr, epochs=100, loss=Metrics.selective_dice_loss)
  # train_unet(ibsr, epochs=50, loss=Metrics.mean_dice_loss)

  # validate_unet(ibsr, tag='Single task')
  # validate_unet(ibsr, loss=Metrics.selective_dice_loss)
  # validate_unet(ibsr, loss=Metrics.mean_dice_loss)
  # validate_unet(ibsr, mrbrains13_val)

  # visualize_unet(ibsr)

  "---------------------------------- BraTS BrainWeb TumorSim -------------------------------------"

  # train_unet(brainweb, epochs=100)
  # train_unet(brainweb, epochs=100, loss=Metrics.mean_dice_loss)

  # train_unet(brats12, epochs=100)
  # train_unet(brats12, epochs=100, loss=Metrics.mean_dice_loss)

  train_unet(brats12_brainweb, epochs=100, loss=Metrics.selective_dice_loss)
  train_unet(brats12_brainweb, epochs=100)
  train_unet(brats12_brainweb, epochs=100, loss=Metrics.mean_dice_loss)
  train_unet(brats12_ignore_bg_brainweb, epochs=100,
             loss=Metrics.selective_sparse_categorical_crossentropy)
  # train_unet(brats12_brainweb, epochs=200, loss=Metrics.bg_selective_sparse_categorical_crossentropy)



  # validate_unet(brainweb, tag='Single task')
  # validate_unet(brats12, tag='Single task')
  # validate_unet(brats12, loss=Metrics.mean_dice_loss, tag='Single task mean dice')

  # validate_unet(brats12_brainweb, tumorsim, tag='Crossentropy')
  # validate_unet(brats12_brainweb, tumorsim, loss=Metrics.mean_dice_loss, tag='Mean dice')
  # validate_unet(brats12_brainweb, tumorsim, loss=Metrics.selective_dice_loss, tag='Selective dice')
  # validate_unet(brats12_brainweb, tumorsim, loss=Metrics.bg_selective_sparse_categorical_crossentropy,
  #               tag='BG selective crossentropy')
  # validate_unet(brats12_ignore_bg_brainweb, tumorsim,
  #               loss=Metrics.selective_sparse_categorical_crossentropy, tag='Selective crossentropy')

  # validate_multiunet([brats12, brainweb], tumorsim, tag='MultiUNet')
  # visualize_unet(brainweb, tumorsim, loss=Metrics.mean_dice_loss, savefile='single_net')
  # visualize_multiunet([brats12, brainweb], tumorsim, savefile='multiunet')

  "---------------------------------- ATLAS -------------------------------------------------------"
  # train_unet(atlas, epochs=4, steps_per_epoch=10000, batch_size=10)
  # validate_unet(atlas, val_steps=10)
  # visualize_unet(atlas)

  "--------------------------------- MRBrainS17 ---------------------------------------------------"
  # train_unet(mrbrains17, epochs=50)
  # train_unet(mrbrains17, epochs=50, loss=Metrics.mean_dice_loss)

  # validate_unet(mrbrains17, tag='Single task')
  # validate_unet(mrbrains17, loss=Metrics.dice_loss)
  # validate_unet(mrbrains17, loss=Metrics.mean_dice_loss)

  # visualize_unet(mrbrains17)

  "--------------------------------- MRBrainS17_IBSR ----------------------------------------------"
  # Tools.set_model_subdir('regularization_metrics')
  # train_unet(mrbrains17_ibsr, epochs=20)
  # train_unet(mrbrains17_ibsr, epochs=200, loss=Metrics.mean_dice_loss)
  # train_unet(mrbrains17_ibsr, epochs=200, loss=Metrics.selective_dice_loss)
  # train_unet(mrbrains17_ibsr_ignore_bg, epochs=200,
  #            loss=Metrics.selective_sparse_categorical_crossentropy)
  # train_unet(mrbrains17_ibsr, epochs=200, loss=Metrics.bg_selective_sparse_categorical_crossentropy)

  # validate_unet(mrbrains17_ibsr, mrbrains18, tag='Crossentropy')
  # validate_unet(mrbrains17_ibsr, mrbrains18, loss=Metrics.mean_dice_loss, tag='Mean dice')
  # validate_unet(mrbrains17_ibsr, mrbrains18, loss=Metrics.selective_dice_loss, tag='Selective dice')
  # validate_unet(mrbrains17_ibsr_ignore_bg, mrbrains18,
  #               loss=Metrics.selective_sparse_categorical_crossentropy, tag='Selective crossentropy')
  # validate_unet(mrbrains17_ibsr, mrbrains18, loss=Metrics.bg_selective_sparse_categorical_crossentropy, tag='BG selective crossentropy')

  # visualize_unet(mrbrains17_ibsr_balance_labels, mrbrains13)
  # visualize_unet(mrbrains17_ibsr_balance_labels, mrbrains13, loss=Metrics.mean_dice_loss)

  # validate_multiunet([mrbrains17, ibsr], mrbrains18, tag='MultiUNet')

  "Balance labels"
  Tools.set_model_subdir('balanced_labels_metrics')
  # train_unet(mrbrains17_ibsr_balance_labels, epochs=200)
  # train_unet(mrbrains17_ibsr_balance_labels, epochs=200, loss=Metrics.mean_dice_loss)
  # train_unet(mrbrains17_ibsr_balance_labels, epochs=200, loss=Metrics.selective_dice_loss)
  # train_unet(mrbrains17_ibsr_ignore_bg_balance_labels, epochs=200,
  #            loss=Metrics.selective_sparse_categorical_crossentropy)

  # validate_unet(mrbrains17_ibsr_balance_labels, mrbrains18)
  # validate_unet(mrbrains17_ibsr_balance_labels, mrbrains18, loss=Metrics.mean_dice_loss)
  # validate_unet(mrbrains17_ibsr_balance_labels, mrbrains18, loss=Metrics.selective_dice_loss)
  # validate_unet(mrbrains17_ibsr_ignore_bg_balance_labels, mrbrains18,
  #               loss=Metrics.selective_sparse_categorical_crossentropy)

  # visualize_unet(mrbrains17_ibsr_balance_labels, mrbrains13)
  # visualize_unet(mrbrains17_ibsr_balance_labels, mrbrains13, loss=Metrics.selective_dice_loss)
  Tools.set_model_subdir('final_metrics')

  "---------------------------------- MRBrainS17_13 -----------------------------------------------"
  # train_unet(mrbrains17_13, epochs=200)
  # train_unet(mrbrains17_13, epochs=200, loss=Metrics.mean_dice_loss)
  # train_unet(mrbrains17_13, epochs=200, loss=Metrics.selective_dice_loss)
  # train_unet(mrbrains17_13_ignore_bg, epochs=200,
  #            loss=Metrics.selective_sparse_categorical_crossentropy)

  # validate_unet(mrbrains17_13, mrbrains18, tag='Crossentropy')
  # validate_unet(mrbrains17_13, mrbrains18, loss=Metrics.mean_dice_loss, tag='Mean dice')
  # validate_unet(mrbrains17_13, mrbrains18, loss=Metrics.selective_dice_loss, tag='Selective dice')
  # validate_unet(mrbrains17_13_ignore_bg, mrbrains18,
  #               loss=Metrics.selective_sparse_categorical_crossentropy, tag='Selective crossentropy')

  # visualize_unet(mrbrains17_13, mrbrains18, loss=Metrics.selective_dice_loss)
  # visualize_unet(mrbrains17_13, mrbrains18)

  # validate_multiunet([mrbrains17, mrbrains13], mrbrains18, tag='MultiUNet')
  "---------------------------------- MultiUNet ---------------------------------------------------"

  # train_multiunet(epochs=200, steps_per_epoch=50)
  # test_multiunet()
  # visualize_multiunet()
