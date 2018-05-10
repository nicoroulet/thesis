import Datasets
import Metrics
import Models
import UNet

import os

from keras.callbacks import ModelCheckpoint  # , TensorBoard

patch_shape = (32, 32, 32)
net_depth = 4

atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
mrbrains = Datasets.MRBrainS()


def train_unet():
  """Build UNet, load the weights (if any), train, save weights."""
  dataset = mrbrains
  savedir = 'weights'
  savefile = savedir + '/unet.h5'

  # tr_gen, val_gen = dataset.get_generators(patch_shape,
  #                                        patch_multiplicity=2**net_depth)
  tr_gen = dataset.get_train_generator(patch_shape)

  n_classes = dataset.n_classes

  model = UNet.UNet(n_classes, depth=net_depth)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy',
                         Metrics.continuous.sparse_dice_coef,
                         Metrics.continuous.mean_dice_coef(),
                         Metrics.discrete.to_continuous(Metrics.discrete.
                                                        sparse_dice_coef),
                         Metrics.discrete.to_continuous(Metrics.discrete.
                                                      mean_dice_coef(n_classes))
                         ])

  print(model.summary(line_length=150, positions=[.25, .55, .67, 1.]))

  if not os.path.exists(savedir):
    os.mkdir(savedir)
  if os.path.exists(savefile):
    model.load_weights(savefile)

  model_checkpoint = ModelCheckpoint(savefile,
                                     monitor='loss',
                                     save_best_only=True)
  # model_checkpoint = Callback()
  # tensor_board = TensorBoard(log_dir='./tensorboard',
  #                            histogram_freq=0,
  #                            write_graph=True,
  #                            write_images=True)

  model.fit_generator(tr_gen.generate_batches(batch_size=5),
                      steps_per_epoch=20,
                      epochs=2,
                      # validation_data=val_gen.generate_batches(batch_size=1),
                      # validation_steps=5,
                      callbacks=[model_checkpoint])

  print("Done")


def train_multiunet(epochs=1, steps_per_epoch=20, batch_size=5):
  """Build a MultiUNet, load weights (if any), train and save weights.

  Args:
      epochs (int, optional): Description
      steps_per_epoch (int, optional): Description
      batch_size (int, optional): Description
  """
  tumor_gen = brats.get_train_generator(patch_shape)

  anatomical_gen = mrbrains.get_train_generator(patch_shape)

  tasks = [{"name": "anatomical",
            "labels": ["CSF",
                       "White matter",
                       "Gray matter"]},
           {"name": "tumor",
            "labels": ["necrosis",
                       "edema",
                       "nonenhancing tumor",
                       "enhancing tumor"]}]

  model = Models.MultiUNet(tasks)

  model.fit_generator("tumor",
                      tumor_gen.generate_batches(batch_size=batch_size),
                      steps_per_epoch=steps_per_epoch,
                      epochs=epochs)
  model.fit_generator("anatomical",
                      anatomical_gen.generate_batches(batch_size=batch_size),
                      steps_per_epoch=steps_per_epoch,
                      epochs=epochs)


def test_multiunet():
  # Using the train generator to fit into memory
  anatomical_gen = mrbrains.get_train_generator(patch_shape)

  tasks = [{"name": "tumor",
            "labels": ["necrosis",
                       "edema",
                       "nonenhancing tumor",
                       "enhancing tumor"]},
           {"name": "anatomical",
            "labels": ["CSF",
                       "White matter",
                       "Gray matter"]}]

  model = Models.MultiUNet(tasks)

  model.evaluate_generator(anatomical_gen)

train_multiunet(epochs=5, steps_per_epoch=50)
