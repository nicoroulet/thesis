import BatchGenerator
import Datasets
import Metrics
import UNet

import os

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback

patch_shape = (32, 32, 32)
net_depth = 4
savefile = 'weights.h5'

atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
mrbrains = Datasets.MRBrainS()

dataset = mrbrains

def run_UNet():
  savefile = 'weights/unet_weights.h5'

  tr_gen, val_gen = dataset.get_generators(patch_shape,
                                         patch_multiplicity=(1<<(net_depth-1)))

  n_classes = dataset.n_classes

  model = UNet.UNet(n_classes, depth=net_depth)

  model.compile(loss='sparse_categorical_crossentropy',#Metrics.sparse_dice_loss,#
                optimizer='sgd',
                metrics=['accuracy',
                         Metrics.sparse_dice_coef,
                         Metrics.mean_dice_coef()
                         # Metrics.sparse_sum_diff
                         ])

  print(model.summary(line_length=150, positions=[.25, .55, .67, 1.]))

  if os.path.exists(savefile):
    model.load_weights(savefile)

  model_checkpoint = ModelCheckpoint(savefile,
                                     monitor='val_loss',
                                     save_best_only=True)
  # model_checkpoint = Callback()
  tensor_board = TensorBoard(log_dir='./tensorboard',
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)

  h = model.fit_generator(tr_gen.generate_batches(batch_size=1),
                          steps_per_epoch=20,
                          epochs=20,
                          validation_data=val_gen.generate_batches(),
                          validation_steps=5,
                          callbacks=[model_checkpoint, tensor_board])

  print("Done")

# def run_MultiUNet()
run_UNet()