import BatchGenerator
import Datasets
import Metrics
import UNet

import os

from keras.callbacks import ModelCheckpoint

patch_shape = (64, 64, 64)
n_classes = 2
net_depth = 4
savefile = 'weights.h5'

atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
tr_gen, val_gen = brats.get_generators(patch_shape,
                                       patch_multiplicity=(1<<(net_depth-1)))


model = UNet.build_unet(n_classes, depth=net_depth)

model.compile(loss=Metrics.sparse_dice_loss,#'sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       Metrics.sparse_dice_coef,
                       Metircs.sum_diff])

print(model.summary())

if os.path.exists(savefile):
  model.load_weights(savefile)

model_checkpoint = ModelCheckpoint(savefile,
                                   monitor='val_loss',
                                   save_best_only=True)

h = model.fit_generator(tr_gen.generate_batches(batch_size=1),
                        steps_per_epoch=20,
                        epochs=10,
                        validation_data=val_gen.generate_batches(),
                        validation_steps=5,
                        callbacks=[model_checkpoint])

print("Done")
