import BatchGenerator
import Datasets
import Metrics
import UNet

import os

from keras.callbacks import ModelCheckpoint

patch_shape = (32, 32, 32)
net_depth = 4
savefile = 'weights.h5'

atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
mrbrains = Datasets.MRBrainS()

dataset = mrbrains

tr_gen, val_gen = mrbrains.get_generators(patch_shape,
                                       patch_multiplicity=(1<<(net_depth-1)))

n_classes = dataset.n_classes

model = UNet.build_unet(n_classes, depth=net_depth)

model.compile(loss='sparse_categorical_crossentropy',#Metrics.sparse_dice_loss,#
              optimizer='adam',
              metrics=['accuracy',
                       Metrics.sparse_dice_coef,
                       Metrics.sparse_sum_diff])

print(model.summary(line_length=150, positions=[.25, .55, .67, 1.]))

if os.path.exists(savefile):
  model.load_weights(savefile)

model_checkpoint = ModelCheckpoint(savefile,
                                   monitor='val_loss',
                                   save_best_only=True)

h = model.fit_generator(tr_gen.generate_batches(batch_size=1),
                        steps_per_epoch=200,
                        epochs=10,
                        validation_data=val_gen.generate_batches(),
                        validation_steps=5,
                        callbacks=[model_checkpoint])

print("Done")
