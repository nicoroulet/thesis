import BatchGenerator
import Datasets
import Metrics
import UNet

import os

from keras.callbacks import ModelCheckpoint

patch_shape = (64, 64, 64)
atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
tr_gen, val_gen = brats.get_generators(patch_shape)

n_classes = 2
savefile = 'weights.h5'

model = UNet.build_unet(n_classes, depth=4)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', Metrics.sparse_dice_coef])

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
