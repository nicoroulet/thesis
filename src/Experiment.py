import BatchGenerator
import Datasets
import Metrics
import UNet



patch_shape = (64, 64, 64)
atlas = Datasets.ATLAS()
brats = Datasets.BraTS()
tr_gen, val_gen = brats.get_generators(patch_shape)

n_classes = 2

model = UNet.build_unet(n_classes, depth=2)
model.compile(loss=Metrics.sparse_dice_loss, #'sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', Metrics.sparse_dice_coef])
print(model.summary())

h = model.fit_generator(tr_gen.generate_batches(batch_size=1),
                        steps_per_epoch=5,
                        epochs=20,
                        validation_data=val_gen.generate_batches(),
                        validation_steps=5)
