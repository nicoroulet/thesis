"""Collection of Brain datasets.

- MRBrainS (anatomical).
- IBSR (anatomical).
- BraTS (tumors).
- ATLAS (stroke lesion).
"""
from glob import glob
import nibabel as nib
import numpy as np
import pickle
from nilearn.image import resample_img

from BatchGenerator import BatchGenerator, Transformations


def normalize(X):
  """Normalize a given image.

  Args:
      X (Numpy array): Input image

  Returns:
      Numpy array: normalized image
  """
  # X -= np.mean(X)
  X /= np.std(X)
  return X


def resample_to_1mm(img, interpolation='continuous'):
  """Resample given image to have an identity affine (thus, a 1mm3 voxel size).

  Args:
      img (Nifty Image): Input image
      interpolation (string): resampling method. 'continuous', 'nearest' or
          'linear'

  Returns:
      Nifty Image: Resampled image
  """
  return resample_img(img, target_affine=np.eye(3), interpolation=interpolation)


def preprocess_dataset(dataset, root_dir):
  """Take a dataset and stores it into memory as plain numpy arrays.

  Args:
      dataset: input dataste
      root_dir (string): directory to store dataset
  """
  import os

  if not os.path.exists(root_dir):
    os.makedirs(root_dir)

  # Shuffle paths in order to guarrantee that training/validation portions are
  # random and not determined by some arbitrary property like file names.
  # Note that when loading the dataset, the partition will not be random, only
  # when preprocessing it.
  paths = dataset.train_paths + dataset.val_paths
  np.random.shuffle(paths)

  for i, path in enumerate(paths):
    save_path = root_dir + '/%d' % i
    print('Processing path: %s' % path)
    data, seg = dataset.load_path(path)

    assert(data.shape[:-1] == seg.shape[:-1])
    np.savez(save_path,
             data=data,
             seg=seg)


class Dataset:
  """ Abstract Dataset class.

  Requires the subclass to implement the load_path and _get_paths methods.
  """

  def __init__(self, root_path, validation_portion=.2):
    paths = self._get_paths(root_path)
    val_n = int(len(paths) * validation_portion)
    if val_n == 0:
      self.train_paths = paths
      self.val_paths = []
    else:
      self.train_paths = paths[:-val_n]
      self.val_paths = paths[-val_n:]

  def load_path(self, path):
    raise NotImplementedError()

  def _get_paths(self, root_path):
    """Get a meaningful path to each sample from the root path."""
    raise NotImplementedError()

  @property
  def n_images(self):
    return len(self.train_paths) + len(self.val_paths)

  def get_train_generator(self,
                          patch_shape=(32, 32, 32),
                          max_queue_size=5,
                          pool_size=5,
                          pool_refresh_period=20,
                          transformations=Transformations.ALL,
                          patch_multiplicity=1):
    return BatchGenerator(patch_shape,
                          self.train_paths,
                          self.load_path,
                          max_queue_size=max_queue_size,
                          pool_size=pool_size,
                          pool_refresh_period=pool_refresh_period,
                          transformations=transformations,
                          patch_multiplicity=patch_multiplicity)

  def get_val_generator(self,
                        patch_shape=None,
                        max_queue_size=2,
                        pool_size=1,
                        pool_refresh_period=1,
                        transformations=Transformations.NONE,
                        patch_multiplicity=1):
    return BatchGenerator(patch_shape,
                          self.val_paths,
                          self.load_path,
                          max_queue_size=max_queue_size,
                          pool_size=pool_size,
                          pool_refresh_period=pool_refresh_period,
                          transformations=transformations,
                          patch_multiplicity=patch_multiplicity)

  # def get_train_generator(self, patch_shape):
  #   return FetcherGenerator(5, patch_shape,
  #                              self.train_paths,
  #                              self.load_path)

  # def get_val_generator(self, patch_multiplicity):
  #   return FetcherGenerator(1, None,
  #                              self.val_paths,
  #                              self.load_path,
  #                              transformations=Transformations.NONE,
  #                              patch_multiplicity=patch_multiplicity)

  def get_patch_generators(self, patch_shape):
    """Get both training generator and validation patched batch generators.

    Both crop patches from the images. The training generator also applies
    augmentation (gaussian noise and random flipping).

    Args:
        patch_shape: dimensions of the training patches.

    Returns:
        tuple: `train_generator`, `val_generator`.
    """
    return (self.get_train_generator(patch_shape=patch_shape),
            self.get_val_generator(patch_shape=patch_shape,
                                   max_queue_size=3,
                                   pool_size=5,
                                   pool_refresh_period=20,
                                   transformations=Transformations.CROP))

  def get_full_volume_generators(self, patch_multiplicity=1):
    """Get both training generator and validation full volume batch generators.

    Both yield full-volume images, without augmentation.

    Args:
        patch_multiplicity (int, optional): Enforced multiplicity of image
            dimensions

    Returns:
        tuple: `train_generator`, `val_generator`.
    """
    print('Full volume patchmul:', patch_multiplicity)
    return (self.get_train_generator(patch_shape=None,
                                     max_queue_size=2,
                                     pool_size=1,
                                     pool_refresh_period=1,
                                     transformations=Transformations.NONE,
                                     patch_multiplicity=patch_multiplicity),
            self.get_val_generator(patch_multiplicity=patch_multiplicity))


class NumpyDataset(Dataset):
  """ Dataset preprocessed by preprocess_dataset function. """

  def load_path(self, path):
    img = np.load(path)
    data = img['data']
    seg = img['seg']
    return data, seg

  def _get_paths(self, root_path):
    return glob(root_path + '*.npz')

# class NumpyDataset(Dataset):
#   """ Dataset preprocessed by preprocess_dataset function. """

#   def load_path(self, path):
#     img = np.load(path)
#     data = img[0]
#     seg = img[1]
#     return data.reshape(*data.shape, 1), seg.reshape(*seg.shape, 1)

#   def _get_paths(self, root_path):
#     return glob(root_path + '*.npy')


class ATLAS(NumpyDataset):
  """ Anatomical Tracing of Lesions After Stroke (ATLAS) dataset wrapper.
  TODO: unclear what the segmentation values mean.
  """

  def __init__(self, root_path='../../data/preprocessed_datasets/atlas/',
               validation_portion=.2):
    super(ATLAS, self).__init__(root_path, validation_portion)

  n_classes = 2

  name = 'atlas'


class BraTS(NumpyDataset):
  """ MICCAI's Multimodal Brain Tumor Segmentation Challenge 2017 dataset.
  Segmentation labels:
    1 for necrosis
    2 for edema
    3 for non-enhancing tumor
    4 for enhancing tumor
    0 for everything else
  """

  def __init__(self, root_path='../../data/preprocessed_datasets/brats/',
               validation_portion=.2):
    super(BraTS, self).__init__(root_path, validation_portion)

  n_classes = 5

  name = 'brats'


class MRBrainS(NumpyDataset):
  """ MRBrainS13 brain image segmentation challenge (anatomical)

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everyting else
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/mrbrains/',
               validation_portion=.2):
    super(MRBrainS, self).__init__(root_path, validation_portion)

  n_classes = 4

  name = 'mrbrains'


class IBSR(NumpyDataset):
  """ IBSR anatomical dataset (v2.0)

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everyting else
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/ibsr/',
               validation_portion=.2):
    super(IBSR, self).__init__(root_path, validation_portion)

  n_classes = 4

  name = 'ibsr'


class NiftiDataset(Dataset):
  """ Dataset that loads files of NIFTI (.nii) format """

  def load_path(self, path):
    data_path = self._get_data_path(path)
    data = nib.load(data_path).get_data().astype('float32')
    data = data.reshape(data.shape + (1,))

    seg_paths = self._get_seg_paths(path)
    seg = (sum(nib.load(path).get_data() for path in seg_paths) != 0).astype(
                                                                        'int8')
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_data_path(self, path):
    """ Given a meaningful path to a sample, get the exact path to the data. """
    raise NotImplementedError()

  def _get_seg_paths(self, path):
    """ Given a meaningful path to a sample, get the exact path to the
    segmentation. """
    raise NotImplementedError()


class RawATLAS(NiftiDataset):
  """ Anatomical Tracing of Lesions After Stroke (ATLAS) dataset wrapper.
  TODO: unclear what the segmentation values mean.
  """

  def __init__(self, root_path='../../data/ATLAS_R1.1/',
               validation_portion=.2):
    super(RawATLAS, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    data = normalize(resample_to_1mm(nib.load(data_path)).get_fdata())
    data = data.reshape(data.shape + (1,))

    seg_paths = self._get_seg_paths(path)
    seg = (sum(resample_to_1mm(nib.load(path), interpolation='nearest').
               get_data() for path in seg_paths) != 0).astype('int8')
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_data_path(self, path):
    data_path = glob(path + '/*deface_stx_robex*')
    assert(len(data_path) == 1)  # There should be exactly one deface per dir
    data_path = data_path[0]
    return data_path

  def _get_seg_paths(self, path):
    return glob(path + '/*LesionSmooth*')

  def _get_paths(self, root_path):
    return glob(root_path + '*/*/*')

  n_classes = 2

  name = 'atlas'


class RawBraTS(NiftiDataset):
  """ MICCAI's Multimodal Brain Tumor Segmentation Challenge 2017 dataset.
  Segmentation labels:
    1 for necrosis
    2 for edema
    3 for non-enhancing tumor
    4 for enhancing tumor
    0 for everything else
  """
  def __init__(self, root_path='../../data/MICCAI_BraTS17_Data_Training/',
               validation_portion=.2):
    super(RawBraTS, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_paths = self._get_data_path(path)
    data = [normalize(resample_to_1mm(nib.load(path)
                                      ).get_fdata().astype('float32'))
                                      for path in data_paths]
    data = np.stack(data, axis=-1)

    seg_path = self._get_seg_paths(path)
    assert(len(seg_path) == 1)
    seg_path = seg_path[0]
    seg = resample_to_1mm(nib.load(seg_path),
                          interpolation='nearest').get_data()
    seg = seg.reshape(seg.shape + (1,)).astype('int8')
    return (data, seg)

  def _get_data_path(self, path):
    data_path = [glob(path + '/*t1.nii.gz')[0],
                 glob(path + '/*t1ce.nii.gz')[0],
                 glob(path + '/*t2.nii.gz')[0],
                 glob(path + '/*flair.nii.gz')[0]]
    return data_path

  def _get_seg_paths(self, path):
    return glob(path + '/*seg*')

  def _get_paths(self, root_path):
    # This merges Higher Grade Glioma (HGG) with Lower Grade Glioma (LGG).
    # That might not be ideal.
    return glob(root_path + '*/*')

  n_classes = 5

  name = 'brats'


class RawMRBrainS(NiftiDataset):
  """ MRBrainS13 brain image segmentation challenge (anatomical).

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everyting else
  """
  def __init__(self, root_path='../../data/MRBrainS13DataNii/',
               validation_portion=.2):
    super(RawMRBrainS, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    data = normalize(resample_to_1mm(nib.load(data_path)).get_fdata().astype(
                                                                    'float32'))
    data = data.reshape(data.shape + (1,))

    seg_path = self._get_seg_paths(path)
    assert(len(seg_path) == 1)
    seg_path = seg_path[0]
    seg = resample_to_1mm(nib.load(seg_path), interpolation='nearest'
                          ).get_data().astype('int8')
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_data_path(self, path):
    data_path = glob(path + '/T1.nii')[0]
    return data_path

  def _get_seg_paths(self, path):
    # LabelsForTraining.nii contains addidional labels, LabelsForTesting.nii
    # only the ones mentioned above
    return glob(path + '/LabelsForTesting.nii')

  def _get_paths(self, root_path):
    return glob(root_path + 'TrainingData/*')

  n_classes = 4

  name = 'mrbrains'


class RawIBSR(NiftiDataset):
  """Internet Brain Segmentation Repository (IBSR) anatomical dataset v2.0."""

  def __init__(self, root_path='../../data/IBSR_nifti_stripped/',
               validation_portion=.2):
    super(RawIBSR, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    data = normalize(resample_to_1mm(nib.load(data_path[0])).get_fdata().astype(
                                                                    'float32'))

    data *= resample_to_1mm(nib.load(data_path[1]),
                            interpolation='nearest').get_fdata()
    seg_path = self._get_seg_paths(path)
    assert(len(seg_path) == 1)
    seg_path = seg_path[0]
    seg = resample_to_1mm(nib.load(seg_path),
                          interpolation='nearest').get_data().astype('int8')
    return (data, seg)

  def _get_data_path(self, path):
    # The first data path is the image, the second the brainmask.
    data_path = [glob(path + '/*ana_strip.nii.gz')[0],
                 glob(path + '/*brainmask.nii.gz')[0]]
    return data_path

  def _get_seg_paths(self, path):
    # LabelsForTraining.nii contains addidional labels, LabelsForTesting.nii
    # only the ones mentioned above
    return glob(path + '/*segTRI_fill_ana.nii.gz')

  def _get_paths(self, root_path):
    return glob(root_path + '/IBSR*')

  n_classes = 4

  name = 'ibsr'


if __name__ == '__main__':
  # preprocess_dataset(RawMRBrainS(), '../../data/preprocessed_datasets/mrbrains')
  preprocess_dataset(RawATLAS(), '../../data/preprocessed_datasets/atlas')
  # preprocess_dataset(RawBraTS(), '../../data/preprocessed_datasets/brats')
  # preprocess_dataset(RawIBSR(), '../../data/preprocessed_datasets/ibsr')
