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

    if data.shape[:-1] != seg.shape[:-1]:
      raise ValueError('Data and Segmentation have incompatible shapes %s and %s' %
                       (str(data.shape), str(seg.shape)))
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
                          patch_multiplicity=1,
                          batch_size=5):
    return BatchGenerator(patch_shape,
                          self.train_paths,
                          self.load_path,
                          max_queue_size=max_queue_size,
                          pool_size=pool_size,
                          pool_refresh_period=pool_refresh_period,
                          transformations=transformations,
                          patch_multiplicity=patch_multiplicity,
                          batch_size=batch_size)

  def get_val_generator(self,
                        patch_shape=None,
                        max_queue_size=2,
                        pool_size=1,
                        pool_refresh_period=1,
                        transformations=Transformations.NONE,
                        patch_multiplicity=1,
                        batch_size=1):
    return BatchGenerator(patch_shape,
                          self.val_paths,
                          self.load_path,
                          max_queue_size=max_queue_size,
                          pool_size=pool_size,
                          pool_refresh_period=pool_refresh_period,
                          transformations=transformations,
                          patch_multiplicity=patch_multiplicity,
                          batch_size=batch_size)

  def get_patch_generators(self, patch_shape, batch_size=5):
    """Get both training generator and validation patched batch generators.

    Both crop patches from the images. The training generator also applies
    augmentation (gaussian noise and random flipping).

    Args:
        patch_shape: dimensions of the training patches.

    Returns:
        tuple: `train_generator`, `val_generator`.
    """
    return (self.get_train_generator(patch_shape=patch_shape,
                                     batch_size=batch_size),
            self.get_val_generator(patch_shape=patch_shape,
                                   max_queue_size=3,
                                   pool_size=5,
                                   pool_refresh_period=20,
                                   transformations=Transformations.CROP,
                                   batch_size=batch_size))

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

  @property
  def n_classes(self):
    """Get the number of classes (labels) on the dataset segmentation.

    Returns:
        int: number of classes.
    """
    return len(self.classes)

  @property
  def n_modalities(self):
    """Get the number of classes (labels) on the dataset segmentation.

    Returns:
        int: number of classes.
    """
    return len(self.modalities)


class NumpyDataset(Dataset):
  """ Dataset preprocessed by preprocess_dataset function. """

  def load_path(self, path):
    img = np.load(path)
    data = img['data']
    seg = img['seg']
    return data, seg

  def _get_paths(self, root_path):
    return glob(root_path + '*.npz')


class ATLAS(NumpyDataset):
  """ Anatomical Tracing of Lesions After Stroke (ATLAS) dataset wrapper.

  Segmentation labels:
    0 Background
    1 Stroke lesion
  """

  def __init__(self, root_path='../../data/preprocessed_datasets/atlas/',
               validation_portion=.2):
    super(ATLAS, self).__init__(root_path, validation_portion)

  classes = ['background', 'stroke']

  modalities = ['t1']

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

  classes = ['background', 'necrosis', 'edema', 'non-enhancing tumor', 'enhancing tumor']

  modalities = ['t1', 't1ce', 't2', 'flair']

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

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1']

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

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1']

  name = 'ibsr'


class WMH(NumpyDataset):
  """MICCAI's 2017 White Matter Hyperintensities challenge dataset.

  Labels:
      0 Background.
      1 White Matter Hyperintensities (WMH).
      2 Other pathology
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/wmh/',
               validation_portion=.2):
    super(WMH, self).__init__(root_path, validation_portion)

  classes = ['background', 'wmh', 'other pathology']

  modalities = ['t1', 'flair']

  name = 'wmh'


class RawATLAS(Dataset):
  """ Anatomical Tracing of Lesions After Stroke (ATLAS) dataset wrapper.

  Segmentation labels:
    0 Background
    1 Stroke lesion
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

  classes = ['background', 'stroke']

  modalities = ['t1']

  name = 'atlas'


class RawBraTS(Dataset):
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
    assert(seg.size)
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

  classes = ['background', 'necrosis', 'edema', 'non-enhancing tumor', 'enhancing tumor']

  modalities = ['t1', 't1ce', 't2', 'flair']

  name = 'brats'


class RawMRBrainS(Dataset):
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

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1']

  name = 'mrbrains'


class RawIBSR(Dataset):
  """ IBSR anatomical dataset (v2.0)

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everyting else
  """

  def __init__(self, root_path='../../data/IBSR_nifti_stripped/',
               validation_portion=.2):
    super(RawIBSR, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    data = resample_to_1mm(nib.load(data_path[0])).get_fdata().astype('float32')

    data *= resample_to_1mm(nib.load(data_path[1]), interpolation='nearest').get_fdata()
    data = normalize(data)

    seg_path = self._get_seg_paths(path)
    assert(len(seg_path) == 1)
    seg_path = seg_path[0]
    seg = resample_to_1mm(nib.load(seg_path), interpolation='nearest').get_data().astype('int8')
    return (data, seg)

  def _get_data_path(self, path):
    # The first data path is the image, the second the brainmask.
    data_path = [glob(path + '/*ana_strip.nii.gz')[0],
                 glob(path + '/*brainmask.nii.gz')[0]]
    return data_path

  def _get_seg_paths(self, path):
    return glob(path + '/*segTRI_fill_ana.nii.gz')

  def _get_paths(self, root_path):
    return glob(root_path + '/IBSR*')


  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1']

  name = 'ibsr'


class RawWMH(Dataset):
  """MICCAI's 2017 White Matter Hyperintensities challenge dataset.

  Labels:
      0 Background.
      1 White Matter Hyperintensities (WMH).
      2 Other pathology
  """

  def __init__(self, root_path='../../data/WMH_MICCAI17/',
               validation_portion=.2):
    super(RawWMH, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    T1 = resample_to_1mm(nib.load(data_path[1])).get_fdata().astype('float32')
    FLAIR = resample_to_1mm(nib.load(data_path[2])).get_fdata().astype('float32')

    brainmask = resample_to_1mm(nib.load(data_path[0]), interpolation='nearest').get_fdata()

    data = np.stack([normalize(T1 * brainmask), normalize(FLAIR * brainmask)], axis=-1)

    seg_path = self._get_seg_path(path)
    seg = nib.load(seg_path)
    seg.affine[np.abs(seg.affine) < .000001] = 0
    seg = resample_to_1mm(seg, interpolation='nearest').get_data().astype('int8')
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_data_path(self, path):
    data_path = [glob(path + '/pre/brainmask.nii.gz')[0],
                 glob(path + '/pre/T1.nii.gz')[0],
                 glob(path + '/pre/FLAIR.nii.gz')[0]]
    return data_path

  def _get_seg_path(self, path):
    return glob(path + '/wmh.nii.gz')[0]

  def _get_paths(self, root_path):
    return glob(root_path + '/*/*/')

  classes = ['background', 'wmh', 'other pathology']

  modalities = ['t1', 'flair']

  name = 'wmh'


if __name__ == '__main__':
  # preprocess_dataset(RawMRBrainS(), '../../data/preprocessed_datasets/mrbrains')
  # preprocess_dataset(RawATLAS(), '../../data/preprocessed_datasets/atlas')
  # preprocess_dataset(RawBraTS(), '../../data/preprocessed_datasets/brats')
  # preprocess_dataset(RawIBSR(), '../../data/preprocessed_datasets/ibsr')
  preprocess_dataset(RawWMH(), '../../data/preprocessed_datasets/wmh')
