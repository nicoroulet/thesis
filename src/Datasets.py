"""Collection of Brain datasets.

- MRBrainS (anatomical).
- IBSR (anatomical).
- BraTS (tumors).
- ATLAS (stroke lesion).
"""
from glob import glob
import itertools
import nibabel as nib
import numpy as np
from nilearn.image import resample_img

from BatchGenerator import BatchGenerator, Transformations
import Tools
import Logger


def normalize(X, roi=None):
  """Normalize a given image.

  Args:
      X (Numpy array): Input image

  Returns:
      Numpy array: normalized image

  """
  if roi is None:
  # X -= np.mean(X)
    X /= np.std(X)
  else:
    roi = roi.astype('bool')
    X[roi] -= np.mean(X[roi])
    X /= np.std(X[roi])
    X[np.logical_not(roi)] = Tools.bg_value
  return X


def resample_to_1mm(img, interpolation='continuous'):
  """Resample given image to have an identity affine (thus, a 1mm3 voxel size).

  Args:
      img (Nifty Image): Input image
      interpolation (string): resampling method. `continuous`, `nearest` or `linear`.

  Returns:
      Nifty Image: Resampled image.

  """
  return resample_img(img, target_affine=np.eye(3), interpolation=interpolation)


def resample_and_intersect(path1, path2, interpolation1='continuous', interpolation2='continuous'):
  """Take two images that resampled to 1mm3 have different sizes and crop the intersections."""
  img1 = resample_img(nib.load(path1), target_affine=np.eye(3), interpolation=interpolation1)
  img2 = resample_img(nib.load(path2), target_affine=np.eye(3), interpolation=interpolation2)

  x1, y1, z1 = map(int, img1.affine[:3,3])
  x2, y2, z2 = map(int, img2.affine[:3,3])

  img1_crop = img1.get_fdata()[max(0, x2-x1):, max(0, y2-y1):, max(0, z2-z1):]
  img2_crop = img2.get_fdata()[max(0, x1-x2):, max(0, y1-y2):, max(0, z1-z2):]

  try:
    img2_crop = np.squeeze(img2_crop, -1)
  except ValueError:
    pass
  try:
    img1_crop = np.squeeze(img1_crop, -1)
  except ValueError:
    pass
  w1, h1, d1 = img1_crop.shape
  w2, h2, d2 = img2_crop.shape
  w, h, d = min(w1, w2), min(h1, h2), min(d1, d2)

  img1_crop = img1_crop[:w, :h, :d]
  img2_crop = img2_crop[:w, :h, :d]

  return img1_crop, img2_crop


def load_resampled(path, **kwargs):
  """Load an image from path and resample it to 1mm.

  Args:
      path (string): the image path.
      kwargs (dict): arguments to pass to `resample_to_1mm`.

  """
  return resample_to_1mm(nib.load(path), **kwargs).get_fdata().astype('float32')



def preprocess_dataset(dataset, root_dir):
  """Take a dataset and stores it into memory as plain numpy arrays.

  Args:
      dataset: input dataset
      root_dir (string): directory to store dataset
  """
  root_dir += dataset.name + '/'
  import os

  Tools.ensure_dir(savedir)

  # Shuffle paths in order to guarrantee that training/validation portions are
  # random and not determined by some arbitrary property like file names.
  # Note that when loading the dataset, the partition will not be random, only
  # when preprocessing it.
  paths = dataset.train_paths + dataset.val_paths
  paths.sort()
  np.random.seed(123)
  np.random.shuffle(paths)

  Logger.info('Preprocessing dataset %s' % dataset.name)
  Logger.debug(paths)
  for i, path in enumerate(paths):
    save_path = root_dir + '/%d' % i
    Logger.info('Processing path: %s' % path)
    data, seg = dataset.load_path(path)

    if data.shape[:-1] != seg.shape[:-1]:
      raise ValueError('Data and Segmentation have incompatible shapes %s and %s' %
                       (str(data.shape), str(seg.shape)))
    np.savez(save_path,
             data=data,
             seg=seg)


class Dataset:
  """Abstract Dataset class.

  Requires the subclass to implement the load_path and _get_paths methods.
  """

  def __init__(self, root_path, validation_portion=.2):
    self.root_path = root_path
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
                          batch_size=5,
                          sample_bg=False):
    return BatchGenerator(patch_shape,
                          self.train_paths,
                          self.load_path,
                          max_queue_size=max_queue_size,
                          pool_size=pool_size,
                          pool_refresh_period=pool_refresh_period,
                          transformations=transformations,
                          patch_multiplicity=patch_multiplicity,
                          batch_size=batch_size,
                          sample_bg=sample_bg)

  def get_val_generator(self,
                        patch_shape=None,
                        max_queue_size=1,
                        pool_size=1,
                        pool_refresh_period=1,
                        transformations=Transformations.NONE,
                        patch_multiplicity=1,
                        batch_size=1,
                        infinite=True,
                        sample_bg=False):
    return BatchGenerator(patch_shape,
                          self.val_paths,
                          self.load_path,
                          max_queue_size=max_queue_size,
                          pool_size=pool_size,
                          pool_refresh_period=pool_refresh_period,
                          transformations=transformations,
                          patch_multiplicity=patch_multiplicity,
                          batch_size=batch_size,
                          infinite=infinite,
                          sample_bg=sample_bg)

  def get_patch_generators(self, patch_shape, batch_size=5, sample_train_bg=True):
    """Get both training generator and validation patched batch generators.

    Both crop patches from the images. The training generator also applies
    augmentation (gaussian noise and random flipping).

    Args:
        patch_shape: dimensions of the training patches.

    Returns:
        tuple: `train_generator`, `val_generator`.
    """
    return (self.get_train_generator(patch_shape=patch_shape,
                                     batch_size=batch_size,
                                     sample_bg=sample_train_bg),
            self.get_val_generator(patch_shape=(128, 128, 128),
                                   max_queue_size=3,
                                   pool_size=5,
                                   pool_refresh_period=5,
                                   transformations=Transformations.CROP,
                                   batch_size=1,
                                   sample_bg=False))

  def get_full_volume_generators(self, patch_multiplicity=1, infinite=True):
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
            self.get_val_generator(patch_multiplicity=patch_multiplicity, infinite=infinite))

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
    paths = glob(root_path + '*.npz')
    if not paths:
      Logger.error('dataset %s is empty' % self.name)
    return paths


class DummyDataset(NumpyDataset):
  def __init__(self, root_path='../../data/preprocessed_datasets/mrbrains13/',
                 validation_portion=.5):
    super(DummyDataset, self).__init__(root_path, validation_portion)

  def _get_paths(self, root_path):
    return glob(root_path + '0.npz') + glob(root_path + '1.npz')

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'ir', 'flair']

  name = 'dummy'


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


class BraTS12(NumpyDataset):
  """ MICCAI's Multimodal Brain Tumor Segmentation Challenge 2017 dataset.
  Segmentation labels:
    0 for background
    1 for edema
    2 for tumor
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/brats12/',
               validation_portion=.2):
    super(BraTS12, self).__init__(root_path, validation_portion)

  classes = ['background', 'edema', 'tumor']

  modalities = ['t1', 't1c', 't2', 'flair']

  name = 'brats12'


class BraTS17(NumpyDataset):
  """ MICCAI's Multimodal Brain Tumor Segmentation Challenge 2017 dataset.
  Segmentation labels:
    1 for necrosis
    2 for edema
    3 for non-enhancing tumor
    4 for enhancing tumor
    0 for everything else
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/brats17/',
               validation_portion=.2):
    super(BraTS17, self).__init__(root_path, validation_portion)

  classes = ['background', 'necrosis', 'edema', 'non-enhancing tumor', 'enhancing tumor']

  modalities = ['t1', 't1c', 't2', 'flair']

  name = 'brats17'

class BrainWeb(NumpyDataset):
  """BrainWeb synthetic anatomic dataset (last 15 subjects).

  Labels:
      0 Background.
      1 CSF.
      2 Gray Matter.
      3 White Matter.
      4 Vessels.

  """
  def __init__(self, root_path='../../data/preprocessed_datasets/brainweb/',
               validation_portion=.2):
    super(BrainWeb, self).__init__(root_path, validation_portion)

  classes = ['background', 'csf', 'gray matter', 'white matter']#, 'vessels']

  modalities = ['t1']

  name = 'brainweb'

class TumorSim(NumpyDataset):
  """BrainWeb synthetic anatomic dataset (last 15 subjects).

  Labels:
      0 Background.
      1 Edema
      2 Tumor
      3 CSF.
      4 Gray Matter.
      5 White Matter.
      6 Vessels.

  """
  def __init__(self, root_path='../../data/preprocessed_datasets/tumorsim_noisy/',
               validation_portion=.2):
    super(TumorSim, self).__init__(root_path, validation_portion)

  classes = ['background', 'edema', 'tumor', 'csf', 'gray matter', 'white matter']#s, 'vessels']

  modalities = ['t1', 't1c', 't2', 'flair']

  name = 'tumorsim'


class MRBrainS13(NumpyDataset):
  """ MRBrainS13 brain image segmentation challenge (anatomical)

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everything else
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/smooth/mrbrains13/',
               validation_portion=.2):
    super(MRBrainS13, self).__init__(root_path, validation_portion)

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'ir', 'flair']

  name = 'mrbrains13_smooth'


class IBSR(NumpyDataset):
  """ IBSR anatomical dataset (v2.0)

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everything else
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/smooth/ibsr/',
               validation_portion=.2):
    super(IBSR, self).__init__(root_path, validation_portion)

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1']

  name = 'ibsr_smooth'


class MRBrainS17(NumpyDataset):
  """MICCAI's MRBrainS17 White Matter Hyperintensities challenge dataset.

  Labels:
      0 Background.
      1 White Matter Hyperintensities (WMH).
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/mrbrains17/',
               validation_portion=.2):
    super(MRBrainS17, self).__init__(root_path, validation_portion)

  classes = ['background', 'wmh']

  modalities = ['t1', 'flair']

  name = 'mrbrains17'


class MRBrainS18(NumpyDataset):
  """MICCAI's MRBrainS18 Anatomical and White Matter Hyperintensities challenge dataset.

  The original labels are 0 - Background, 1 - Cortical gray matter, 2 - Basal ganglia,
  3 - White matter, 4 - White matter lesions, 5 - Cerebrospinal fluid in the extracerebral space,
  6 - Ventricles, 7 - Cerebellum, 8 - Brain stem, 9 - Infarction, 10 - Other, but have been reduced
  to:
      0 Background.
      1 White Matter Hyperintensities (WMH).
      2 CSF.
      3 Gray Matter.
      4 White Matter.
      -1 Ignore.
  'Ignore' groups Cerebellum, Brain Stem, Infarction and Other.
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/smooth/mrbrains18/',
               validation_portion=.2):
    super(MRBrainS18, self).__init__(root_path, validation_portion)

  classes = ['background', 'wmh', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'flair', 'ir']

  name = 'mrbrains18_smooth'


# -------------------------------- Raw Datasets ----------------------------------------------------


class RawATLAS(Dataset):
  """Anatomical Tracing of Lesions After Stroke (ATLAS) dataset wrapper.

  Segmentation labels:
    0 Background
    1 Stroke lesion
  """
  def __init__(self, root_path='../../data/ATLAS_R1.1/',
               validation_portion=.2):
    super(RawATLAS, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    data = resample_to_1mm(nib.load(data_path[0])).get_fdata()
    brainmask = resample_to_1mm(nib.load(data_path[1]), interpolation='nearest').get_fdata()
    data = normalize(data, roi=brainmask)
    data = data.reshape(data.shape + (1,))

    seg_paths = self._get_seg_paths(path)
    seg = (sum(resample_to_1mm(nib.load(path), interpolation='nearest').
               get_data() for path in seg_paths) != 0).astype('int8')
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_data_path(self, path):
    data_path = [glob(path + '/*deface_stx_robex*')[0],
                 glob(path + '/*brainmask*')[0]]
    return data_path

  def _get_seg_paths(self, path):
    return glob(path + '/*LesionSmooth*')

  def _get_paths(self, root_path):
    return glob(root_path + 'Site*/*/*')

  classes = ['background', 'stroke']

  modalities = ['t1']

  name = 'atlas'


class RawBraTS12(Dataset):
  """ MICCAI's Multimodal Brain Tumor Segmentation Challenge 2017 dataset.
  Segmentation labels:
    0 for background
    1 for edema
    2 for tumor
  """
  def __init__(self, root_path='../../data/BRATS2012/',
               validation_portion=.2):
    super(RawBraTS12, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_paths = self._get_data_path(path)
    t1 = load_resampled(data_paths['t1'])
    t1c = load_resampled(data_paths['t1c'])
    t2 = load_resampled(data_paths['t2'])
    flair = load_resampled(data_paths['flair'])
    brainmask = load_resampled(data_paths['t1'], interpolation='nearest') != 0
    data = np.stack([normalize(datum, roi=brainmask)
                     for datum in [t1, t1c, t2, flair]], axis=-1)

    seg_path, = self._get_seg_paths(path)
    seg = resample_to_1mm(nib.load(seg_path),
                          interpolation='nearest').get_data()
    seg *= brainmask
    seg = seg.reshape(seg.shape + (1,)).astype('int8')
    assert(seg.size)
    return (data, seg)

  def _get_data_path(self, path):
    data_path = {'t1': glob(path + '/*T1/*.nii.gz')[0],
                 't1c': glob(path + '/*T1c/*.nii.gz')[0],
                 't2': glob(path + '/*T2/*.nii.gz')[0],
                 'flair': glob(path + '/*Flair/*.nii.gz')[0]}
    return data_path

  def _get_seg_paths(self, path):
    return glob(path + '/*1more*/*.nii.gz')

  def _get_paths(self, root_path):
    # This merges Higher Grade Glioma (HGG) with Lower Grade Glioma (LGG).
    # That might not be ideal.
    return glob(root_path + 'Synthetic_Data/*/*')

  classes = ['background', 'edema', 'tumor']

  modalities = ['t1', 't1c', 't2', 'flair']

  name = 'brats12'

class RawBraTS17(Dataset):
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
    super(RawBraTS17, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_paths = self._get_data_path(path)
    data = [resample_to_1mm(nib.load(path)).get_fdata().astype('float32') for path in data_paths]
    data = [normalize(datum, roi=(np.abs(datum) > 0.0001)) for datum in data]
    data = np.stack(data, axis=-1)

    seg_path, = self._get_seg_paths(path)
    seg = resample_to_1mm(nib.load(seg_path),
                          interpolation='nearest').get_data()
    seg *= brainmask
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

  modalities = ['t1', 't1c', 't2', 'flair']

  name = 'brats17'


class RawMRBrainS13(Dataset):
  """ MRBrainS13 brain image segmentation challenge (anatomical).

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everything else
  """
  def __init__(self, root_path='../../data/MRBrainS13DataNii/',
               validation_portion=.2):
    super(RawMRBrainS13, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    subpaths = self._get_subpaths(path)
    t1, seg = resample_and_intersect(subpaths['t1'], subpaths['seg'], interpolation2='nearest')
    ir, seg = resample_and_intersect(subpaths['ir'], subpaths['seg'], interpolation2='nearest')
    flair, seg = resample_and_intersect(subpaths['flair'], subpaths['seg'], interpolation2='nearest')
    brainmask, seg = resample_and_intersect(subpaths['brainmask'], subpaths['seg'], interpolation2='nearest')
    brainmask = brainmask.astype('int8')
    seg = seg.astype('int8')

    # t1 = resample_to_1mm(nib.load(data_path[0])).get_fdata()
    # ir = resample_to_1mm(nib.load(data_path[1])).get_fdata()
    # flair = resample_to_1mm(nib.load(data_path[2])).get_fdata()
    # brainmask = resample_to_1mm(nib.load(data_path[3]), interpolation='nearest').get_fdata().astype('int8')

    data = np.stack([normalize(t1, roi=brainmask),
                     normalize(ir, roi=brainmask),
                     normalize(flair, roi=brainmask)], axis=-1).astype('float32')

    # seg_path, = self._get_seg_paths(path)
    # seg = nib.load(seg_path).get_data().astype('int8')
    # seg = resample_to_1mm(nib.load(seg_path), interpolation='nearest').get_data().astype('int8')
    seg *= brainmask
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_subpaths(self, path):
    subpaths = {}
    subpaths['t1'], = glob(path + '/T1.nii')
    subpaths['ir'], = glob(path + '/T1_IR.nii')
    subpaths['flair'], = glob(path + '/T2_FLAIR.nii')
    subpaths['brainmask'], = glob(path + '/brainmask.nii.gz')
    subpaths['seg'], = glob(path + '/LabelsForTesting_preproc.nii')
    return subpaths

  def _get_seg_paths(self, path):
    # LabelsForTraining.nii contains addidional labels, LabelsForTesting.nii
    # only the ones mentioned above
    return glob(path + '/LabelsForTesting.nii')

  def _get_paths(self, root_path):
    return glob(root_path + 'TrainingData/*')

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'ir', 'flair']

  name = 'mrbrains13_smooth'


class RawIBSR(Dataset):
  """ IBSR anatomical dataset (v2.0)

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everything else
  """

  def __init__(self, root_path='../../data/IBSR_nifti_stripped/',
               validation_portion=.2):
    super(RawIBSR, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    subpaths = self._get_subpaths(path)

    t1, seg = resample_and_intersect(subpaths['t1'], subpaths['seg'], interpolation2='nearest')
    brainmask, seg = resample_and_intersect(subpaths['brainmask'], subpaths['seg'], interpolation2='nearest')
    brainmask = np.round(brainmask).astype('int8')
    seg = seg.astype('int8')
    # t1 = resample_to_1mm(nib.load(subpaths['t1'])).get_fdata().astype('float32')

    # brainmask = resample_to_1mm(nib.load(subpaths['brainmask']), interpolation='nearest').get_fdata().astype('int8')
    data = np.stack([normalize(t1, roi=brainmask)], axis=-1)

    # seg_path, = subpaths['seg']
    # seg = resample_to_1mm(nib.load(seg_path), interpolation='nearest').get_fdata().astype('int8')
    seg *= brainmask
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_subpaths(self, path):
    # The first data path is the image, the second the brainmask.
    subpaths = {}
    subpaths['t1'], = glob(path + '/*ana_strip.nii.gz')
    subpaths['brainmask'], = glob(path + '/*brainmask.nii.gz')
    subpaths['seg'], = glob(path + '/*segTRI_fill_ana_preproc.nii.gz')
    # data_path = [glob(path + '/*ana_strip.nii.gz')[0],
    #              glob(path + '/*brainmask.nii.gz')[0]]
    return subpaths

  def _get_seg_paths(self, path):
    return glob(path + '/*segTRI_fill_ana_preproc.nii.gz')

  def _get_paths(self, root_path):
    return glob(root_path + '/IBSR*')

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1']

  name = 'ibsr_smooth'


class RawMRBrainS17(Dataset):
  """MICCAI's MRBrainS17 White Matter Hyperintensities challenge dataset.

  Labels:
      0 Background.
      1 White Matter Hyperintensities (WMH).
      2 Other pathology --> This label is merged with background and is not present.
  """

  def __init__(self, root_path='../../data/WMH_MICCAI17/',
               validation_portion=.2):
    super(RawMRBrainS17, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    t1 = resample_to_1mm(nib.load(data_path[1])).get_fdata().astype('float32')
    flair = resample_to_1mm(nib.load(data_path[2])).get_fdata().astype('float32')

    brainmask = resample_to_1mm(nib.load(data_path[0]), interpolation='nearest').get_fdata().astype('int8')

    data = np.stack([normalize(t1, roi=brainmask),
                     normalize(flair, roi=brainmask)], axis=-1)

    seg_path = self._get_seg_path(path)
    seg = nib.load(seg_path)
    seg.affine[np.abs(seg.affine) < .000001] = 0
    seg = resample_to_1mm(seg, interpolation='nearest').get_data().astype('int8')
    # Merge label 'other pathology' with background.
    seg[seg == 2] = 0
    seg *= brainmask
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

  classes = ['background', 'wmh']

  modalities = ['t1', 'flair']

  name = 'mrbrains17_smooth'


class RawMRBrainS18(Dataset):
  """MICCAI's MRBrainS18 Anatomical and White Matter Hyperintensities challenge dataset.

  The original labels are 0 - Background, 1 - Cortical gray matter, 2 - Basal ganglia,
  3 - White matter, 4 - White matter lesions, 5 - Cerebrospinal fluid in the extracerebral space,
  6 - Ventricles, 7 - Cerebellum, 8 - Brain stem, 9 - Infarction, 10 - Other, but have been reduced
  to:
      0 Background.
      1 White Matter Hyperintensities (WMH).
      2 CSF.
      3 Gray Matter.
      4 White Matter.
      -1 Ignore.
  'Ignore' groups Cerebellum, Brain Stem, Infarction and Other.
  """
  def __init__(self, root_path='../../data/MRBrainS18/',
               validation_portion=.2):
    super(RawMRBrainS18, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    subpaths = self._get_subpaths(path)
    t1, seg = resample_and_intersect(subpaths['t1'], subpaths['seg'], interpolation2='nearest')
    ir, seg = resample_and_intersect(subpaths['ir'], subpaths['seg'], interpolation2='nearest')
    flair, seg = resample_and_intersect(subpaths['flair'], subpaths['seg'], interpolation2='nearest')
    # t1 = resample_to_1mm(nib.load(data_paths['t1'])).get_fdata().astype('float32')
    # ir = resample_to_1mm(nib.load(data_paths['ir'])).get_fdata().astype('float32')
    # flair = resample_to_1mm(nib.load(data_paths['flair'])).get_fdata().astype('float32')

    brainmask, seg = resample_and_intersect(subpaths['brainmask'], subpaths['seg'], interpolation2='nearest')
    # brainmask = resample_to_1mm(nib.load(data_paths['brainmask']), interpolation='nearest').get_fdata().astype('int8')
    brainmask = brainmask.astype('int8')
    seg = seg.astype('int8')
    data = np.stack([normalize(t1, roi=brainmask),
                     normalize(flair, roi=brainmask),
                     normalize(ir, roi=brainmask)],
                     axis=-1)

    # seg = nib.load(seg_path)
    # seg.affine[np.abs(seg.affine) < .000001] = 0
    # seg = resample_to_1mm(seg, interpolation='nearest').get_data().astype('int8')
    # Unify 1 and 2 as gray matter; 5 and 6 as CSF; 7, 8, 9, 10 as Ignore.
    # Note: this sends Cerebellum, Brain Stem, Infarction and Other to -1. This makes the
    # dataset unfit for training, but it's fine for validation with Dice, because those sections
    # will be ignored.
    # 1, 2 -> 3
    # 3 -> 4
    # 4 -> 1
    # 5, 6 -> 2
    # 7, 8, 9, 10 -> -1
    seg[seg >= 7] = -1
    seg[seg == 1] = 2
    seg[seg == 4] = 1
    seg[seg == 3] = 4
    seg[seg == 2] = 3
    seg[seg >= 5] = 2
    seg *= brainmask
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_subpaths(self, path):
    subpaths = {}

    subpaths['brainmask'], = glob(path + '/pre/reg_brainmask.nii.gz')
    subpaths['t1'], = glob(path + '/pre/reg_T1.nii.gz')
    subpaths['ir'], = glob(path + '/pre/reg_IR.nii.gz')
    subpaths['flair'], = glob(path + '/pre/FLAIR.nii.gz')
    subpaths['seg'], = glob(path + '/segm_preproc.nii.gz')
    return subpaths

  def _get_paths(self, root_path):
    return glob(root_path + '/*/')

  classes = ['background', 'wmh', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'flair', 'ir']

  name = 'mrbrains18_smooth'


class RawBrainWeb(Dataset):
  """BrainWeb synthetic anatomic dataset (last 15 subjects).

  Labels:
      0 Background.
      1 CSF.
      2 Gray Matter.
      3 White Matter.
      4 Vessels.
  """
  def __init__(self, root_path='../../data/BrainWeb/',
               validation_portion=.2):
    super(RawBrainWeb, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    subpaths = self._get_subpaths(path)
    t1, seg = resample_and_intersect(subpaths['t1'], subpaths['seg'], interpolation2='nearest')
    _, brainmask = resample_and_intersect(subpaths['seg'], subpaths['brainmask'])
    brainmask = brainmask.astype('int8')
    seg = seg.astype('int8')
    # t1 = load_resampled(subpaths['t1'])
    # brainmask = load_resampled(subpaths['brainmask'], interpolation='nearest')

    data = np.stack([normalize(t1, roi=brainmask)], axis=-1)

    seg = (seg * brainmask).reshape(seg.shape + (1,))
    # _, seg = intersect_and_resample(subpaths['t1'], subpaths['seg'])
    # seg = load_resampled(subpaths['seg'], interpolation='nearest').astype('int8')
    # Original labels are (in order): Background, CSF, Grey Matter, White Matter, Fat, Muscle,
    # Muscle / Skin, Skull, Vessels, Connective, Dura, Marrow.
    # Wanted labels are: Background, CSF, Grey Matter, White Matter, Vessels
    seg[seg == 4] = 5
    seg[seg == 8] = 1
    seg[seg > 4] = 0
    return (data, seg)

  def _get_subpaths(self, path):
    (t1,) = glob(path + '/*t1w*.mnc.gz')
    (brainmask,) = glob(path + '/brainmask.nii.gz')
    (seg,) = glob(path + '/*crisp*.mnc.gz')
    return {'t1': t1,
            'brainmask': brainmask,
            'seg': seg}

  def _get_paths(self, root_path):
    return glob(root_path + '/*/')

  classes = ['background', 'csf', 'gray matter', 'white matter']#, 'vessels']

  modalities = ['t1']

  name = 'brainweb'


class RawTumorSim(Dataset):
  """TumorSim simulations on first 5 BrainWeb subjects.

  Labels:
      0 Background.
      1 Edema
      2 Tumor
      3 CSF.
      4 Gray Matter.
      5 White Matter.
      6 Vessels.

  """
  def __init__(self, root_path='../../data/TumorSimLowNoise/',
               validation_portion=.2):
    super(RawTumorSim, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    subpaths = self._get_subpaths(path)
    t1 = load_resampled(subpaths['t1'])
    t1c = load_resampled(subpaths['t1c'])
    t2 = load_resampled(subpaths['t2'])
    flair = load_resampled(subpaths['flair'])
    brainmask = load_resampled(subpaths['t1'], interpolation='nearest') > 20
    data = np.stack([normalize(datum, roi=brainmask) # + np.random.normal(0, 0.2, datum.shape) * brainmask
                     for datum in [t1, t1c, t2, flair]], axis=-1)

    seg = load_resampled(subpaths['seg'], interpolation='nearest') * brainmask
    assert(seg.size)

    # Original labels are (in order): Background, White Matter, Grey Matter, CSF, Edema, Tumor,
    # Vessels.
    # Wanted labels are: Background, Edema, Tumor, CSF, Grey Matter, White Matter, Vessels
    mapping = {0: 0, 1: 5, 2: 4, 3: 3, 4: 1, 5: 2, 6: 3}
    mapper = np.vectorize(lambda x: mapping[x])
    seg = mapper(seg).astype('int8')
    # print('seg values: ', set(seg.flat))
    # seg[seg == 6] = 7
    # seg[seg == 5] = 6
    # seg[seg == 4] = 5
    # seg[seg == 7] = 4
    seg *= brainmask
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_subpaths(self, path):
    print(path)
    (t1,) = glob(path + '/*T1.nii.gz')
    (t1c,) = glob(path + '/*T1Gad.nii.gz')
    (t2,) = glob(path + '/*T2.nii.gz')
    (flair,) = glob(path + '/*FLAIR.nii.gz')
    (seg,) = glob(path + '/*discrete_truth.nii.gz')
    return {'t1': t1,
            't1c': t1c,
            't2': t2,
            'flair': flair,
            'seg': seg}

  def _get_paths(self, root_path):
    return glob(root_path + '/*/')

  classes = ['background', 'edema', 'tumor', 'csf', 'gray matter', 'white matter']#, 'vessels']

  modalities = ['t1', 't1c', 't2', 'flair']

  name = 'tumorsim'

class MultiDataset(NumpyDataset):
  """Combination of multiple datasets.

  Dataset paths are pooled together in equal proportions (paths from the smaller datasets are
  repeated to match the biggest dataset size.

  Datasets are expected to have disjoint labels, and label values are increased incrementally.

  Note: both train and validation parts of training datasets are used for training and both train
  and validation parts of validation dataset are used for validation.
  """

  def __init__(self, datasets, ignore_backgrounds=False, balance='datasets'):
    """Initialize MultiDataset.

    Args:
        datasets (list): List of training datasets.
        val_dataset (Dataset): Validation dataset
        ignore_backgrounds (bool, optional): if `True`, set backgrounds for all but the last dataset
            to -1. This can be combined with loss functions that ignore the -1 labels.
        balance: either `datasets` (default) or `labels`. `datasets` means `self.train_paths`
            will have equal number of paths from each dataset. `labels` means the number of paths
            from each dataset will be balanced according to the number of labels of the
            corresponding dataset (more labels, more paths of that dataset). This means
            patches will be sampled equiprobably between labels instead of datasets.

    """
    self.datasets = datasets
    self.name = '_'.join(dataset.name for dataset in datasets)
    # self.val_dataset = val_dataset
    self.ignore_backgrounds = ignore_backgrounds
    self.balance = balance
    # self.train_paths = [path for dataset in datasets for path in dataset.train_paths]
    # self.val_paths = [path for dataset in datasets for path in dataset.val_paths]
    self.train_paths, self.val_paths = self._aggregate_paths()
    #                                        [dataset.train_paths + dataset.val_paths
    #                                           for dataset in datasets])
    np.random.seed(123)
    np.random.shuffle(self.train_paths)
    # self.val_paths = val_dataset.train_paths + val_dataset.val_paths
    np.random.seed(321)
    np.random.shuffle(self.val_paths)
    self.modalities = list(set.intersection(*[set(dataset.modalities) for dataset in datasets]))
    assert(self.modalities), "Given datasets don't have intersecting modalities"
    self.classes = ['background'] + [clss for dataset in datasets for clss in dataset.classes[1:]]
    # if balance != 'datasets':
    #   self.name += '_' + balance

  def load_path(self, path):
    """Load a given dataset path.

    This handles autoincrementing the labels between datasets. If `self.ignore_backgrounds`,
    this is also handled here.

    Args:
        path (string): path to a dataset element.

    Returns:
        tuple: (x, y) image, segmentation.

    """
    # print(path)
    x, y = super(MultiDataset, self).load_path(path)
    # if path.startswith(self.val_dataset.root_path):
    #   x_filtered = Tools.filter_modalities(self.val_dataset.modalities, self.modalities, x)
    #   # print('Seg labels:', set(y.flat))
    #   return x_filtered, y
    label_offset = 0
    for dataset in self.datasets:
      # FIXME: this is ugly.
      if path.startswith(dataset.root_path):
        source_dataset = dataset
        break
      label_offset += dataset.n_classes - 1
    x_filtered = Tools.filter_modalities(source_dataset.modalities, self.modalities, x)
    y[y > 0] += label_offset
    if self.ignore_backgrounds and source_dataset is not self.datasets[-1]:
      y[y == 0] = -1
    # print('Seg labels:', set(y.flat))
    return x_filtered, y

  def _aggregate_paths(self):
    """Combine paths with balanced amounts from each list of paths.

    Args:
        paths_list (list of lists): List of lists of paths to be aggregated.

    """
    aggregated_train_paths = []
    aggregated_val_paths = []
    if self.balance == 'datasets':
      max_n_images = max(len(dataset.train_paths) for dataset in self.datasets)
      for dataset in self.datasets:
        paths = dataset.train_paths
        aggregated_train_paths.extend(itertools.islice(itertools.cycle(paths), max_n_images))
      max_n_images = max(len(dataset.val_paths) for dataset in self.datasets)
      for dataset in self.datasets:
        paths = dataset.val_paths
        aggregated_val_paths.extend(itertools.islice(itertools.cycle(paths), max_n_images))
    elif self.balance == 'labels':  # warning: this is obsolete now
      # Dataset with highest images / n_classes ratio
      '''
      max_idx = np.argmax([dataset.n_images / dataset.n_classes for dataset in self.datasets])
      max_dataset = self.datasets[max_idx]
      for dataset in self.datasets:
        n_paths = (max_dataset.n_images * dataset.n_classes) // max_dataset.n_classes
        aggregated_paths.extend(itertools.islice(itertools.cycle(dataset.train_paths +
                                                                dataset.val_paths), n_paths))
      '''
    else:
      raise ValueError("balance attribute must be either 'datasets' or 'labels'")
    return aggregated_train_paths, aggregated_val_paths


if __name__ == '__main__':
  # preprocess_dataset(RawMRBrainS13(), '../../data/preprocessed_datasets/smooth/')
  # preprocess_dataset(RawIBSR(), '../../data/preprocessed_datasets/smooth/')
  # preprocess_dataset(RawMRBrainS17(), '../../data/preprocessed_datasets/')
  # preprocess_dataset(RawMRBrainS18(), '../../data/preprocessed_datasets/smooth/')
  # preprocess_dataset(RawBraTS12(), '../../data/preprocessed_datasets/')
  # preprocess_dataset(RawBrainWeb(), '../../data/preprocessed_datasets/')
  preprocess_dataset(RawTumorSim(), '../../data/preprocessed_datasets/')
