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
      interpolation (string): resampling method. `continuous`, `nearest` or `linear`.

  Returns:
      Nifty Image: Resampled image.

  """
  return resample_img(img, target_affine=np.eye(3), interpolation=interpolation)


def preprocess_dataset(dataset, root_dir):
  """Take a dataset and stores it into memory as plain numpy arrays.

  Args:
      dataset: input dataste
      root_dir (string): directory to store dataset
  """
  root_dir += dataset.name + '/'
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
                        max_queue_size=4,
                        pool_size=1,
                        pool_refresh_period=1,
                        transformations=Transformations.NONE,
                        patch_multiplicity=1,
                        batch_size=1,
                        infinite=True):
    return BatchGenerator(patch_shape,
                          self.val_paths,
                          self.load_path,
                          max_queue_size=max_queue_size,
                          pool_size=pool_size,
                          pool_refresh_period=pool_refresh_period,
                          transformations=transformations,
                          patch_multiplicity=patch_multiplicity,
                          batch_size=batch_size,
                          infinite=infinite)

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
            self.get_val_generator(patch_shape=(128, 128, 128),
                                   max_queue_size=3,
                                   pool_size=5,
                                   pool_refresh_period=20,
                                   transformations=Transformations.CROP,
                                   batch_size=1))

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


class MRBrainS13(NumpyDataset):
  """ MRBrainS13 brain image segmentation challenge (anatomical)

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everyting else
  """
  def __init__(self, root_path='../../data/preprocessed_datasets/mrbrains13/',
               validation_portion=.2):
    super(MRBrainS13, self).__init__(root_path, validation_portion)

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'ir', 'flair']

  name = 'mrbrains13'


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
  def __init__(self, root_path='../../data/preprocessed_datasets/mrbrains18/',
               validation_portion=.2):
    super(MRBrainS18, self).__init__(root_path, validation_portion)

  classes = ['background', 'wmh', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'flair', 'ir']

  name = 'mrbrains18'


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


class RawMRBrainS13(Dataset):
  """ MRBrainS13 brain image segmentation challenge (anatomical).

  Labels:
    1 Cerebrospinal fluid (including ventricles)
    2 Gray matter (cortical gray matter and basal ganglia)
    3 White matter (including white matter lesions)
    0 Everyting else
  """
  def __init__(self, root_path='../../data/MRBrainS13DataNii/',
               validation_portion=.2):
    super(RawMRBrainS13, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_path = self._get_data_path(path)
    t1 = resample_to_1mm(nib.load(data_path[0])).get_fdata()
    ir = resample_to_1mm(nib.load(data_path[1])).get_fdata()
    flair = resample_to_1mm(nib.load(data_path[2])).get_fdata()
    brainmask = resample_to_1mm(nib.load(data_path[3]), interpolation='nearest').get_fdata()

    data = np.stack([normalize(t1 * brainmask),
                     normalize(ir * brainmask),
                     normalize(flair * brainmask)], axis=-1).astype('float32')

    seg_path = self._get_seg_paths(path)
    assert(len(seg_path) == 1)
    seg_path = seg_path[0]
    seg = resample_to_1mm(nib.load(seg_path), interpolation='nearest').get_data().astype('int8')
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

  def _get_data_path(self, path):
    data_path = [glob(path + '/T1.nii')[0],
                 glob(path + '/T1_IR.nii')[0],
                 glob(path + '/T2_FLAIR.nii')[0]
,                 glob(path + '/brainmask.nii.gz')[0]]
    return data_path

  def _get_seg_paths(self, path):
    # LabelsForTraining.nii contains addidional labels, LabelsForTesting.nii
    # only the ones mentioned above
    return glob(path + '/LabelsForTesting.nii')

  def _get_paths(self, root_path):
    return glob(root_path + 'TrainingData/*')

  classes = ['background', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'ir', 'flair']

  name = 'mrbrains13'


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
    T1 = resample_to_1mm(nib.load(data_path[1])).get_fdata().astype('float32')
    FLAIR = resample_to_1mm(nib.load(data_path[2])).get_fdata().astype('float32')

    brainmask = resample_to_1mm(nib.load(data_path[0]), interpolation='nearest').get_fdata()

    data = np.stack([normalize(T1 * brainmask), normalize(FLAIR * brainmask)], axis=-1)

    seg_path = self._get_seg_path(path)
    seg = nib.load(seg_path)
    seg.affine[np.abs(seg.affine) < .000001] = 0
    seg = resample_to_1mm(seg, interpolation='nearest').get_data().astype('int8')
    seg = seg.reshape(seg.shape + (1,))
    # Merge label 'other pathology' with background.
    seg[seg == 2] = 0
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

  name = 'mrbrains17'


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
  def __init__(self, root_path='../../data/MICCAI18_WMH/',
               validation_portion=.2):
    super(RawMRBrainS18, self).__init__(root_path, validation_portion)

  def load_path(self, path):
    data_paths = self._get_data_path(path)
    T1 = resample_to_1mm(nib.load(data_paths[1])).get_fdata().astype('float32')
    IR = resample_to_1mm(nib.load(data_paths[2])).get_fdata().astype('float32')
    FLAIR = resample_to_1mm(nib.load(data_paths[3])).get_fdata().astype('float32')

    brainmask = resample_to_1mm(nib.load(data_paths[0]), interpolation='nearest').get_fdata()

    data = np.stack([normalize(T1 * brainmask),
                     normalize(FLAIR * brainmask),
                     normalize(IR * brainmask)],
                     axis=-1)

    seg_path = self._get_seg_path(path)
    seg = nib.load(seg_path)
    seg.affine[np.abs(seg.affine) < .000001] = 0
    seg = resample_to_1mm(seg, interpolation='nearest').get_data().astype('int8')
    seg = seg.reshape(seg.shape + (1,))
    # Unify 1 and 2 as gray matter; 5 and 6 as CSF; 7, 8, 9, 10 as Ignore.
    # Note: this sends Cerebellum, Brain Stem, Infarction and Other to background. This makes the
    # dataset unfit for training, but it's fine for validation with Dice, because those sections
    # will be ignored.
    # 1, 2 -> 3
    # 3 -> 4
    # 4 -> 1
    # 5, 6 -> 2
    # 7, 8, 9, 10 -> 5
    seg[seg >= 7] = -1
    seg[seg == 1] = 2
    seg[seg == 4] = 1
    seg[seg == 3] = 4
    seg[seg == 2] = 3
    seg[seg >= 5] = 2
    return (data, seg)

  def _get_data_path(self, path):
    data_path = [glob(path + '/pre/reg_brainmask2.nii.gz')[0],
                 glob(path + '/pre/reg_T1.nii.gz')[0],
                 glob(path + '/pre/reg_IR.nii.gz')[0],
                 glob(path + '/pre/FLAIR.nii.gz')[0]]
    return data_path

  def _get_seg_path(self, path):
    return glob(path + '/segm.nii.gz')[0]

  def _get_paths(self, root_path):
    return glob(root_path + '/*/')

  classes = ['background', 'wmh', 'csf', 'gray matter', 'white matter']

  modalities = ['t1', 'flair', 'ir']

  name = 'mrbrains18'


class MultiDataset(NumpyDataset):
  """Combination of multiple datasets.

  Dataset paths are pooled together in equal proportions (paths from the smaller datasets are
  repeated to match the biggest dataset size.
  Datasets are expected to have disjoint labels, and label values are increased incrementally.
  """

  def __init__(self, datasets, val_dataset, ignore_backgrounds=False):
    """Initialize MultiDataset.

    Args:
        datasets (list): List of training datasets.
        val_dataset (Dataset): Validation dataset
        ignore_backgrounds (bool, optional): if `True`, set backgrounds for all but the last dataset
            to -1. This can be combined with loss functions that ignore the -1 labels.

    """
    self.datasets = datasets
    self.val_dataset = val_dataset
    self.ignore_backgrounds = ignore_backgrounds
    # self.train_paths = [path for dataset in datasets for path in dataset.train_paths]
    # self.val_paths = [path for dataset in datasets for path in dataset.val_paths]
    self.train_paths = self._aggregate_paths([dataset.train_paths + dataset.train_paths
                                              for dataset in datasets])
    np.random.seed(123)
    np.random.shuffle(self.train_paths)
    self.val_paths = val_dataset.train_paths + val_dataset.val_paths
    np.random.seed(321)
    np.random.shuffle(self.val_paths)
    self.modalities = list(set.intersection(*[set(dataset.modalities) for dataset in datasets]))
    assert(self.modalities), "Given datasets don't have intersecting modalities"
    self.classes = ['background'] + [clss for dataset in datasets for clss in dataset.classes[1:]]
    self.name = '_'.join(dataset.name for dataset in datasets)

  def load_path(self, path):
    """Load a given dataset path.

    This handles autoincrementing the labels between datasets. If `self.ignore_backgrounds`,
    this is also handled here.

    Args:
        path (string): path to a dataset element.

    Returns:
        tuple: (x, y) image, segmentation.

    """
    x, y = super(MultiDataset, self).load_path(path)
    if path.startswith(self.val_dataset.root_path):
      x_filtered = Tools.filter_modalities(self.val_dataset.modalities, self.modalities, x)
      return x_filtered, y
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
    return x_filtered, y

  @staticmethod
  def _aggregate_paths(paths_list):
    """Combine paths with balanced amounts from each list of paths.

    Args:
        paths_list (list of lists): List of lists of paths to be aggregated.

    """
    max_len_paths = max(len(paths) for paths in paths_list)
    aggregated_paths = []
    for paths in paths_list:
      aggregated_paths.extend(itertools.islice(itertools.cycle(paths), max_len_paths))
    return aggregated_paths


if __name__ == '__main__':
  preprocess_dataset(RawMRBrainS13(), '../../data/preprocessed_datasets/')
  # preprocess_dataset(RawATLAS(), '../../data/preprocessed_datasets/')
  # preprocess_dataset(RawBraTS(), '../../data/preprocessed_datasets/')
  # preprocess_dataset(RawIBSR(), '../../data/preprocessed_datasets/')
  # preprocess_dataset(RawMRBrainS17(), '../../data/preprocessed_datasets/')
  # preprocess_dataset(RawMRBrainS18(), '../../data/preprocessed_datasets/')
