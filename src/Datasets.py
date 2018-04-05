from glob import glob
import nibabel as nib
import numpy as np
import pickle

from BatchGenerator import AsyncBatchGenerator, Transformations

# Deprecate IBSR
class IBSR(object):

  n_classes = 4

  def __init__(self, from_pickle=True, path=None):
    self.subjects = {}
    if from_pickle:
      if path is None:
        self.load_pickle()
      else:
        self.load_pickle(path)
    else:
      if path is None:
        self.load_raw()
      else:
        self.load_raw(path)


  def load_pickle(self, path='ibsr.pkl'):
    """ Loads IBSR dataset from a pikled version, with the offsets already
    aligned """
    f = open(path, 'rb')
    self.subjects = pickle.load(f)
    f.close()
    return self

  def load_raw(self, root_path='../../data/NITRC-ibsr-Downloads/'):
    """ Loads IBSR dataset from the raw data dowloaded from the IBSR repo """

    # Values taken from the ibsr readme, adjusted based on the starting index of
    # each scan
    offsets = {
        '1_24'  : 0,
        '2_4'   : 6,
        '5_8'   : 1,
        '4_8'   : 2,
        '6_10'  : 0,
        '7_8'   : 0,
        '8_4'   : 2,
        '11_3'  : 1,
        '12_3'  : 0,
        '13_3'  : 0,
        '15_3'  : 0,
        '16_3'  : 0,
        '17_3'  : 3,
        '100_23': 0,
        '110_3' : 2,
        '111_2' : 3,
        '112_2' : 1,
        '191_3' : 3,
        '202_3' : 3,
        '205_3' : 2
    }

    def get_IBSR_paths(root_path=''):
      """ Returns the paths for the proper files in containing the IBSR dataset.
      Paths are structured in a dict of {scan_id : (data_files,
      segmentation_file)}.
      Notice that images come in one layer per file, while segmentation comes in
      a single file for all layers. """

      data_dirs = sorted(glob(root_path + '20Normals_T1/20Normals_T1/*'))
      def get_scan_id(path):
        """ Given a data directory, returns the scan number.
            In this case, it's the name of the directory. """
        return path.split('/')[-1]

      paths = {}
      for data_dir in data_dirs:
        scan_id = get_scan_id(data_dir)
        data = sorted(glob(data_dir + '/*'),
                      key=lambda s: int(s[:-4].split('_')[-1]))
        seg = '%s20Normals_T1_seg/20Normals_T1_seg/%s.buchar' % (root_path,
                                                                 scan_id)
        paths[scan_id] = data, seg
      return paths


    def load_volumetric_image(data_paths):
      """ Create a single 3D matrix from a list of files containing one layer
      each.
      Files are 16-bit raw images, big-endian (thus the '>i2') """
      return np.array([np.fromfile(path, dtype='>i2')
                          for path in data_paths]).reshape((-1, 256, 256, 1))

    def load_buchar(path):
      """ Load a buchar file containing the segmentation information. """
      seg = np.fromfile(path, dtype='uint8').reshape((-1, 256, 256, 1))
      translate_labels = lambda x: {0:0, 128:1, 192:2, 254:3}[x]
      seg = np.vectorize(translate_labels)(seg)
      return seg

    def load_path(scan_id):
      print('Loading scan', scan_id)
      data, seg = paths[scan_id]
      # Apply offsets between data layers and segmentation layers.
      data, seg = (load_volumetric_image(data), load_buchar(seg))
      start = offsets[scan_id]
      end = start + seg.shape[0]
      data = data[start:end,:,:]
      try:
        assert(data.shape[:-1] == seg.shape[:-1])
      except AssertionError:
        print('%s:\tdata and seg have different shapes %s and %s' %
              (scan_id, data.shape, seg.shape))
      return scan_id, (data, seg)


    paths = get_IBSR_paths(root_path)

    subjects = dict(load_path(scan_id) for scan_id in paths)

    self.subjects = subjects
    return self

  def save_pickle(self, path='ibsr.pkl'):
    f = open(path, 'wb')
    pickle.dump(self.subjects, f)
    f.close()

  def get_X(self):
    return [data for data, _ in self.subjects.values()]

  def get_Y(self):
    return [seg for _, seg in self.subjects.values()]


class TrainableDataset(object):

  def __init__(self, dataset, validation_portion=.2, normalize=True):
    X = dataset.get_X()
    n = len(X)
    if normalize:
      for i in range(n):
        X[i] = (X[i] - np.mean(X[i])) / np.std(X[i])

    Y = dataset.get_Y()

    val_n = int(n * validation_portion)
    # Last val_n samples are used as validation
    self.X_train = X[:-val_n]
    self.Y_train = Y[:-val_n]
    self.X_val = X[-val_n:]
    self.Y_val = Y[-val_n:]

if __name__ == '__main__':
  """ Load raw dataset and save to pickle """
  dataset = IBSR(from_pickle=False)
  dataset.save_pickle()



class Dataset(object):
  """ Abstract Dataset class. Requires the subclass to contain train_paths and
      val_paths, and implement the load_path and _get_paths methods
  """

  def __init__(self, root_path, validation_portion):
    paths = self._get_paths(root_path)
    val_n = int(len(paths) * validation_portion)
    self.train_paths = paths[:-val_n]
    self.val_paths = paths[-val_n:]

  @staticmethod
  def normalize(X):
    X -= np.mean(X)
    X /= np.std(X)
    return X

  def load_path(self, path):
    raise NotImplementedError()

  def _get_paths(self, root_path):
    """ Get a meaningful path to each sample from the root path """
    raise NotImplementedError()

  def _get_data_path(self, path):
    """ Given a meaningful path to a sample, get the exact path to the data. """
    raise NotImplementedError()

  def _get_seg_paths(self, path):
    """ Given a meaningful path to a sample, get the exact path to the
    segmentation. """
    raise NotImplementedError()

  def get_generators(self, patch_shape, patch_multiplicity=1):
    """ Get both training generator and validation generator.
        The training generator crops images and applies augmentation,
        the validation generator doesn't.
    Args:
        patch_shape: dimensions of the training patches.
        patch_multiplicity: validation patches are only cropped to have dims
                            multiples of this value.

    """
    train_generator = AsyncBatchGenerator(patch_shape,
                                          self.train_paths,
                                          self.load_path)
    # val_generator = AsyncBatchGenerator(patch_shape,
    #                                     self.val_paths,
    #                                     self.load_path)
    val_generator = AsyncBatchGenerator(None,
                                        self.val_paths,
                                        self.load_path,
                                        transformations=Transformations.NONE,
                                        patch_multiplicity=patch_multiplicity)
    return train_generator, val_generator

class NiftiDataset(Dataset):
  """ Dataset that loads files of NIFTI (.nii) format """

  def load_path(self, path):
    data_path = self._get_data_path(path)
    data = nib.load(data_path).get_data().astype('float32')
    data = self.normalize(data.reshape(data.shape + (1,)))

    seg_paths = self._get_seg_paths(path)
    seg = (sum(nib.load(path).get_data() for path in seg_paths) != 0).astype(
                                                                        'int8')
    seg = seg.reshape(seg.shape + (1,))
    return (data, seg)

class ATLAS(NiftiDataset):

  def __init__(self, root_path='../../data/ATLAS_R1.1/',
               validation_portion=.2):
    super(ATLAS, self).__init__(root_path, validation_portion)

  def _get_data_path(self, path):
    data_path = glob(path + '/*deface*')
    assert(len(data_path) == 1)  # There should be exactly one deface per dir
    data_path = data_path[0]
    return data_path

  def _get_seg_paths(self, path):
    return glob(path + '/*LesionSmooth*')

  def _get_paths(self, root_path):
    return glob(root_path + '*/*/*')


class BraTS(NiftiDataset):

  def __init__(self, root_path='../../data/MICCAI_BraTS17_Data_Training/',
               validation_portion=.2):
    super(BraTS, self).__init__(root_path, validation_portion)

  def _get_data_path(self, path):
    data_path = glob(path + '/*t1.nii.gz')
    if len(data_path) != 1:
      print(path, data_path)
    assert(len(data_path) == 1)  # There should be exactly one deface per dir
    data_path = data_path[0]
    return data_path

  def _get_seg_paths(self, path):
    return glob(path + '/*seg*')

  def _get_paths(self, root_path):
    # This merges Higher Grade Glioma (HGG) with Lower Grade Glioma (LGG).
    # That might not be ideal.
    return glob(root_path + '*/*')

