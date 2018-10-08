#!/usr/bin/python3
import nibabel as nib
import numpy as np
from sys import argv

'Example usage: npz2nii input.npz output  (will produce output_img.nii.gz and output_seg.nii.gz)'

npz_path = argv[1]

assert npz_path.endswith('.npz'), 'Paths given should have .npz extension, got %s instead' % npz_path
img = np.load(npz_path)
data = nib.Nifti1Image(img['data'], np.eye(4))
seg = nib.Nifti1Image(img['seg'], np.eye(4))
fname = argv[2]
print("Saving data to", fname + "_img.nii.gz")
nib.save(data, fname + "_img.nii.gz")
print("Saving segmentation to", fname + "_seg.nii.gz")
nib.save(seg, fname + "_seg.nii.gz")
