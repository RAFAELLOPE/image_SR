import numpy as np
import nibabel as nib


test_img = '/imagesTr/hippocampus_004.nii.gz'
vol_img = nib.load(test_img)
vol = vol_img.get_fdata()

with open('volume_shape', 'rw') as file:
    file.save(f"Volume shape: {vol.shape}")