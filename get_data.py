import numpy as np
import nibabel as nib
import os
import json
from torch.utils.data import Dataset
import torch
from scipy.fft import fft2, ifft2, fftshift


class Data(Dataset):
  def __init__(self, project_path='data', train=True, img_shape=(60,60,60)):
    
    json_file_path = os.path.join(project_path, 'dataset.json')
    with open(json_file_path) as json_file:
      dataset = json.load(json_file)
    
    self.shape = img_shape
    
    if train:
      self.imgs = [os.path.join(project_path, x['image'][2:]) for x in dataset['training']]
    else:
      self.imgs = [os.path.join(project_path, x[2:]) for x in dataset['test']]

  def __getitem__(self, index):
    # Gernarate data on sample data
    hr_img = nib.load(self.imgs[index]).get_fdata()
    hr_img = self.resize_imgs(hr_img, self.shape)
    hr_img = self.normalize_img_values(hr_img)
    lr_img = self.downgrade_resolution(hr_img, offset=20)

    # We also need to include channels
    return torch.from_numpy(lr_img).unsqueeze(0).to(torch.float), torch.from_numpy(hr_img).unsqueeze(0).to(torch.float)
    
  def __len__(self):
    return len(self.imgs)
  
  @staticmethod
  def downgrade_resolution(hr_img, offset=10):
    # Compute FFT of the image to obtain its K-space
    # Degrade the resolution by zeroing the outer part of the 
    # 3D k-space along two axes representing two MR phase encoding direction
    lr_img = np.zeros(hr_img.shape)
    filter = np.zeros(hr_img.shape[:-1])
    filter[offset:-offset, offset:-offset] = 1
  
    for slc in range(hr_img.shape[-1]):
      # Take the 2-dimensional DFT and centre the frequencies
      hr_dft = fft2(hr_img[:,:,slc])
      hr_dft = fftshift(hr_dft)

      lr_dft = hr_dft * filter
      lr_img[:,:,slc] = np.abs(ifft2(lr_dft))
    return lr_img

  @staticmethod
  def resize_imgs(org_vol, new_shape):
    resized_vol = np.zeros(new_shape)
    resized_vol[:org_vol.shape[0], :org_vol.shape[1], :org_vol.shape[2]] = org_vol
    return resized_vol
  
  @staticmethod
  def normalize_img_values(vol):
    vol_min = np.min(vol)
    vol_max = np.max(vol)
    return (vol - vol_min) / (vol_max - vol_min)
