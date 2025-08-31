# EMIDEC Mask Dataset for DiffMask training
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import SimpleITK as sitk
import torchio as tio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# EMIDEC specific preprocessing - dimensions: 72x72x10
PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.Clamp(out_min=-1000, out_max=400),
    tio.RescaleIntensity(in_min_max=(-1000, 400), out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(10, 72, 72))  # EMIDEC dimensions (depth, height, width)
])

PREPROCESSING_MASK_TRANSFORMS = tio.Compose([
    tio.CropOrPad(target_shape=(10, 72, 72))  # EMIDEC dimensions
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1,), flip_probability=0.5),
])

class EMIDECMASKDataset(Dataset):
    def __init__(self, root_dir, text_txt_path, augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = text_txt_path
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSFORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSFORMS

    def train_transform(self, image, label, p):
        train_transforms = tio.Compose([
            tio.RandomFlip(axes=(1,), flip_probability=p),
        ])
        image = train_transforms(image)
        label = train_transforms(label)
        return image, label

    def get_file_names(self):
        # Get all files from Image directory
        all_file_names = glob.glob(os.path.join(self.root_dir, 'Image', '*.nii.gz'))
        test_file_names = set()

        # Read test file names
        with open(self.remove_test_path, 'r') as file:
            for line in file:
                test_file_name = line.strip()
                test_file_names.add(test_file_name)

        # Filter out test files
        filtered_file_names = [
            f for f in all_file_names
            if os.path.basename(f) not in test_file_names
        ]
        return filtered_file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        
        # Load image (2 channels for EMIDEC)
        img = tio.ScalarImage(path)
        
        # Get corresponding mask path
        mask_path = path.replace("/Image/", "/Mask/")
        mask = tio.LabelMap(mask_path)

        # Apply preprocessing
        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)

        # Apply augmentation if training
        p = np.random.choice([0, 1])
        img, mask = self.train_transform(img, mask, p)

        mask = mask.data
        img = img.data
        
        # Compute histogram for condition
        hist = torch.histc(img[mask > 0], bins=16, min=-1, max=1) / mask.sum()
        if torch.sum(hist) == 0 or torch.isnan(hist).any():
            print(index, mask.sum(), "----", hist)
            print(img[mask > 0])

        # Compute minimal enclosing sphere
        sphere = []
        for c in range(mask.shape[0]):
            center, radius = self.min_enclosing_sphere(mask[c])
            sphere_mask = self.create_sphere_mask(mask[c].shape, center, radius)
            sphere.append(sphere_mask)

        sphere = torch.stack(sphere, dim=0)

        # Normalize to [-1, 1]
        sphere = sphere * 2 - 1
        mask = mask * 2 - 1

        return {
            'data': img,
            'label': mask,
            'hist': hist,
            'sphere': sphere
        }

    def min_enclosing_sphere(self, mask):
        """Find the minimal enclosing sphere for a binary mask"""
        indices = torch.nonzero(mask)
        if len(indices) == 0:
            center = torch.tensor([mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 2], dtype=torch.float32)
            radius = 0.0
        else:
            center = indices.float().mean(dim=0)
            distances = torch.norm(indices.float() - center, dim=1)
            radius = distances.max().item()
        return center, radius

    def create_sphere_mask(self, shape, center, radius):
        """Create a sphere mask given center and radius"""
        x = torch.arange(shape[0])
        y = torch.arange(shape[1])
        z = torch.arange(shape[2])
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        distance = torch.sqrt((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2)
        sphere_mask = (distance <= radius).float()
        
        return sphere_mask