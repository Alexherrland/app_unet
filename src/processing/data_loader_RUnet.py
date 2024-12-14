import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import json

class ImageDataset(Dataset):
    def __init__(self, is_train=True, normalization_stats_path='normalization_stats.json'):
        # Paths
        self.low_img_dir = 'data/train/train_low_redimensionadas_model'
        self.high_img_dir = 'data/train/train_high_redimensionadas_model'
        
        # Load images
        self.low_images = os.listdir(self.low_img_dir)
        self.high_images = os.listdir(self.high_img_dir)
        
        # Load normalization statistics
        with open(normalization_stats_path, 'r') as f:
            stats = json.load(f)
        
        self.low_mean = stats['low_resolution']['mean']
        self.low_std = stats['low_resolution']['std']
        self.high_mean = stats['high_resolution']['mean']
        self.high_std = stats['high_resolution']['std']
        
        # Training flag
        self.is_train = is_train
        
        # Resize transformation
        #self.resize = transforms.Resize((128, 128), antialias=True)

    def __len__(self):
        return len(self.low_images)

    def normalize(self, input_image, target_image):
        """
        Advanced normalization using pre-calculated statistics
        """
        # Convert to tensor
        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(target_image)
        
        # Normalize using channel-wise mean and std
        input_image = transforms.functional.normalize(input_image, 
                                                     mean=self.low_mean, 
                                                     std=self.low_std)
        target_image = transforms.functional.normalize(target_image, 
                                                      mean=self.high_mean, 
                                                      std=self.high_std)
        
        return input_image, target_image
    
    def normalize_basic(self, input_image, target_image):
        """
        Normal normalization
        """

        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(target_image)
        
        input_image  = input_image*2 - 1
        target_image = target_image*2 - 1

        return input_image, target_image
        
    def augment(self, input_image, target_image):
        """
        Advanced data augmentation
        """
        # Random horizontal flip
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
        
        # Random rotation
        if torch.rand([]) < 0.3:
            angle = torch.randint(-30, 30, (1,)).item()
            #input_image = transforms.functional.rotate(input_image, angle)
            #target_image = transforms.functional.rotate(target_image, angle)
        
        # Color jittering
        if torch.rand([]) < 0.3:
            color_jitter = transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2
            )
            #input_image = color_jitter(input_image)
        
        return input_image, target_image

    def __getitem__(self, idx):
        # Load images
        low_img_path = os.path.join(self.low_img_dir, self.low_images[idx])
        high_img_path = os.path.join(self.high_img_dir, self.high_images[idx])

        input_image = Image.open(low_img_path).convert("RGB")
        target_image = Image.open(high_img_path).convert("RGB")


        # Normalize
        input_image, target_image = self.normalize_basic(input_image, target_image)

        # Augment only during training
        #if self.is_train:
            #input_image, target_image = self.augment(input_image, target_image)

        return input_image, target_image