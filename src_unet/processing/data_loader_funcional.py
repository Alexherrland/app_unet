from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import os

class VideoDataset(Dataset):
    def __init__(self, low_quality_path, high_quality_path, crop_size=None):
        self.low_quality_images = [os.path.join(low_quality_path, img) for img in os.listdir(low_quality_path) if img.endswith(('.jpg', '.png'))]
        self.high_quality_images = [os.path.join(high_quality_path, img) for img in os.listdir(high_quality_path) if img.endswith(('.jpg', '.png'))]
        
        self.crop_size = crop_size
         # Transformaciones para baja resolución
        self.transform_low = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3098, 0.2859, 0.2710], std=[0.2019, 0.1929, 0.1909])
        ])

        # Transformaciones para alta resolución
        self.transform_high = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3098, 0.2859, 0.2710], std=[0.2019, 0.1929, 0.1909])
        ])

    def __len__(self):
        return len(self.low_quality_images)

    def _paired_random_crop(self, lr_img, hr_img):
        """Crop mantiene la relación de escala"""
        if self.crop_size is None:
            return lr_img, hr_img
        
        w, h = hr_img.size 
        new_h, new_w = self.crop_size, self.crop_size
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        lr_img = lr_img.crop((left//4, top//4, left//4 + new_w//4, top//4 + new_h//4))
        hr_img = hr_img.crop((left, top, left + new_w, top + new_h))

        return lr_img, hr_img

    def _paired_rotation(self, lr_img, hr_img):
        """Rotaciones random a ambas imagenes"""
        # Rotaciones
        rotations = [
            (None, None), 
            (Image.ROTATE_90, Image.ROTATE_90),
            (Image.ROTATE_180, Image.ROTATE_180),
            (Image.ROTATE_270, Image.ROTATE_270),
            (Image.FLIP_LEFT_RIGHT, Image.FLIP_LEFT_RIGHT)
        ]
        rot_lr, rot_hr = random.choice(rotations)
        
        if rot_lr:
            lr_img = lr_img.transpose(rot_lr) 
        if rot_hr:
            hr_img = hr_img.transpose(rot_hr)
        
        return lr_img, hr_img

    def __getitem__(self, idx):
        low_quality_image = Image.open(self.low_quality_images[idx]).convert('RGB')
        high_quality_image = Image.open(self.high_quality_images[idx]).convert('RGB')

        # Rotaciones
        low_quality_image, high_quality_image = self._paired_rotation(low_quality_image, high_quality_image)

        # Crop
        low_quality_image, high_quality_image = self._paired_random_crop(low_quality_image, high_quality_image) 

        low_quality_image = self.transform_low(low_quality_image)
        high_quality_image = self.transform_high(high_quality_image)

        return low_quality_image, high_quality_image

def get_dataloader(low_quality_path, high_quality_path, batch_size=4):

    # Dataset con las transformaciones
    dataset = VideoDataset(low_quality_path, high_quality_path)

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #media, std = calcular_media_std(dataloader)
    #print("Media:", media)
    #print("Desviación estándar:", std)
    return dataloader


def calcular_media_std(dataloader):
    """
    Calcula la media y la desviación estándar de un DataLoader.
    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in dataloader:
        low_res_images, high_res_images = data
        batch_samples = low_res_images.size(0)
        low_res_images = low_res_images.view(batch_samples, low_res_images.size(1), -1)
        high_res_images = high_res_images.view(batch_samples, high_res_images.size(1), -1)
        mean += low_res_images.mean(2).sum(0)
        std += low_res_images.std(2).sum(0)
        mean += high_res_images.mean(2).sum(0)
        std += high_res_images.std(2).sum(0)
        nb_samples += batch_samples * 2

    mean /= nb_samples
    std /= nb_samples
    return mean, std