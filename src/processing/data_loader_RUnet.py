import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(66)

TRAIN_LOW_PATH = 'data/train/train_low_redimensionadas_model' 
TRAIN_HIGH_PATH = 'data/train/train_high_redimensionadas_model'  
VAL_LOW_PATH = 'data/val/train_low_redimensionadas_model' 
VAL_HIGH_PATH = 'data/val/train_high_redimensionadas_model' 
LOW_IMG_HEIGHT = 128
LOW_IMG_WIDTH = 128

class ImageDataset(Dataset):
    def __init__(self, is_train=True):
        self.resize = transforms.Resize((LOW_IMG_WIDTH, LOW_IMG_HEIGHT), antialias=True)
        self.is_train = is_train
        self.low_img_dir = 'data/train/train_low_redimensionadas_model'
        self.high_img_dir = 'data/train/train_high_redimensionadas_model'
        self.low_images = os.listdir('data/train/train_low_redimensionadas_model')
        self.high_images = os.listdir('data/train/train_high_redimensionadas_model')


    def __len__(self):
        return len(self.low_images)

    def normalize(self, input_image, target_image):
        input_image = input_image * 2 - 1
        target_image = target_image * 2 - 1
        return input_image, target_image

    def random_jitter(self, input_image, target_image):
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
        return input_image, target_image

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_img_dir, self.low_images[idx])
        high_img_path = os.path.join(self.high_img_dir, self.high_images[idx])

        input_image = np.array(Image.open(low_img_path).convert("RGB"))
        input_image = transforms.functional.to_tensor(input_image)

        target_image = np.array(Image.open(high_img_path).convert("RGB"))
        target_image = transforms.functional.to_tensor(target_image)


        target_image = target_image.type(torch.float32)

        input_image, target_image = self.normalize(input_image, target_image)

        if self.is_train:
            input_image, target_image = self.random_jitter(input_image, target_image)

        return input_image, target_image