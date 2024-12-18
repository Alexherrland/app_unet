import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import json

class ImageDataset(Dataset):
    def __init__(self, is_train=True, normalization_stats_path='normalization_stats.json'):
        """
        Dataset para cargar y preprocesar imágenes de super resolución:
        
        Funcionalidades:
        - Carga de imágenes de baja y alta resolución
        - Normalización con estadísticas precalculadas
        - Data augmentation opcional
        - Transformaciones y preprocesamiento
        
        Métodos principales:
        - __len__: Retorna número de imágenes
        - __getitem__: Carga y preprocesa imagen individual
        - normalize: Normalización avanzada con estadísticas
        - augment: Aumento de datos con transformaciones
        """
        self.low_img_dir = 'data/train/train_low_redimensionadas_modelCS'
        self.high_img_dir = 'data/train/train_high_redimensionadas_modelCS'
        
        self.low_images = os.listdir(self.low_img_dir)
        self.high_images = os.listdir(self.high_img_dir)
        
        with open(normalization_stats_path, 'r') as f:
            stats = json.load(f)
        
        self.low_mean = stats['low_resolution']['mean']
        self.low_std = stats['low_resolution']['std']
        self.high_mean = stats['high_resolution']['mean']
        self.high_std = stats['high_resolution']['std']

        self.is_train = is_train
        
        # Resize si es necesario
        #self.resize = transforms.Resize((128, 128), antialias=True)

    def __len__(self):
        return len(self.low_images)

    def normalize(self, input_image, target_image):
        """
        Normalización usando estadísticas precalculadas por canal
        """
        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(target_image)
        
        input_image = transforms.functional.normalize(input_image, 
                                                     mean=self.low_mean, 
                                                     std=self.low_std)
        target_image = transforms.functional.normalize(target_image, 
                                                      mean=self.high_mean, 
                                                      std=self.high_std)
        
        return input_image, target_image
    
    def normalize_basic(self, input_image, target_image):
        """
        Normalización básica al rango [-1, 1]
        """

        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(target_image)
        
        input_image  = input_image*2 - 1
        target_image = target_image*2 - 1

        return input_image, target_image
        
    def augment(self, input_image, target_image):
        """
        Aumento de datos con:
        - Volteo horizontal
        - Rotación 
        """
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
        
        if torch.rand([]) < 0.3:
            angle = torch.randint(-30, 30, (1,)).item()
            input_image = transforms.functional.rotate(input_image, angle)
            target_image = transforms.functional.rotate(target_image, angle)
        
        return input_image, target_image

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_img_dir, self.low_images[idx])
        high_img_path = os.path.join(self.high_img_dir, self.high_images[idx])

        input_image = Image.open(low_img_path).convert("RGB")
        target_image = Image.open(high_img_path).convert("RGB")


        input_image, target_image = self.normalize_basic(input_image, target_image)

        #if self.is_train:
            #input_image, target_image = self.augment(input_image, target_image)

        return input_image, target_image