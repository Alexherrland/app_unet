import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import torchvision.transforms as transforms
import os

class VideoDataset(Dataset):
    def __init__(self, low_quality_folder, high_quality_folder):
        self.low_quality_files = sorted([os.path.join(low_quality_folder, f) for f in os.listdir(low_quality_folder) if f.endswith(('.jpg', '.png'))])
        self.high_quality_files = sorted([os.path.join(high_quality_folder, f) for f in os.listdir(high_quality_folder) if f.endswith(('.jpg', '.png'))])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.low_quality_files)

    def __getitem__(self, idx):
        low_quality_image = cv2.imread(self.low_quality_files[idx])
        high_quality_image = cv2.imread(self.high_quality_files[idx])

        low_quality_image = cv2.cvtColor(low_quality_image, cv2.COLOR_BGR2RGB)
        high_quality_image = cv2.cvtColor(high_quality_image, cv2.COLOR_BGR2RGB)

        low_quality_image = self.transform(low_quality_image)
        high_quality_image = self.transform(high_quality_image)

        return low_quality_image, high_quality_image

def get_dataloader(low_quality_folder, high_quality_folder, batch_size=1, shuffle=True):
    dataset = VideoDataset(low_quality_folder, high_quality_folder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)