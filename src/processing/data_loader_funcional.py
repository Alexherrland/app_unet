from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class VideoDataset(Dataset):
    def __init__(self, low_quality_path, high_quality_path, transform=None):
        self.low_quality_images = [os.path.join(low_quality_path, img) for img in os.listdir(low_quality_path) if img.endswith(('.jpg', '.png'))]
        self.high_quality_images = [os.path.join(high_quality_path, img) for img in os.listdir(high_quality_path) if img.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.low_quality_images)

    def __getitem__(self, idx):
        low_quality_image = Image.open(self.low_quality_images[idx]).convert('RGB')
        high_quality_image = Image.open(self.high_quality_images[idx]).convert('RGB')

        if self.transform:
            low_quality_image = self.transform(low_quality_image)
            high_quality_image = self.transform(high_quality_image)

        return low_quality_image, high_quality_image

def get_dataloader(low_quality_path, high_quality_path, batch_size=4):
    # Transformaciones de augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset con las transformaciones
    dataset = VideoDataset(low_quality_path, high_quality_path, transform=transform)

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader