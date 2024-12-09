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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5062, 0.4583, 0.4214], std=[0.1695, 0.1677, 0.1675])
    ])

    # Dataset con las transformaciones
    dataset = VideoDataset(low_quality_path, high_quality_path, transform=transform)

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