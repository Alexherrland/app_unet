import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms

class SyntheticDataset(Dataset):
    def __init__(self, count=100, image_size=(256, 256)):
        self.count = count
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # Generate a random image (simple shapes on a black background)
        image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (200, 200), (255, 255, 255), -1)  # White square
        cv2.circle(image, (128, 128), 50, (0, 0, 255), -1)  # Red circle

        # Introduce random noise
        noisy_image = image + np.random.randint(0, 50, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        # Convert images to PyTorch tensors
        image = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        noisy_image = self.transform(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))

        return noisy_image, image

def get_dataloader(batch_size=10, shuffle=True):
    dataset = SyntheticDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage:
if __name__ == '__main__':
    dataloader = get_dataloader()
    for noisy, clean in dataloader:
        print("Noisy image batch shape:", noisy.shape)
        print("Clean image batch shape:", clean.shape)