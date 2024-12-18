import os
import numpy as np
from PIL import Image
import json

class NormalizationStatsCalculator:
    def __init__(self, low_img_dir, high_img_dir):
        self.low_img_dir = low_img_dir
        self.high_img_dir = high_img_dir

        # List all image files
        self.low_images = [f for f in os.listdir(low_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.high_images = [f for f in os.listdir(high_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def _load_and_convert_image(self, directory, filename):
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path).convert('RGB')
        return np.array(img) / 255.0  # Normalize to [0,1] for calculation

    def calculate_statistics(self, batch_size=1000):

        low_means = []
        low_stds = []
        high_means = []
        high_stds = []

        for i in range(0, len(self.low_images), batch_size):
            low_batch = self.low_images[i: i + batch_size]
            low_images_data = [self._load_and_convert_image(self.low_img_dir, f) for f in low_batch]
            low_images_array = np.stack(low_images_data, axis=0)
            low_means.append(low_images_array.mean(axis=(0, 1, 2)))
            low_stds.append(low_images_array.std(axis=(0, 1, 2)))

        for i in range(0, len(self.high_images), batch_size):
            high_batch = self.high_images[i: i + batch_size]
            high_images_data = [self._load_and_convert_image(self.high_img_dir, f) for f in high_batch]
            high_images_array = np.stack(high_images_data, axis=0)
            high_means.append(high_images_array.mean(axis=(0, 1, 2)))
            high_stds.append(high_images_array.std(axis=(0, 1, 2)))

        overall_low_mean = np.mean(np.array(low_means), axis=0)
        overall_low_std = np.mean(np.array(low_stds), axis=0)
        overall_high_mean = np.mean(np.array(high_means), axis=0)
        overall_high_std = np.mean(np.array(high_stds), axis=0)

        return {
            'low_resolution': {
                'mean': overall_low_mean.tolist(),
                'std': overall_low_std.tolist()
            },
            'high_resolution': {
                'mean': overall_high_mean.tolist(),
                'std': overall_high_std.tolist()
            }
        }

    def save_statistics(self, output_path='normalization_stats.json'):
       
        stats = self.calculate_statistics() 

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"Normalization statistics saved to {output_path}")
        return stats

if __name__ == "__main__":
    LOW_IMG_DIR = 'data/train/train_low_redimensionadas_model'
    HIGH_IMG_DIR = 'data/train/train_high_redimensionadas_model'

    calculator = NormalizationStatsCalculator(LOW_IMG_DIR, HIGH_IMG_DIR)

    stats = calculator.save_statistics()

    print("Low Resolution Mean:", stats['low_resolution']['mean'])
    print("Low Resolution Std:", stats['low_resolution']['std'])
    print("High Resolution Mean:", stats['high_resolution']['mean'])
    print("High Resolution Std:", stats['high_resolution']['std'])