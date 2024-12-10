import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image

def clean_dataset(lr_folder, hr_folder, min_psnr=18):
    valid_pairs = []
    invalid_count = 0

    for lr_filename in os.listdir(lr_folder):
        hr_filename = lr_filename
        
        lr_path = os.path.join(lr_folder, lr_filename)
        hr_path = os.path.join(hr_folder, hr_filename)
        
        if not os.path.exists(hr_path):
            continue
        
        # Cargar imágenes
        lr_img = Image.open(lr_path)
        hr_img = Image.open(hr_path)

        # (downsample 4x)
        hr_img = hr_img.resize(lr_img.size, Image.BICUBIC) 
        lr_img = np.array(lr_img) / 255.0
        hr_img = np.array(hr_img) / 255.0
        
        # Calcular PSNR
        psnr = peak_signal_noise_ratio(hr_img, lr_img)
        
        # Si el PSNR es mayor que el mínimo, guardar el par
        if psnr > min_psnr:
            valid_pairs.append((lr_filename, hr_filename))
        else:
            invalid_count += 1 
    
    print(f"Number of image pairs to be removed: {invalid_count}")
    return valid_pairs

# Uso
files_dir = "data/train"
low_quality_path = os.path.join(files_dir, "train_low")
high_quality_path = os.path.join(files_dir, "train_high")

valid_image_pairs = clean_dataset(low_quality_path, high_quality_path)
{ñ}
def remove_invalid_pairs(lr_folder, hr_folder, valid_pairs):
    valid_filenames = set(pair[0] for pair in valid_pairs)
    
    for filename in os.listdir(lr_folder):
        if filename not in valid_filenames:
            os.remove(os.path.join(lr_folder, filename))
    
    for filename in os.listdir(hr_folder):
        if filename not in valid_filenames:
            os.remove(os.path.join(hr_folder, filename))

# Eliminar pares no válidos que NO superen los 18 de SSIM , Number of image pairs to be removed: 156
remove_invalid_pairs(low_quality_path, high_quality_path, valid_image_pairs)