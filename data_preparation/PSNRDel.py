import os
import numpy as np
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2gray

class ImagePSNRFilter:
    def __init__(self, low_img_dir, high_img_dir, psnr_threshold=21):
        """
        Inicializa el filtro de imágenes basado en PSNR
        
        :param low_img_dir: Directorio de imágenes de baja resolución
        :param high_img_dir: Directorio de imágenes de alta resolución
        :param psnr_threshold: Umbral mínimo de PSNR (dB)
        """
        self.low_img_dir = low_img_dir
        self.high_img_dir = high_img_dir
        self.psnr_threshold = psnr_threshold
        
    def preprocess_image(self, img):
        """
        Preprocesa la imagen:
        - Convierte imágenes RGBA a RGB
        - Convierte a escala de grises si es necesario
        
        :param img: Imagen de entrada
        :return: Imagen procesada
        """
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        
        if img.ndim == 3 and img.shape[2] > 1:
            img = rgb2gray(img)
        
        return img
    
    def calculate_psnr(self, low_img, high_img):
        """
        Calcula el PSNR entre imagen de baja y alta resolución
        
        :param low_img: Imagen de baja resolución
        :param high_img: Imagen de alta resolución
        :return: Valor de PSNR en dB
        """
        high_img_resized = transform.resize(high_img, (128, 128), anti_aliasing=True)
        
        low_img = self.preprocess_image(low_img)
        high_img_resized = self.preprocess_image(high_img_resized)
        
        return peak_signal_noise_ratio(low_img, high_img_resized)
    
    def filter_images(self):
        """
        Filtra imágenes basándose en el umbral de PSNR
        Elimina imágenes que no cumplan con el criterio de calidad
        """
        low_images = sorted([f for f in os.listdir(self.low_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        high_images = sorted([f for f in os.listdir(self.high_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(low_images) == len(high_images), "El número de imágenes de baja y alta resolución debe ser igual"
        
        total_images = len(low_images)
        deleted_images = 0
        
        for low_img_name, high_img_name in zip(low_images, high_images):
            try:
                low_img_path = os.path.join(self.low_img_dir, low_img_name)
                high_img_path = os.path.join(self.high_img_dir, high_img_name)
                
                low_img = io.imread(low_img_path)
                high_img = io.imread(high_img_path)
                
                psnr_value = self.calculate_psnr(low_img, high_img)
                
                if psnr_value < self.psnr_threshold:
                    os.remove(low_img_path)
                    os.remove(high_img_path)
                    deleted_images += 1
                    print(f"Eliminando {low_img_name} con PSNR: {psnr_value:.2f} dB")
            
            except Exception as e:
                print(f"Error procesando {low_img_name}: {e}")
        
        print(f"\nResumen:")
        print(f"Imágenes totales: {total_images}")
        print(f"Imágenes eliminadas: {deleted_images}")
        print(f"Imágenes restantes: {total_images - deleted_images}")

if __name__ == "__main__":
    low_img_dir = 'd:/app_unet/data/train/train_low_redimensionadas_model'
    high_img_dir = 'd:/app_unet/data/train/train_high_redimensionadas_model'
    
    psnr_filter = ImagePSNRFilter(low_img_dir, high_img_dir, psnr_threshold=25)
    
    psnr_filter.filter_images()