import os

import cv2
import numpy as np
import torch
from PIL import Image
from processing.unet_model import UNet
from torchvision import transforms

def process_image(input_image_path, output_image_path, model_path, scale_factor=4):
    """
    Procesa una imagen aplicando el modelo U-Net para superresolución.

    Args:
        input_image_path: Ruta a la imagen de entrada.
        output_image_path: Ruta para guardar la imagen de salida.
        model_path: Ruta al archivo del modelo U-Net.
        scale_factor: Factor de escalado para la superresolución.
    """

    # Cargar la imagen de entrada
    img = Image.open(input_image_path).convert('RGB')

    # Definir las transformaciones para la inferencia
    transform = transforms.Compose([
        transforms.Resize((img.width // scale_factor, img.height // scale_factor)),  # Redimensionar para la entrada del modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3093, 0.2858, 0.2714], std=[0.2021, 0.1933, 0.1915])
    ])

    # Aplicar las transformaciones a la imagen
    img_tensor = transform(img).unsqueeze(0)  # Añadir dimensión de batch

    # Cargar el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(scale_factor=scale_factor) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inferencia
    with torch.no_grad():
        output = model(img_tensor.to(device))

    # Procesar la salida
    output = output.squeeze(0).permute(1, 2, 0)
    output = torch.clamp(output, 0, 1)
    output = output.cpu().detach().numpy()
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Guardar la imagen de salida
    cv2.imwrite(output_image_path, output)

if __name__ == "__main__":
    input_image_path = ""
    output_image_path = ""
    model_path = ""
    process_image(input_image_path, output_image_path, model_path)