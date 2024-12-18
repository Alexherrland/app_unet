import torch
import numpy as np
from scipy.stats import norm
import os
import matplotlib.pyplot as plt

def mean_absolute_error(predictions, labels):
    """
    Calcula el Error Absoluto Medio (MAE)
    
    Args:
    - predictions (torch.Tensor): Predicciones del modelo
    - labels (torch.Tensor): Etiquetas reales
    
    Returns:
    - float: Valor de MAE
    """
    return torch.mean(torch.abs(predictions - labels)).item()

def crps_gaussian(predictions, labels):
    """
    Calcula el CRPS asumiendo una distribución Gaussiana
    
    Args:
    - predictions (torch.Tensor): Predicciones del modelo
    - labels (torch.Tensor): Etiquetas reales
    
    Returns:
    - float: Valor de CRPS
    """
    # Calcular media y desviación estándar de las predicciones
    pred_mean = predictions.mean().cpu().item()
    pred_std = predictions.std().cpu().item()
    
    # Convertir a numpy para usar scipy
    labels_np = labels.cpu().numpy()
    
    # Calcular CRPS
    z = (labels_np - pred_mean) / pred_std
    crps = pred_std * (
        z * (2 * norm.cdf(z) - 1) + 
        2 * norm.pdf(z) - 
        1 / np.sqrt(np.pi)
    )
    
    return np.mean(crps)


def generate_images(model, inputs, labels, epoch):
    """
    Genera comparativas visuales de imágenes durante el entrenamiento:
    - Guarda imágenes de entrada, objetivo y predicción
    - Crea una visualización lado a lado
    - Guarda en carpeta 'training_comparisons'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(66)
    model.eval()
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)
    inputs, labels, predictions = inputs.cpu().numpy(), labels.cpu().numpy(), predictions.cpu().numpy()
    
    os.makedirs('training_comparisons', exist_ok=True)
    
    plt.figure(figsize=(15,20))

    display_list = [inputs[-1].transpose((1, 2, 0)), labels[-1].transpose((1, 2, 0)), predictions[-1].transpose((1, 2, 0))]
    title = ['Input', 'Real', 'Predicted']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow((display_list[i] + 1) / 2)
        plt.axis('off')
    
    plt.savefig(f'training_comparisons/epoch_{epoch}_comparison.png')
    plt.close()