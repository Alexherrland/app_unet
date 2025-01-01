import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from data_loader_RUnet import ImageDataset
from Runet_model import SR_Unet_Residual_Deep, SR_Unet
from torcheval.metrics.functional import peak_signal_noise_ratio
from utils import mean_absolute_error, crps_gaussian

def evaluate_single_image(model_path, image_idx, save_dir='evaluation_results'):
    """
    Evalúa el modelo en una imagen específica del dataset y guarda los resultados
    
    Args:
    - model_path (str): Ruta al modelo guardado
    - image_idx (int): Índice de la imagen en el dataset
    - save_dir (str): Directorio donde guardar los resultados
    
    Returns:
    - dict: Métricas de evaluación
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = ImageDataset(is_train=True)
    
    if image_idx >= len(dataset) or image_idx < 0:
        raise ValueError(f"Índice {image_idx} fuera de rango. El dataset tiene {len(dataset)} imágenes.")
    
    model = SR_Unet_Residual_Deep()
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False)) 
    model = model.to(device)
    model.eval()
    
    input_image, target_image = dataset[image_idx]

    input_batch = input_image.unsqueeze(0).to(device)
    target_batch = target_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)
    
    psnr = peak_signal_noise_ratio(prediction, target_batch).item()
    mae = mean_absolute_error(prediction, target_batch)
    crps = crps_gaussian(prediction, target_batch)
    
    metrics = {
        'PSNR': psnr,
        'MAE': mae,
        'CRPS': crps
    }
    
    plt.figure(figsize=(15, 5))
    
    def denormalize(img):
        return (img.cpu().numpy().transpose(1, 2, 0) + 1) / 2
    
    images = [input_image, target_image, prediction.squeeze(0)]
    titles = ['Input', 'Target', 'Prediction']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i+1)
        plt.imshow(denormalize(img))
        plt.title(f'{title}\n{img.shape[1]}x{img.shape[2]}')
        plt.axis('off')
    
    plt.suptitle(f'PSNR: {psnr:.2f} | MAE: {mae:.4f} | CRPS: {crps:.4f}')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'evaluation_image_{image_idx}.png'))
    plt.close()
    
    with open(os.path.join(save_dir, f'metrics_image_{image_idx}.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f'{metric}: {value}\n')
    
    return metrics

def evaluate_multiple_images(model_path, image_indices, save_dir='evaluation_results'):
    """
    Evalúa el modelo en múltiples imágenes del dataset
    
    Args:
    - model_path (str): Ruta al modelo guardado
    - image_indices (list): Lista de índices de imágenes a evaluar
    - save_dir (str): Directorio donde guardar los resultados
    
    Returns:
    - dict: Métricas promedio de todas las imágenes evaluadas
    """
    all_metrics = []
    
    for idx in image_indices:
        print(f"Evaluando imagen {idx}...")
        metrics = evaluate_single_image(model_path, idx, save_dir)
        all_metrics.append(metrics)
    
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[f'avg_{metric}'] = sum(m[metric] for m in all_metrics) / len(all_metrics)
    
    with open(os.path.join(save_dir, 'average_metrics.txt'), 'w') as f:
        for metric, value in avg_metrics.items():
            f.write(f'{metric}: {value}\n')
    
    return avg_metrics

if __name__ == "__main__":
    # Evaluar una sola imagen
    model_path = "robustunet_modelSRResidual.pt"
    # metrics = evaluate_single_image(model_path, image_idx=678)
    #print("Métricas para imagen individual:", metrics)
    
    # Evaluar múltiples imágenes
    image_indices = [10, 100, 200, 300, 400]
    avg_metrics = evaluate_multiple_images(model_path, image_indices)
    print("Métricas promedio:", avg_metrics)


