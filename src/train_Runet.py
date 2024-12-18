# Script principal de entrenamiento para el modelo RU-Net de Super Resolución

import torch
import torch.optim as optim
import torch.nn as nn
import os
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, random_split

# Importaciones de módulos propios para entrenamiento, pérdida personalizada y modelo
from processing.train import train_epoch, evaluate_epoch, train_model
from processing.L1SSIMLoss import L1SSIMLoss 
from processing.Runet_model import SR_Unet , SR_Unet_Residual , SR_Unet_Residual_Deep
from processing.data_loader_RUnet import ImageDataset

def train(
    epochs=150,
    batch_size=16,
    learning_rate=0.0001,
    optimizer = optim.AdamW,
    loss_function=None,
    previous_model_path=None,
    enable_mixed_precision=False,
    previous_model=False,
    enable_scheduler = True
):
    """
    Función principal de entrenamiento con múltiples configuraciones:
    
    - Carga de modelo previo opcional
    - Configuración de dispositivo (GPU)
    - Selección de función de pérdida
    - Configuración de optimizador y scheduler
    - División de dataset en entrenamiento y prueba
    - Entrenamiento del modelo
    """
    if previous_model :
        try:
            checkpoint = torch.load(previous_model_path,weights_only=False)
            model = SR_Unet_Residual_Deep()
            model.load_state_dict(checkpoint)
            print(f"Cargando modelo anterior desde: {previous_model_path}")
        except FileNotFoundError:
            print(f"Error: No se encontró el modelo anterior en {previous_model_path}")
            return
    else:
        model = SR_Unet_Residual_Deep()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    criterion = loss_function
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    scheduler = None
    if enable_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',  # Reduce el LR cuando se activa
            factor=0.3,  # Cuanto se reduce la tasa de aprendizaje cuando se activa
            patience=2,  # Cuantos epochs deben pasar hasta que se active
            min_lr=1e-6  # Minimo LR aceptado
        )
     # Crear el dataset
    dataset = ImageDataset(is_train=True)


    # Dividir el dataset en conjuntos de entrenamiento y prueba (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Crear los dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Entrenar el modelo
    train_model(
        model=model, 
        model_name='robustunet_model', 
        save_model='.', 
        optimizer=optimizer, 
        criterion=criterion, 
        train_dataloader=train_dataloader, 
        valid_dataloader=test_dataloader, 
        num_epochs=epochs, 
        device=device,
        scheduler=scheduler
    )


if __name__ == "__main__":
    # Configuración de hiperparámetros
    epochs = 150
    batch_size = 2
    learning_rate = 0.0001
    optimizer = optim.AdamW
    # Creación de función de pérdida personalizada L1SSIM
    loss_function = L1SSIMLoss(l1_weight=0.1, ssim_weight=1) 
    
    train(
        epochs=150,
        batch_size=batch_size,
        loss_function=loss_function,
        learning_rate=learning_rate,
        optimizer = optimizer,
        enable_mixed_precision = False,
        enable_scheduler = True,
        previous_model=True, 
        previous_model_path='robustunet_model.pt',  
    )