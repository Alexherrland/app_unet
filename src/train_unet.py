import torch
import torch.optim as optim
import torch.nn as nn
import os
from processing.unet_model import UNet
from processing.data_loader_funcional  import get_dataloader

def train(
    low_quality_path, 
    high_quality_path, 
    epochs=10, 
    batch_size=4,  # Modificar batch_size
    unet_depth=5, 
    unet_wf=6, 
    unet_up_mode='upconv',
    loss_function=nn.MSELoss(),  # Cambia la función de pérdida a MSE para denoising
    optimizer_class=optim.Adam, #  Define el optimizador (Adam)
    learning_rate=0.001         # tasa de aprendizaje (learning rate) de 0.001
):
    model = UNet() # Modificar la arquitectura   depth=unet_depth, wf=unet_wf, up_mode=unet_up_mode
    criterion = nn.MSELoss() # criterion = loss_function   # Cambia la función de pérdida a MSE para denoising
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Selecciona el dispositivo (GPU si está disponible, sino CPU)
    model.to(device)

    dataloader = get_dataloader(low_quality_path, high_quality_path, batch_size=batch_size)  # Crea el dataloader


    for epoch in range(epochs):  # Ajusta el número de epochs
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'unet_model.pth')  # Guarda el modelo

if __name__ == "__main__":
    # Define las rutas a tus datos
    files_dir = "data/train"
    low_quality_path = os.path.join(files_dir, "train_low")
    high_quality_path = os.path.join(files_dir, "train_high")

    #  Configura los parámetros del entrenamiento (experimenta con valores desde aqui, no tomar la funcion)
    epochs = 50
    batch_size = 8
    learning_rate = 0.0001
    unet_depth = 5
    unet_wf = 6
    unet_up_mode = 'upconv' #upconv y upsample
    loss_function = nn.MSELoss() #Investigar cambios de funcion de perdida, actualmente MSELoss y L1Loss
    optimizer_class = optim.Adam

    train(
        low_quality_path,
        high_quality_path,
        epochs,
        batch_size,
        learning_rate,
        unet_depth,
        unet_wf,
        unet_up_mode,
        loss_function,
        optimizer_class
    )