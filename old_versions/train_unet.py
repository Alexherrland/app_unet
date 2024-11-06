import torch
import torch.optim as optim
import torch.nn as nn
from processing.unet_model import UNet
from processing.data_loader_funcional  import get_dataloader

def train():
    model = UNet() # Crea una instancia del modelo U-Net
    criterion = nn.MSELoss() # Cambia la función de pérdida a MSE para denoising
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Define el optimizador (Adam) con una tasa de aprendizaje (learning rate) de 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Selecciona el dispositivo (GPU si está disponible, sino CPU)
    model.to(device)

    dataloader = get_dataloader("path/to/low_quality_frames", "path/to/high_quality_frames", batch_size=4) # Crea el dataloader

    for epoch in range(10):  # Ajusta el número de epochs
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
    train()