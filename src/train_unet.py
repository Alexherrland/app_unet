import torch
import torch.optim as optim
import torch.nn as nn
import os
from processing.unet_model import UNet
from processing.data_loader_funcional  import get_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # Importa las métricas PSNR y SSIM

def train(
    low_quality_path,
    high_quality_path,
    epochs=10,
    batch_size=4,
    unet_depth=5,
    unet_wf=6,
    unet_padding=False,  # Nuevo parámetro para padding
    unet_batch_norm=False,  # Nuevo parámetro para batch normalization
    unet_up_mode='upconv',
    loss_function=nn.MSELoss(),
    optimizer_class=optim.Adam,
    learning_rate=0.001
):
    
    model = UNet(depth=unet_depth, wf=unet_wf, padding=unet_padding,
                 batch_norm=unet_batch_norm, up_mode=unet_up_mode)
    criterion = loss_function
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    #criterion = nn.MSELoss() # criterion = loss_function   # Cambia la función de pérdida a MSE para denoising
    #optimizer = optim.Adam(model.parameters(), lr=0.0001) # optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device("cpu")

    dataloader = get_dataloader(low_quality_path, high_quality_path, batch_size=batch_size)  # Crea el dataloader


    for epoch in range(epochs):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calcula las métricas PSNR y SSIM
            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            for i in range(outputs_np.shape[0]):
                output_image = outputs_np[i].transpose(1, 2, 0)  # Transpone las dimensiones para que coincidan con la entrada de las funciones de métricas
                label_image = labels_np[i].transpose(1, 2, 0)
                epoch_psnr += peak_signal_noise_ratio(label_image, output_image, data_range=1.0)  # data_range=1.0 para imágenes normalizadas
                epoch_ssim += structural_similarity(label_image, output_image, multichannel=True, data_range=1.0)
            epoch_loss += loss.item()
        
        # Calcula el promedio de las métricas en la época
        epoch_loss /= len(dataloader)
        epoch_psnr /= len(dataloader.dataset)
        epoch_ssim /= len(dataloader.dataset)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}')

    torch.save(model.state_dict(), 'unet_model.pth')  # Guarda el modelo

if __name__ == "__main__":
    # Define las rutas a tus datos
    files_dir = "data/train"
    low_quality_path = os.path.join(files_dir, "train_low")
    high_quality_path = os.path.join(files_dir, "train_high")

    #  Configura los parámetros del entrenamiento (experimenta con valores desde aqui, no tomar la funcion)
    epochs = 150
    batch_size = 8
    learning_rate = 0.0001
    unet_depth = 5
    unet_wf = 6
    unet_padding = True  # Ajusta el valor de padding
    unet_batch_norm = True  # Ajusta el valor de batch normalization
    unet_up_mode = 'upconv' #upconv y upsample
    loss_function = nn.MSELoss() #Investigar cambios de funcion de perdida, actualmente MSELoss y L1Loss
    optimizer_class = optim.Adam

    train(
        low_quality_path,
        high_quality_path,
        epochs=epochs,
        batch_size=batch_size,
        unet_depth=unet_depth,
        unet_wf=unet_wf,
        unet_padding=unet_padding,
        unet_batch_norm=unet_batch_norm,
        unet_up_mode=unet_up_mode,
        loss_function=loss_function,
        learning_rate=learning_rate
    )