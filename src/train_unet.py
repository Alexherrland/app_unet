import torch
import torch.optim as optim
import torch.nn as nn
import os
import torch.cuda.amp as amp
import wandb

from processing.L1SSIMLoss import L1SSIMLoss #Funcion de loss personalizada
from processing.unet_model import UNet , ResidualUNet # Cargamos Unet y ResidualUnet (sin uso temporalmente)
from processing.data_loader_funcional  import get_dataloader # Importar las imagenes ya preparadas
from skimage.metrics import peak_signal_noise_ratio, structural_similarity #Metricas de entrenamiento

def train(
    low_quality_path,
    high_quality_path,
    epochs=10,
    batch_size=4,
    unet_depth=4,
    unet_wf=6,
    unet_padding=False,
    unet_batch_norm=False,
    unet_up_mode='upconv',
    loss_function=nn.MSELoss(),
    optimizer_class=optim.Adam,
    learning_rate=0.001,
    previous_model=False,  
    previous_model_path='unet_model.pth',
    scale_factor=4,
    use_residual= True,
    enable_mixed_precision=True,
    enable_scheduler = True,
    run_name=None 
):
    # Inicializar wandb
    wandb.init(
        project="super-resolution-unet",
        name=run_name or f"UNet-{'-'.join(str(x) for x in [unet_depth, unet_wf, scale_factor])}",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "unet_depth": unet_depth,
            "unet_wf": unet_wf,
            "unet_padding": unet_padding,
            "unet_batch_norm": unet_batch_norm,
            "unet_up_mode": unet_up_mode,
            "scale_factor": scale_factor,
            "use_residual": use_residual,
            "mixed_precision": enable_mixed_precision,
            "loss_function": str(loss_function),
            "optimizer": optimizer_class.__name__
        }
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    ModelClass = ResidualUNet if use_residual else UNet
    if previous_model:
        # Cargar el estado del modelo anterior
        try:
            checkpoint = torch.load(previous_model_path)
            model = ModelClass(depth=unet_depth, wf=unet_wf, padding=unet_padding,
                     batch_norm=unet_batch_norm, up_mode=unet_up_mode,
                     scale_factor=scale_factor)
            model.load_state_dict(checkpoint)
            print(f"Cargando modelo anterior desde: {previous_model_path}")
        except FileNotFoundError:
            print(f"Error: No se encontró el modelo anterior en {previous_model_path}")
            return
    else:
        # Crear un nuevo modelo
        model = ModelClass(depth=unet_depth, wf=unet_wf, padding=unet_padding,
                     batch_norm=unet_batch_norm, up_mode=unet_up_mode,
                     scale_factor=scale_factor)
    #Movemos el modelo a la GPU
    model.to(device)

    criterion = loss_function
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduler (Probar mas adelante, de momento seguir con el LR normal)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', #Reduce el LR cuando se activa
        factor=0.5, #Cuanto se reduce la tasa de aprendizaje cuando se activa
        patience=5, #Cuantos epochs deben pasar hasta que se active
        verbose=True,
        min_lr=1e-6 #Minimo LR aceptado
    )

    # Precision Mixta
    if enable_mixed_precision and torch.cuda.is_available():
        scaler = amp.GradScaler(device)
    else:
        scaler = None

    dataloader = get_dataloader(low_quality_path, high_quality_path, batch_size=batch_size)  # Crea el dataloader


    start_epoch = 0  # Empezar en la epoca que se guardo el estado del modelo
    best_psnr = 0 #Metrica para wandb

    for epoch in range(start_epoch, epochs) :
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        for inputs, labels in dataloader:
            #print("Input shape:", inputs.shape)
            #print("Labels shape:", labels.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Usar o no Mixed Precision
            if scaler is not None:
                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Entrenamiento default
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Calcula las métricas PSNR y SSIM
            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            for i in range(outputs_np.shape[0]):
                output_image = outputs_np[i].transpose(1, 2, 0)
                label_image = labels_np[i].transpose(1, 2, 0)
                epoch_psnr += peak_signal_noise_ratio(label_image, output_image, data_range=1.0)  # data_range=1.0 para imágenes normalizadas
                epoch_ssim += structural_similarity(label_image, output_image, multichannel=True, data_range=1.0, win_size=3)
            epoch_loss += loss.item()
        
        #Calculo de las metricas
        epoch_loss /= len(dataloader)
        epoch_psnr /= len(dataloader.dataset)
        epoch_ssim /= len(dataloader.dataset)

        # Logging con wandb
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "psnr": epoch_psnr,
            "ssim": epoch_ssim,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        # Log de imágenes de ejemplo cada 10 épocas
        if epoch % 10 == 0:
            # Seleccionar algunas imágenes de entrada, salida y etiqueta
            input_images = [wandb.Image(inputs[j].cpu(), caption=f"Input {j}") for j in range(min(3, inputs.shape[0]))]
            output_images = [wandb.Image(outputs[j].detach().cpu(), caption=f"Output {j}") for j in range(min(3, outputs.shape[0]))]
            label_images = [wandb.Image(labels[j].cpu(), caption=f"Label {j}") for j in range(min(3, labels.shape[0]))]
            
            wandb.log({
                "input_images": input_images,
                "output_images": output_images,
                "label_images": label_images
            })
        
        # Guardar el mejor modelo
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            torch.save(model.state_dict(), 'best_unet_model.pth')
            wandb.save('best_unet_model.pth')
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}')

        #Learning Rate Scheduler, de momento dejar desactivado hasta poder hacer pruebas
        if enable_scheduler:
            scheduler.step(epoch_loss)

        # Guarda el modelo al final de cada epoch
        torch.save(model.state_dict(), f'unet_model_epoch_{epoch+1}.pth')

    #Guarda el modelo una vez se ha terminado el entrenamiento (es redundante ya que se guarda por cada epoch, pero tampoco pasa nada por tenerlo)
    torch.save(model.state_dict(), 'unet_model.pth')

    # Finalizar sesión de wandb
    wandb.finish()

if __name__ == "__main__":
    # Define las rutas a tus datos
    files_dir = "data/train"
    low_quality_path = os.path.join(files_dir, "train_low")
    high_quality_path = os.path.join(files_dir, "train_high")

    # Cambiar los parámetros del entrenamiento
    epochs = 150
    batch_size = 16
    learning_rate = 0.0005
    unet_depth = 4  # Ajustar la profundidad
    unet_wf = 6
    unet_padding = True
    unet_batch_norm = True
    unet_up_mode = 'upconv'
    loss_function = L1SSIMLoss(l1_weight=0.1, ssim_weight=1.0) # Por default: nn.L1Loss()
    optimizer_class = optim.Adam
    scale_factor = 4

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
        learning_rate=learning_rate,
        scale_factor=scale_factor,
        use_residual=False,
        enable_mixed_precision = False,
        enable_scheduler = False,
        run_name="Experimento-001",
        previous_model=True,  # Variable para indicar si se usa un modelo anterior en vez de iniciar un nuevo entrenamiento
        previous_model_path='unet_model_epoch_13.pth'  # Ruta al modelo anterior
    )
