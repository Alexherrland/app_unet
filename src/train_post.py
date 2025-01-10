import torch
import torch.nn as nn
import time
from torcheval.metrics.functional import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from processing.Runet_model import SR_Unet_Residual_Deep
from processing.post_processing import PostProcessDenoiser
from processing.utils import generate_images

def train_postprocessing(
    sr_model_path,
    train_dataloader,
    num_epochs=5,
    learning_rate=1e-4,
    device='cuda',
    save_path='denoising_model.pt'
):
    sr_model = SR_Unet_Residual_Deep()
    sr_model.load_state_dict(torch.load(sr_model_path,weights_only=False))
    sr_model.eval()
    sr_model.to(device)
    
    denoising_model = PostProcessDenoiser()
    denoising_model.to(device)
    
    optimizer = torch.optim.AdamW(denoising_model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    
    best_loss = float('inf')
    log_interval = 500
    
    for epoch in range(num_epochs):
        denoising_model.train()
        total_loss = 0
        total_psnr = 0
        total_count = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                sr_output = sr_model(inputs)
            denoised = denoising_model(sr_output)
            
            loss = criterion(denoised, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_psnr += peak_signal_noise_ratio(denoised, targets)
            total_count += 1
            
            if batch_idx % log_interval == 0 and batch_idx > 0:
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| psnr {:8.3f}".format(
                        epoch, batch_idx, len(train_dataloader), 
                        total_psnr / total_count
                    )
                )
                generate_images(denoising_model, sr_output.detach(), targets, f'post_epoch_{epoch}')
        
        avg_loss = total_loss / len(train_dataloader)
        avg_psnr = total_psnr / total_count
        epoch_time = time.time() - start_time
        
        print("-" * 89)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train psnr {:8.3f} | Train Loss {:8.3f}".format(
                epoch, epoch_time, avg_psnr, avg_loss
            )
        )
        print("-" * 89)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(denoising_model.state_dict(), save_path)
            print(f'Nuevo mejor modelo guardado con pérdida {avg_loss:.4f}')
    
    return denoising_model

def apply_postprocess(image, sr_model, denoising_model, device='cuda'):
    """Aplica SR y post-procesamiento a una imagen"""
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        sr_output = sr_model(image.to(device))
        final_output = denoising_model(sr_output)
        
    return final_output

if __name__ == "__main__":
    sr_model_path = 'robustunet_model.pt'
    batch_size = 8
    
    from processing.data_loader_RUnet import ImageDataset
    dataset = ImageDataset(is_train=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    denoising_model = train_postprocessing(
        sr_model_path=sr_model_path,
        train_dataloader=train_dataloader,
        num_epochs=50
    )