import torch
from torcheval.metrics.functional import peak_signal_noise_ratio
from Runet_model import SR_Unet_Residual_Deep, SR_Unet
from post_processing import PostProcessDenoiser
from utils import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def compare_models(test_image, target_image, base_model_path, enhanced_model_path, denoising_model_path, device='cuda'):
    """
    Compare results between base SR model and SR model with denoising
    
    Args:
        test_image: Input low resolution image (tensor)
        target_image: Ground truth high resolution image (tensor)
        base_model_path: Path to base SR model without denoising
        enhanced_model_path: Path to SR model with residual learning
        denoising_model_path: Path to trained denoising model
        device: Device to run models on
    """
    # Load models
    base_model = SR_Unet()
    base_model.load_state_dict(torch.load(base_model_path,weights_only=False))
    base_model.eval()
    base_model.to(device)
    
    enhanced_model = SR_Unet_Residual_Deep()
    enhanced_model.load_state_dict(torch.load(enhanced_model_path,weights_only=False))
    enhanced_model.eval()
    enhanced_model.to(device)
    
    denoising_model = PostProcessDenoiser()
    denoising_model.load_state_dict(torch.load(denoising_model_path,weights_only=False))
    denoising_model.eval()
    denoising_model.to(device)
    
    with torch.no_grad():
        base_output = base_model(test_image.to(device))
        
        enhanced_output = enhanced_model(test_image.to(device))
        
        denoised_output = denoising_model(enhanced_output)
        
        base_psnr = peak_signal_noise_ratio(base_output, target_image.to(device))
        base_mae = mean_absolute_error(base_output, target_image.to(device))
        
        enhanced_psnr = peak_signal_noise_ratio(enhanced_output, target_image.to(device))
        enhanced_mae = mean_absolute_error(enhanced_output, target_image.to(device))
        
        denoised_psnr = peak_signal_noise_ratio(denoised_output, target_image.to(device))
        denoised_mae = mean_absolute_error(denoised_output, target_image.to(device))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        axes[0].imshow(np.transpose(test_image.squeeze().cpu().numpy(), (1, 2, 0)))
        axes[0].set_title('Input Image')
        
        axes[1].imshow(np.transpose(base_output.squeeze().cpu().numpy(), (1, 2, 0)))
        axes[1].set_title(f'Base Model\nPSNR: {base_psnr:.2f}, MAE: {base_mae:.4f}')
        
        axes[2].imshow(np.transpose(enhanced_output.squeeze().cpu().numpy(), (1, 2, 0)))
        axes[2].set_title(f'Enhanced Model\nPSNR: {enhanced_psnr:.2f}, MAE: {enhanced_mae:.4f}')
        
        axes[3].imshow(np.transpose(denoised_output.squeeze().cpu().numpy(), (1, 2, 0)))
        axes[3].set_title(f'Enhanced + Denoised\nPSNR: {denoised_psnr:.2f}, MAE: {denoised_mae:.4f}')
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

        print("-" * 50)
        print("Detailed Metrics Comparison:")
        print("-" * 50)
        print(f"Base Model        - PSNR: {base_psnr:.2f}, MAE: {base_mae:.4f}")
        print(f"Enhanced Model    - PSNR: {enhanced_psnr:.2f}, MAE: {enhanced_mae:.4f}")
        print(f"Enhanced+Denoised - PSNR: {denoised_psnr:.2f}, MAE: {denoised_mae:.4f}")
        
        return base_output, enhanced_output, denoised_output

if __name__ == "__main__":
    from data_loader_RUnet import ImageDataset
    from torch.utils.data import DataLoader
    
    test_dataset = ImageDataset(is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    test_input, test_target = next(iter(test_loader))
    
    outputs = compare_models(
        test_input,
        test_target,
        base_model_path='robustunet_modelSR.pt',
        enhanced_model_path='robustunet_model.pt',
        denoising_model_path='denoising_model.pt'
    )