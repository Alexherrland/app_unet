import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class L1SSIMLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=1.0):
        super(L1SSIMLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, outputs, labels):
        # Calcular L1 Loss
        l1_loss = F.l1_loss(outputs, labels)
        
        # Calcular SSIM Loss (1 - SSIM, para que sea un loss que minimizar)
        ssim_loss = 1 - self._ssim(outputs, labels)
        
        # Combinar ambas pérdidas
        combined_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        return combined_loss

    def _ssim(self, img1, img2, window_size=11, size_average=True):
        # Implementación de SSIM en PyTorch
        channel = img1.size()[1]
        window = self._create_window(window_size, channel)
        
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def _create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window