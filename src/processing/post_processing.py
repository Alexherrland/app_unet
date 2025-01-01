import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveNoiseEstimator(nn.Module):
    """Estimador de ruido adaptativo"""
    def __init__(self, channels=3):
        super(AdaptiveNoiseEstimator, self).__init__()
        self.noise_estimation = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.noise_estimation(x)

class EdgePreservingDenoiser(nn.Module):
    """Denoiser que preserva bordes"""
    def __init__(self, channels=3):
        super(EdgePreservingDenoiser, self).__init__()
        
        self.edge_detector = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.smooth_branch = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, padding=1)
        )
        
        self.detail_branch = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, kernel_size=1)
        )

    def forward(self, x):
        edges = self.edge_detector(x)
        smooth = self.smooth_branch(x)
        details = self.detail_branch(x)
        return edges * details + (1 - edges) * smooth

class PostProcessDenoiser(nn.Module):
    def __init__(self, channels=3):
        super(PostProcessDenoiser, self).__init__()
        
        self.noise_estimator = AdaptiveNoiseEstimator(channels)
        self.edge_denoiser = EdgePreservingDenoiser(channels)
        
        self.refinement = nn.Sequential(
            nn.Conv2d(channels*2, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, padding=1)
        )
        
        self.residual_weight = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        noise_map = self.noise_estimator(x)
        denoised = self.edge_denoiser(x)
        concat_features = torch.cat([x, denoised], dim=1)
        refined = self.refinement(concat_features)
        return x + self.residual_weight * refined