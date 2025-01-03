import torch
import torch.nn as nn
from torchvision import transforms

LOW_IMG_HEIGHT = 128
LOW_IMG_WIDTH = 128

class FirstFeatureNoSkip(nn.Module):
    '''
    UNET sin Skip Connection
    '''
    def __init__(self, in_channels, out_channels):
        super(FirstFeatureNoSkip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlockNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockNoSkip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class EncoderNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(EncoderNoSkip, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlockNoSkip(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class DecoderNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderNoSkip, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels*2),
            nn.LeakyReLU(),
        )
        self.conv_block = ConvBlockNoSkip(out_channels*2, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_block(x)
        return x

class FinalOutputNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutputNoSkip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

class SR_Unet_NoSkip(nn.Module):
    def __init__(
            self, n_channels=3, n_classes=3
    ):
        super(SR_Unet_NoSkip, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.resize_fnc = transforms.Resize((LOW_IMG_HEIGHT*4, LOW_IMG_HEIGHT*4),
                                             antialias=True)
        self.in_conv1 = FirstFeatureNoSkip(n_channels, 64)
        self.in_conv2 = ConvBlockNoSkip(64, 64)

        self.enc_1 = EncoderNoSkip(64, 128)
        self.enc_2 = EncoderNoSkip(128, 256)
        self.enc_3 = EncoderNoSkip(256, 512)
        self.enc_4 = EncoderNoSkip(512, 1024)

        self.dec_1 = DecoderNoSkip(1024, 512)
        self.dec_2 = DecoderNoSkip(512, 256)
        self.dec_3 = DecoderNoSkip(256, 128)
        self.dec_4 = DecoderNoSkip(128, 64)

        self.out_conv = FinalOutputNoSkip(64, n_classes)


    def forward(self, x):
        x = self.resize_fnc(x)
        x = self.in_conv1(x)
        x = self.in_conv2(x)

        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)

        x = self.dec_1(x)
        x = self.dec_2(x)
        x = self.dec_3(x)
        x = self.dec_4(x)

        x = self.out_conv(x)
        return x

class FirstFeature(nn.Module):
    '''
    UNET con Skip connections
    '''
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

class SR_Unet(nn.Module):
    def __init__(
            self, n_channels=3, n_classes=3
    ):
        super(SR_Unet, self).__init__()
        """
        Arquitectura SRU-Net básica para Super Resolución:
        - Redimensiona imagen de entrada 
        - Múltiples etapas de codificación y decodificación
        - Skip connections entre etapas
        - Salida con transformación Tanh
        """
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.resize_fnc = transforms.Resize(
            (LOW_IMG_HEIGHT*4, LOW_IMG_WIDTH*4), 
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )
        self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = ConvBlock(64, 64)

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)
        self.enc_5 = Encoder(1024, 2048)

        self.dec_0 = Decoder(2048, 1024) 
        self.dec_1 = Decoder(1024, 512)
        self.dec_2 = Decoder(512, 256)
        self.dec_3 = Decoder(256, 128)
        self.dec_4 = Decoder(128, 64)

        self.out_conv = FinalOutput(64, n_classes)

    def forward(self, x):
        x = self.resize_fnc(x)
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)
        x6 = self.enc_5(x5)

        x = self.dec_0(x6, x5)
        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.out_conv(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.leaky_relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += self.shortcut(residual)
        out = self.leaky_relu(out)
        
        return out

class SR_Unet_Residual(nn.Module):
    def __init__(
            self, n_channels=3, n_classes=3
    ):
        super(SR_Unet_Residual, self).__init__()
        """
        Variante de SRU-Net con bloques residuales:
        - Reemplaza bloques convolucionales por bloques residuales
        - Mejora el flujo de gradientes
        """
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.resize_fnc = transforms.Resize(
            (LOW_IMG_HEIGHT*4, LOW_IMG_WIDTH*4), 
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )
        self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = ResidualBlock(64, 64)

        self.enc_1 = ResidualBlock(64, 128, downsample=True)
        self.enc_2 = ResidualBlock(128, 256, downsample=True)
        self.enc_3 = ResidualBlock(256, 512, downsample=True)
        self.enc_4 = ResidualBlock(512, 1024, downsample=True)
        self.enc_5 = ResidualBlock(1024, 2048, downsample=True)

        self.dec_0 = Decoder(2048, 1024) 
        self.dec_1 = Decoder(1024, 512)
        self.dec_2 = Decoder(512, 256)
        self.dec_3 = Decoder(256, 128)
        self.dec_4 = Decoder(128, 64)

        self.out_conv = FinalOutput(64, n_classes)

    def forward(self, x):
        x = self.resize_fnc(x)
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)
        x6 = self.enc_5(x5)

        x = self.dec_0(x6, x5)
        x = self.dec_1(x, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.out_conv(x)
        return x 
    

class SR_Unet_Residual_Deep(nn.Module):
    def __init__(
            self, n_channels=3, n_classes=3
    ):
        super(SR_Unet_Residual_Deep, self).__init__()
        """
        Versión más profunda de SRU-Net residual:
        - Bloques residuales más complejos
        - Decodificadores con bloques residuales adicionales
        - Mayor capacidad de representación
        """
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.resize_fnc = transforms.Resize(
            (LOW_IMG_HEIGHT*4, LOW_IMG_WIDTH*4), 
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )
        self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = ResidualBlock(64, 64)

        self.enc_1 = ResidualBlock(64, 128, downsample=True)
        self.enc_2 = ResidualBlock(128, 256, downsample=True)
        self.enc_3 = ResidualBlock(256, 512, downsample=True)
        self.enc_4 = ResidualBlock(512, 1024, downsample=True)
        self.enc_5 = ResidualBlock(1024, 2048, downsample=True)

        self.dec_0 = DecoderResidual(2048, 1024) 
        self.dec_1 = DecoderResidual(1024, 512)
        self.dec_2 = DecoderResidual(512, 256)
        self.dec_3 = DecoderResidual(256, 128)
        self.dec_4 = DecoderResidual(128, 64)

        self.out_conv = FinalOutput(64, n_classes)

    def forward(self, x):
        x = self.resize_fnc(x)
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)
        x6 = self.enc_5(x5)

        x = self.dec_0(x6, x5)
        x = self.dec_1(x, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.out_conv(x)
        return x 
    
class ResidualBlockDeep(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlockDeep, self).__init__()
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += self.shortcut(residual)
        out = self.leaky_relu(out)
        
        return out

class DecoderResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderResidual, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
        self.residual_block = ResidualBlockDeep(out_channels + out_channels, out_channels)
        self.conv_block = ConvBlock(out_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x_concat = torch.concat([x, skip], dim=1)
        residual = self.residual_block(x_concat)
        x = self.conv_block(x_concat)
        x = x + residual
        return x