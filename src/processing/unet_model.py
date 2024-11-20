import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, depth=5, wf=6, padding=True,
                 batch_norm=True, up_mode='upconv'):
        """
        Implementación de la nueva arquitectura U-Net.

        Args:
            in_channels (int): número de canales de entrada. Por defecto 3 para imágenes RGB.
            n_classes (int): número de clases de salida. Por defecto 3 para imágenes RGB.
            depth (int): profundidad de la red. Controla el número de capas de downsampling y upsampling.
            wf (int): factor de ancho. Escala el número de canales en cada capa.
            padding (bool): si se aplica padding o no en las convoluciones. 
                             Si es True, mantiene el tamaño espacial en cada capa.
            batch_norm (bool): si se aplica batch normalization o no después de cada convolución.
            up_mode (str): modo de upsampling. Puede ser 'upconv' para convolución transpuesta o 
                           'upsample' para upsampling bilineal seguido de una convolución 1x1.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            # Bloques convolucionales en la ruta descendente (encoder)
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            # Bloques convolucionales en la ruta ascendente (decoder)
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        # Capa convolucional final 1x1 para obtener la salida
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            # Ruta descendente (encoder)
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            # Ruta ascendente (decoder)
            x = up(x, blocks[-i-1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        """
        Bloque convolucional usado en la U-Net.

        Args:
            in_size (int): número de canales de entrada.
            out_size (int): número de canales de salida.
            padding (bool): si se aplica padding o no.
            batch_norm (bool): si se aplica batch normalization o no.
        """
        super(UNetConvBlock, self).__init__()
        block = []

        # Dos capas convolucionales 3x3 con ReLU y opcionalmente Batch Normalization
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        """
        Bloque de upsampling usado en la U-Net.

        Args:
            in_size (int): número de canales de entrada.
            out_size (int): número de canales de salida.
            up_mode (str): modo de upsampling ('upconv' o 'upsample').
            padding (bool): si se aplica padding o no.
            batch_norm (bool): si se aplica batch normalization o no.
        """
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            # Upsampling con convolución transpuesta
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            # Upsampling bilineal seguido de una convolución 1x1
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        # Recorta la capa de entrada para que tenga el mismo tamaño que la capa objetivo
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        # Recorta la capa de la ruta descendente para que coincida con el tamaño de la capa upsampleada
        crop1 = self.center_crop(bridge, up.shape[2:])
        # Concatena la capa upsampleada con la capa recortada de la ruta descendente
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out