import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=3,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        
        Version Adaptada para la mejora de video.

        Args:
            in_channels (int): Número de canales de entrada (3 para RGB).
            n_classes (int): Número de canales de salida (3 para RGB).
            depth (int): Profundidad de la red (número de bloques convolucionales descendentes).  
                         Aumentar la profundidad incrementa la capacidad del modelo pero también el coste computacional.
                         Comentario: Seria interesante dedicarle un apartado al TFG modificando la profundidad viendo resultados, hacer cuando tengamos la base creada
            wf (int): Factor de ancho (width factor). El número de filtros en la primera capa es 2**wf. 
                      Aumentar wf incrementa el número de filtros en cada capa, lo que aumenta la capacidad del modelo pero también el coste computacional.
                      Comentario: Buscar mas informacion sobre el wf, si ves que es interesante, documentalo a fondo en el TFG
            padding (bool): Si es True, aplica padding para mantener el tamaño espacial, es el mismo que la salida.
                            Cuidado: Esto puede introducir artefactos si tienes este problema, desactivalo.
            batch_norm (bool): Si es True, usa Batch Normalization después de cada capa convolucional.
            up_mode (str): Modo de upsampling ('upconv' para convolución transpuesta, 'upsample' para upsampling bilineal).
                           'upconv' generalmente produce mejores resultados pero es más costoso computacionalmente.
                           Comentario: De base usaremos el mejor, pero estara interesante ver resultados con upsample, decidir si introducir en el TFG
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out