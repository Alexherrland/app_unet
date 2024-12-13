import os
from PIL import Image

def reducir_calidad_imagenes(carpeta_entrada, carpeta_salida,nuevo_valor):
    """
    Reduce la calidad de las imágenes a la mitad redimensionándolas.

    Args:
        carpeta_entrada: Ruta a la carpeta con las imágenes originales.
        carpeta_salida: Ruta a la carpeta donde se guardarán las imágenes redimensionadas.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    for filename in os.listdir(carpeta_entrada):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            ruta_imagen = os.path.join(carpeta_entrada, filename)
            imagen = Image.open(ruta_imagen)
            ancho, alto = imagen.size
            nuevo_ancho = int(ancho * 0.5)
            nuevo_alto = int(alto * 0.5)

            #imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
            imagen_redimensionada = imagen.resize((nuevo_valor, nuevo_valor), Image.LANCZOS)
            imagen_redimensionada.save(os.path.join(carpeta_salida, filename))


carpeta_entrada_low = "data/train/train_low_redimensionadas"
carpeta_salida_low = "data/train/train_low_redimensionadas_2"
nuevo_valor = 64
reducir_calidad_imagenes(carpeta_entrada_low, carpeta_salida_low,nuevo_valor)

carpeta_entrada_high = "data/train/train_high_redimensionadas"
carpeta_salida_high = "data/train/train_high_redimensionadas_2"
nuevo_valor = 256
reducir_calidad_imagenes(carpeta_entrada_high, carpeta_salida_high,nuevo_valor)