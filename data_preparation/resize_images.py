from PIL import Image
import os

def redimensionar_con_recorte(carpeta_entrada, carpeta_salida, nuevo_ancho, nuevo_alto):
    os.makedirs(carpeta_salida, exist_ok=True)

    for archivo in os.listdir(carpeta_entrada):
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        ruta_salida = os.path.join(carpeta_salida, archivo)

        try:
            # Abrir la imagen
            with Image.open(ruta_entrada) as img:
                escala = max(nuevo_ancho / img.width, nuevo_alto / img.height)
                nuevo_tamano = (int(img.width * escala), int(img.height * escala))

                img_redimensionada = img.resize(nuevo_tamano)

                izquierda = (img_redimensionada.width - nuevo_ancho) // 2
                superior = (img_redimensionada.height - nuevo_alto) // 2
                derecha = izquierda + nuevo_ancho
                inferior = superior + nuevo_alto
                img_recortada = img_redimensionada.crop((izquierda, superior, derecha, inferior))

                img_recortada.save(ruta_salida)

                print(f"Imagen procesada y guardada: {ruta_salida}")

        except Exception as e:
            print(f"Error al procesar {ruta_entrada}: {e}")

carpeta_entrada_low = "data/train/train_low"
carpeta_salida_low = "data/train/train_low_redimensionadas_model"
carpeta_entrada_high = "data/train/train_high"
carpeta_salida_high = "data/train/train_high_redimensionadas_model"

tamano_low = (128, 128)
redimensionar_con_recorte(carpeta_entrada_low, carpeta_salida_low, *tamano_low)

tamano_high = (512, 512)
redimensionar_con_recorte(carpeta_entrada_high, carpeta_salida_high, *tamano_high)
