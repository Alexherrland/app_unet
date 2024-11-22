import cv2
import numpy as np

def detectar_mapa(video_path, mapas_referencia, umbral_similitud=0.7):
    """
    Detecta el mapa de CS2 en un video usando ORB y similitud estructural.

    Args:
      video_path: Ruta al video de entrada.
      mapas_referencia: Diccionario con las imágenes de referencia de cada mapa.
      umbral_similitud: Umbral de similitud para considerar un mapa como detectado.

    Returns:
      Nombre del mapa detectado o None si no se detecta ningún mapa.
    """

    # Crear objeto ORB
    orb = cv2.ORB_create()

    # Extraer frames clave del video
    frames_clave = extraer_frames_clave(video_path)

    mejor_similitud = 0
    mapa_detectado = None

    for mapa, imagen_referencia in mapas_referencia.items():
        # Calcular descriptores de la imagen de referencia
        kp_ref, des_ref = orb.detectAndCompute(imagen_referencia, None)

        for frame in frames_clave:
            # Calcular descriptores del frame
            kp_frame, des_frame = orb.detectAndCompute(frame, None)

            # Comparar descriptores usando un matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_ref, des_frame)

            # Ordenar los matches por distancia
            matches = sorted(matches, key=lambda x: x.distance)

            # Calcular la similitud estructural
            similitud = structural_similarity(imagen_referencia, frame, multichannel=True)

            # Combinar la similitud de descriptores y la similitud estructural
            puntuacion = len(matches) * similitud

            if puntuacion > mejor_similitud:
                mejor_similitud = puntuacion
                mapa_detectado = mapa

    # Verificar si la similitud supera el umbral
    if mejor_similitud >= umbral_similitud:
        return mapa_detectado
    else:
        return None

def extraer_frames_clave(video_path):
    """
    Extrae frames clave de un video.
    """
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 10 == 0:  # Extraer un frame cada 10 frames
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames