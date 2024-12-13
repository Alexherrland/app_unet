import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2

low_res_video_path = None  # Inicializar variables globales FUERA de las funciones
high_res_video_path = None


def select_video(resolution):
    global low_res_video_path, high_res_video_path
    if resolution == "low":
        label = low_res_video_label
        path_variable = low_res_video_path
        title = "Seleccionar video de baja resolución"
    elif resolution == "high":
        global high_res_video_path
        label = high_res_video_label
        path_variable = high_res_video_path
        title = "Seleccionar video de alta resolución"
    else:
        return

    file_path = filedialog.askopenfilename(title=title, filetypes=[("Archivos MP4", "*.mp4"), ("Archivos MKV", "*.mkv")])
    if file_path:
        label.config(text=f"Video {resolution} resolución: {file_path}")
        path_variable = file_path  # Actualizar la variable LOCAL
        if resolution == "low":
            low_res_video_path = path_variable # Actualizar la variable GLOBAL
        elif resolution == "high":
            high_res_video_path = path_variable # Actualizar la variable GLOBAL

        check_if_both_videos_selected()
    else:
        label.config(text=f"No se ha seleccionado ningún video de {resolution} resolución")


def check_if_both_videos_selected():
    if low_res_video_path and high_res_video_path:
        extract_frames_button.config(state=tk.NORMAL)
    else:
        extract_frames_button.config(state=tk.NORMAL) #DISABLED , lo mismo que en el boton, mirar por que no se aplica la logica, de momento ignorar

def get_last_frame_number(directory):
    """
    Obtiene el número del último frame guardado en un directorio, no esta en uso de momento, deberia pasar esta funcion a utils.

    Args:
      directory: Ruta al directorio donde se guardan los frames.

    Returns:
      Número del último frame guardado o 0 si no hay frames en el directorio.
    """
    try:
        last_frame = sorted(os.listdir(directory))[-1]
        return int(last_frame.split('_')[1].split('.')[0])
    except IndexError:
        return 0

def extract_frames_from_video():

    
    output_dir = "data/train"
    train_low_dir = os.path.join(output_dir, "train_low")
    train_high_dir = os.path.join(output_dir, "train_high")

    os.makedirs(train_low_dir, exist_ok=True)
    os.makedirs(train_high_dir, exist_ok=True)

    low_vidcap = cv2.VideoCapture(low_res_video_path)
    high_vidcap = cv2.VideoCapture(high_res_video_path)

    low_success, low_image = low_vidcap.read()
    high_success, high_image = high_vidcap.read()
    
    last_frame_number = get_last_frame_number(train_low_dir)  # la idea es que se puedan introducir diferentes videos y sigan los frames, de momento no esta implementado
    count = 0  # Continuar la cuenta desde el último frame  # last_frame_number + 1
    frame_counter = 0 
    while low_success and high_success:  #  Iterar mientras ambos videos tengan frames
        high_quality_image = high_image.copy()
        #high_quality_image = cv2.resize(high_image, (256, 256))  # Si es necesario redimensionar

        low_quality_image = low_image.copy()
        #low_quality_image = cv2.resize(low_image, (256, 256)) # Si es necesario redimensionar

        if frame_counter % 15 == 0:  # Guarda solo cada x frames
            filename = f"frame_{count:05d}.jpg" #  Mismo nombre de archivo para ambos
            high_quality_path = os.path.join(train_high_dir, filename)
            low_quality_path = os.path.join(train_low_dir, filename)

            cv2.imwrite(high_quality_path, high_quality_image)
            cv2.imwrite(low_quality_path, low_quality_image)
            count += 1  # Incrementa el contador de archivos guardados
        
        frame_counter += 1  # Incrementa el contador de frames
        low_success, low_image = low_vidcap.read()
        high_success, high_image = high_vidcap.read()

    low_vidcap.release()
    high_vidcap.release()

    messagebox.showinfo("Extracción de Frames", f"Frames guardados en: {output_dir}")

low_res_video_path = None
high_res_video_path = None


root = tk.Tk()
root.title("Mejora de Calidad de Video con U-Net")


select_low_res_button = tk.Button(root, text="Seleccionar Video Baja Resolución", command=lambda: select_video("low"))
select_low_res_button.pack(pady=5)

low_res_video_label = tk.Label(root, text="No se ha seleccionado ningún video de baja resolución")
low_res_video_label.pack(pady=5)

select_high_res_button = tk.Button(root, text="Seleccionar Video Alta Resolución", command=lambda: select_video("high"))
select_high_res_button.pack(pady=5)

high_res_video_label = tk.Label(root, text="No se ha seleccionado ningún video de alta resolución")
high_res_video_label.pack(pady=5)

extract_frames_button = tk.Button(root, text="Extraer Frames de Video", command=extract_frames_from_video) #, state=tk.DISABLED no funciona la logica para cuando ambos videos esten seleccionados, de momento ignorar ya lo arreglare
extract_frames_button.pack(pady=10)


root.mainloop()