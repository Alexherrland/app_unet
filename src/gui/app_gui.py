import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processing')))
from video_processing import process_video 


def select_file():
    global input_video_path  # Variable global para almacenar la ruta
    file_path = filedialog.askopenfilename(title="Selecciona un video", filetypes=[("Archivos MP4", "*.mp4")])
    if file_path:
        input_video_label.config(text=f"Video seleccionado: {file_path}")
        process_button.config(state=tk.NORMAL)
        input_video_path = file_path
    else:
        input_video_label.config(text="No se ha seleccionado ningún video")
        process_button.config(state=tk.DISABLED)
        return None

def select_output_file():
    directory = filedialog.askdirectory(title="Seleccionar carpeta de salida")
    if directory:
        output_video_label.config(text=f"Carpeta de salida: {directory}")
        return directory  # Devolver la ruta completa
    else:
        output_video_label.config(text="No se ha seleccionado una carpeta de salida")
        return None  # Devolver None si no se selecciona nada

def process_video_file():
    output_path = select_output_file()
    if output_path and input_video_path:  # Verifica si se seleccionaron ambos archivos
        try:
            process_video(input_video_path, output_path) # Pasa la ruta de salida
            messagebox.showinfo("Procesamiento Completo", f"Video procesado guardado en: {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}") # Muestra la excepción completa
        finally:
            process_button.config(state=tk.DISABLED)

def extract_frames_from_video():

    video_path = filedialog.askopenfilename(title="Seleccionar video para extraer frames", filetypes=[("Archivos MP4", "*.mp4")])
    if not video_path:
        return

    output_dir = "data/train"
    train_low_dir = os.path.join(output_dir, "train_low")
    train_high_dir = os.path.join(output_dir, "train_high")

    os.makedirs(train_low_dir, exist_ok=True)
    os.makedirs(train_high_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        high_quality_image =  image.copy()  #  cv2.resize(image,(256,256)) en el caso de que fuese necesario, podria preparar las imagenes para el modelo
        
        high_quality_path = os.path.join(train_high_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(high_quality_path, high_quality_image)
        
        #low_quality_image = cv2.resize(low_quality_image,(256,256), interpolation = cv2.INTER_AREA)
        #cv2.imwrite(os.path.join(train_high_dir, f"frame_{count:04d}.jpg"), high_quality_image) # <- Cambio de nombre de directorio
        #cv2.imwrite(os.path.join(train_low_dir, f"frame_{count:04d}.jpg"), low_quality_image) # <- Cambio de nombre de directorio

        success, image = vidcap.read()
        count += 1
    vidcap.release()

    messagebox.showinfo("Extracción de Frames", f"Frames guardados en: {output_dir}")

root = tk.Tk()
root.title("Mejora de Calidad de Video con U-Net")

select_input_button = tk.Button(root, text="Seleccionar Video de Entrada", command=select_file)
select_input_button.pack(pady=10)

input_video_label = tk.Label(root, text="No se ha seleccionado ningún video de entrada")
input_video_label.pack(pady=5)

select_output_button = tk.Button(root, text="Seleccionar carpeta de salida", command=select_output_file)
select_output_button.pack(pady=10)

output_video_label = tk.Label(root, text="No se ha seleccionado una ubicación para guardar el video")
output_video_label.pack(pady=5)

extract_frames_button = tk.Button(root, text="Extraer Frames de Video", command=extract_frames_from_video)
extract_frames_button.pack(pady=10)

process_button = tk.Button(root, text="Procesar Video", command=process_video_file, state=tk.DISABLED)
process_button.pack(pady=20)

root.mainloop()
