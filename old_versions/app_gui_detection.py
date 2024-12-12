import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processing')))
from video_processing import process_video 
from map_detection import detectar_mapa

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
            # Detectar el mapa del video
            mapas_referencia = {
                "Dust2": [
                    cv2.imread("dust2_referencia1.jpg"),
                    cv2.imread("dust2_referencia2.jpg")
                ],
                "Mirage": [
                    cv2.imread("mirage_referencia1.jpg")
                ]
            }
            mapa_detectado = detectar_mapa(input_video_path, mapas_referencia)

            if mapa_detectado:
                # Llamar a la función process_video con el modelo como argumento
                process_video(input_video_path, output_path, mapa_detectado)  

                messagebox.showinfo("Procesamiento Completo", f"Video procesado guardado en: {output_path}")
            else:
                messagebox.showerror("Error", "No se pudo detectar el mapa del video.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")  # Muestra la excepción completa
        finally:
            process_button.config(state=tk.DISABLED)

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

process_button = tk.Button(root, text="Procesar Video", command=process_video_file, state=tk.DISABLED)
process_button.pack(pady=20)

root.mainloop()