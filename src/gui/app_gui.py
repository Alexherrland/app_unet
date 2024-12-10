import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processing')))
from image_processing import process_image 


def select_file():
    global input_image_path 
    file_path = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Archivos de imagen", "*.png *.jpg *.jpeg")])  # Aceptar diferentes formatos de imagen
    if file_path:
        input_video_label.config(text=f"Imagen seleccionada: {file_path}") 
        process_button.config(state=tk.NORMAL)
        input_image_path = file_path 
    else:
        input_video_label.config(text="No se ha seleccionado ninguna imagen")
        process_button.config(state=tk.DISABLED)
        return None

def select_output_file():
    directory = filedialog.askdirectory(title="Seleccionar carpeta de salida")
    if directory:
        output_video_label.config(text=f"Carpeta de salida: {directory}")
        return directory 
    else:
        output_video_label.config(text="No se ha seleccionado una carpeta de salida")
        return None 

def process_image_file():
    output_path = select_output_file()
    if output_path and input_image_path:  # Verificar input_image_path
        try:
            model_path = "ruta/al/modelo/unet_model.pth" 
            process_image(input_image_path, output_path, model_path) 
            messagebox.showinfo("Procesamiento Completo", f"Imagen procesada guardada en: {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")
        finally:
            process_button.config(state=tk.DISABLED)


root = tk.Tk()
root.title("Mejora de Calidad de Video con U-Net")

select_input_button = tk.Button(root, text="Seleccionar Imagen de Entrada", command=select_file)
select_input_button.pack(pady=10)

input_video_label = tk.Label(root, text="No se ha seleccionado ningún video de entrada")
input_video_label.pack(pady=5)

select_output_button = tk.Button(root, text="Seleccionar carpeta de salida", command=select_output_file)
select_output_button.pack(pady=10)

output_video_label = tk.Label(root, text="No se ha seleccionado una ubicación para guardar el video")
output_video_label.pack(pady=5)


process_button = tk.Button(root, text="Procesar Imagen", command=process_image_file, state=tk.DISABLED)
process_button.pack(pady=20)


root.mainloop()
