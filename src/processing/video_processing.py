# video_processing.py
import cv2
import os  
import torch
import torchvision.transforms as transforms
import numpy as np
from unet_model import UNet
from tqdm import tqdm

def process_video(input_video_path, output_video_path="output.mp4"):  # Agregar output path como parámetro

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_video_path = os.path.join(script_dir, "video_procesado.mp4")

    print(f"Ruta de entrada: {input_video_path}")
    print(f"Ruta de salida: {output_video_path}")
    print(f"Ruta de salida absoluta: {os.path.abspath(output_video_path)}")

    cap = cv2.VideoCapture(input_video_path)  # Abre el video enviado de process_video_file()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) # Usar dimensiones originales, investigar si es posible reescalar de 4:3 a 16:9 sin perder calidad, y ver si hacer antes o despues del modelo
    
    if not out.isOpened():
        print("Error: No se pudo crear el archivo de salida")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load('models/unet_model_dust2_v1.pth', map_location=device)) # Cargar modelo en grafica o CPU, falta ver si se usara solo un modelo global o uno para cada mapa, haría falta para eso una forma de detectar el mapa previamente, de momento, intentare un solo modelo
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Redimensionar para el modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pbar = tqdm(total=total_frames, desc="Procesando video")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_processed = apply_unet_to_frame(frame, model, transform, device) # Pasar transformaciones y device
        out.write(frame_processed)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    if os.path.exists(output_video_path):
        return output_video_path
    else:
        print("Error: El archivo de salida no se creó correctamente")
        return None


def apply_unet_to_frame(frame, model, transform, device): # Recibir transformaciones y device
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frame_tensor)

    output = output.squeeze(0).permute(1, 2, 0)
    output = torch.clamp(output, 0, 1)
    output = output.cpu().detach().numpy() # Usar detach() para evitar problemas de memoria
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output = cv2.resize(output, (frame.shape[1], frame.shape[0])) # Redimensionar de vuelta al tamaño original

    return output
