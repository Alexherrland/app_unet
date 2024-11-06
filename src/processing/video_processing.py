# video_processing.py
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from unet_model import UNet

def process_video(input_video_path, output_video_path="output.mp4"):  # Agregar output path como parámetro
    cap = cv2.VideoCapture(input_video_path)  # Abre el video enviado de process_video_file()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) # Usar dimensiones originales, investigar si es posible reescalar de 4:3 a 16:9 sin perder calidad, y ver si hacer antes o despues del modelo
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load('unet_model.pth', map_location=device)) # Cargar modelo en grafica o CPU, falta ver si se usara solo un modelo global o uno para cada mapa, haría falta para eso una forma de detectar el mapa previamente, de momento, intentare un solo modelo
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Redimensionar para el modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_processed = apply_unet_to_frame(frame, model, transform, device) # Pasar transformaciones y device
        out.write(frame_processed)

    cap.release()
    out.release()
    return output_video_path


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
