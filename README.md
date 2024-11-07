# app_unet
 TFG Inteligencia artificial

El funcionamiento para el entrenamiento de la UNET es el siguiente actualmente:

-Las imagenes para el entrenamiento se sacaran desde train_gui.py (deberia de ver de mover la logica a otro lado, al fin y al cabo se supone que es una interfaz)
-El entrenamiento se lanza desde train_unet.py
-Obtiene el modelo del archivo unet_model.py, el cual desde la funcion train se pueden modificar los parametros.
-Se llama a la función get_dataloader, la cual se encargara de preparar las imagenes para el entrenamiento
-Una vez se termine se guarda el modelo "unet_model.pth" (ver si deberia cambiar nombre, o crear diferentes versiones en vez de reescribirlo para luego en el TFG poder hacer comparaciones)
-Una vez tengamos el modelo, desde app_gui se pedira un video para aplicar el modelo y la carpeta de salida, se le aplicara el modelo usando la logica de video_processing (aqui tengo mucho trabajo pendiente, primero termina el modelo)
-el archivo utils.py actualmente no se usa (deberia mover la logica de train_gui ahi? podria ser una opcion )