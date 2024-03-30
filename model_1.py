# """
# Created on Wed Mar 27 01:13:05 2024

# @author: ocamp
# """
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Cargar el modelo YOLO
# model = YOLO('best.pt')

# # Realizar la predicción en la imagen
# results = model.predict('1_generada.png', conf=0.5, save=False)

# # Cargar la imagen con OpenCV
# image = cv2.imread('1_generada.png')

# # Variable para almacenar todas las clases juntas
# all_classes = ""

# # Iterar sobre los resultados
# for r in results:
#     boxes = r.boxes
#     class_names = r.names  # Obtener los nombres de las clases
    
#     # Obtener las coordenadas x de los bounding boxes
#     x_coords = [box.xyxy[0][0] for box in boxes]
#     # Ordenar los índices de los bounding boxes en función de las coordenadas x
#     sorted_indices = np.argsort(x_coords)
    
#     # Iterar sobre los índices ordenados
#     for idx in sorted_indices:
#         box = boxes[idx]
#         # Obtener las coordenadas del bounding box
#         b = box.xyxy[0]
#         coord = np.array(b, dtype=int)
        
#         # Obtener la clase predicha
#         class_index = int(box.cls)
#         class_name = class_names[class_index]
        
#         # Concatenar la clase a la cadena
#         all_classes += class_name

#         # Dibujar el bounding box en la imagen
#         cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)  # Color verde, grosor del borde: 2
#         # Agregar etiqueta de clase al cuadro delimitador
#         cv2.putText(image, class_name, (coord[0], coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# # Imprimir todas las clases juntas sin espacios
# print(all_classes)

# # Mostrar la imagen con los bounding boxes y etiquetas de clase
# cv2.imshow('Image with Bounding Boxes and Class Labels', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 01:13:05 2024

@author: ocamp
"""
import os
from ultralytics import YOLO
import cv2
import numpy as np

# Ruta de la carpeta de entrada (imágenes)
input_folder = 'C:/Users/ocamp/Downloads/Data/Data/train/img_h'

# Ruta de la carpeta de salida (archivos de texto con clases)
output_folder = 'C:/Users/ocamp/Downloads/Data/Data/train/etiqueta_img_h'

# Cargar el modelo YOLO
model = YOLO('best.pt')

# Obtener la lista de archivos en la carpeta de entrada
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# Iterar sobre cada archivo de imagen
for image_file in image_files:
    # Ruta completa de la imagen de entrada
    input_image_path = os.path.join(input_folder, image_file)
    
    # Realizar la predicción en la imagen
    results = model.predict(input_image_path, conf=0.5, save=False)
    
    # Variable para almacenar todas las clases juntas
    all_classes = ""

    # Iterar sobre los resultados
    for r in results:
        boxes = r.boxes
        class_names = r.names  # Obtener los nombres de las clases
        
        # Obtener las coordenadas x de los bounding boxes
        x_coords = [box.xyxy[0][0] for box in boxes]
        # Ordenar los índices de los bounding boxes en función de las coordenadas x
        sorted_indices = np.argsort(x_coords)
        
        # Iterar sobre los índices ordenados
        for idx in sorted_indices:
            box = boxes[idx]
            # Obtener la clase predicha
            class_index = int(box.cls)
            class_name = class_names[class_index]
            
            # Concatenar la clase a la cadena
            all_classes += class_name

    # Crear el nombre del archivo de salida (archivo de texto)
    output_text_file = os.path.splitext(image_file)[0] + '.txt'
    output_text_path = os.path.join(output_folder, output_text_file)
    
    # Escribir las clases detectadas en el archivo de texto
    with open(output_text_path, 'w') as file:
        file.write(all_classes)

print("Proceso completado.")
