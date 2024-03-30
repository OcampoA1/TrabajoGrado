# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 01:13:05 2024

@author: ocamp
"""
from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo YOLO
model = YOLO('best.pt')

# Realizar la predicción en la imagen
results = model.predict('1_generada.png', conf=0.5, save=False)

# Cargar la imagen con OpenCV
image = cv2.imread('1_generada.png')

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
        # Obtener las coordenadas del bounding box
        b = box.xyxy[0]
        coord = np.array(b, dtype=int)
        
        # Obtener la clase predicha
        class_index = int(box.cls)
        class_name = class_names[class_index]
        
        # Imprimir la etiqueta de clase
        print(class_name)
        
        # Dibujar el bounding box en la imagen
        cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)  # Color verde, grosor del borde: 2
        # Agregar etiqueta de clase al cuadro delimitador
        cv2.putText(image, class_name, (coord[0], coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostrar la imagen con los bounding boxes y etiquetas de clase
cv2.imshow('Image with Bounding Boxes and Class Labels', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
