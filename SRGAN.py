import os
import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo generador
generator = load_model('C:/Users/ocamp/Downloads/Data/Data/test/pesos/SRGAN_13000/gen_e_20.h5', compile=False)

# Ruta de la carpeta con imágenes de baja resolución
input_folder = 'C:/Users/ocamp/Downloads/Data/Data/test/test_lr_images'

# Ruta de la carpeta donde se guardarán las imágenes generadas
output_folder = 'C:/Users/ocamp/Downloads/Data/Data/test/imagenes_super_20epochs_13000'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Obtener la lista de archivos en la carpeta de entrada
image_files = os.listdir(input_folder)

# Iterar sobre los archivos de imagen en la carpeta de entrada
for image_file in image_files:
    # Ruta completa de la imagen de baja resolución
    lr_image_path = os.path.join(input_folder, image_file)

    # Cargar la imagen de baja resolución
    lr_image = cv2.imread(lr_image_path)
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)  # Convertir a formato RGB
    lr_image = lr_image / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
    lr_image = np.expand_dims(lr_image, axis=0)  # Expandir la dimensión para tener forma (1, height, width, channels)

    # Generar la imagen de alta resolución
    gen_image = generator.predict(lr_image)

    # Normalizar los valores de píxeles en el rango 0-1
    gen_image = np.squeeze(gen_image)  # Eliminar la dimensión adicional
    gen_image = (gen_image - np.min(gen_image)) / (np.max(gen_image) - np.min(gen_image))

    # Convertir la imagen de alta resolución de vuelta a escala de 0-255
    gen_image = (gen_image * 255).astype(np.uint8)

    # Guardar la imagen generada en la carpeta de salida con el mismo nombre que la imagen de baja resolución
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))

print("Proceso completado. Las imágenes de alta resolución generadas se han guardado en la carpeta 'images_superresolution'.")
