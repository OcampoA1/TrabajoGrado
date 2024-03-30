import os

# Rutas de las carpetas con los archivos de texto
folder1 = 'C:/Users/ocamp/Downloads/Data/Data/train/etiqueta_images'
folder2 = 'C:/Users/ocamp/Downloads/Data/Data/train/etiqueta_img_h'

# Obtener la lista de archivos en cada carpeta
files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))

# Crear un archivo de texto para registrar las comparaciones
output_file = 'comparaciones.txt'

# Abrir el archivo en modo de escritura
with open(output_file, 'w') as file:
    # Iterar sobre los archivos de ambas carpetas
    for file1, file2 in zip(files1, files2):
        # Escribir en el archivo los nombres de los archivos que se están comparando
        file.write(f'{file1} - {file2}\n')

# Contador para el total de caracteres correctos
total_correct_chars = 0
# Contador para el total de caracteres en la referencia
total_reference_chars = 0

# Iterar sobre los archivos de ambas carpetas
for file1, file2 in zip(files1, files2):
    # Rutas completas de los archivos
    path1 = os.path.join(folder1, file1)
    path2 = os.path.join(folder2, file2)

    # Leer los contenidos de los archivos
    with open(path1, 'r') as f1, open(path2, 'r') as f2:
        content1 = f1.read().strip()
        content2 = f2.read().strip()

    # Calcular la cantidad de caracteres correctos
    correct_chars = sum(1 for c1, c2 in zip(content1, content2) if c1 == c2)
    # Calcular la cantidad total de caracteres en la referencia
    total_reference_chars += len(content1)
    # Sumar los caracteres correctos
    total_correct_chars += correct_chars

# Calcular la precisión
accuracy = (total_correct_chars / total_reference_chars) * 100

# Imprimir el resultado
print(f'La precisión entre las carpetas es: {accuracy:.2f}%')
