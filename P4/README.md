# Práctica 4. Reconocimiento de matrículas

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Elena%20Morales%20Gil-brightgreen?style=flat-square&logo=github)](https://github.com/ElenaMoralesGil)

[![GitHub](https://img.shields.io/badge/GitHub-Giovanni%20León%20Corujo-yellow?style=flat-square&logo=github)](https://github.com/DevGiovanniLC)

---
## Tecnologias
  -  Python: ![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)

## Bibliotecas Utilizadas

- **OS**: ![OS](https://img.shields.io/badge/OS-Latest-lightgray?style=flat-square)
  - **Importación**: 
    ```python
    import os
    ```
  - **Descripción**: Biblioteca que proporciona funciones para interactuar con el sistema operativo, como la manipulación de archivos y directorios.

- **Torch**: ![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=flat-square&logo=pytorch)
  - **Importación**: 
    ```python
    import torch
    ```
  - **Descripción**: Biblioteca de aprendizaje profundo que proporciona herramientas para construir y entrenar modelos de redes neuronales.

- **OpenCV**: ![OpenCV](https://img.shields.io/badge/OpenCV-Latest-brightgreen?style=flat-square&logo=opencv)
  - **Importación**: 
    ```python
    import cv2
    ```
  - **Descripción**: Biblioteca de visión por computadora utilizada para la manipulación de imágenes y detección de características como contornos y círculos.

- **EasyOCR**: ![EasyOCR](https://img.shields.io/badge/EasyOCR-Latest-ff69b4?style=flat-square)
  - **Importación**: 
    ```python
    import easyocr
    ```
  - **Descripción**: Biblioteca de reconocimiento óptico de caracteres (OCR) que permite extraer texto de imágenes.

- **CSV**: ![CSV](https://img.shields.io/badge/CSV-Latest-lightblue?style=flat-square)
  - **Importación**: 
    ```python
    import csv
    ```
  - **Descripción**: Módulo para leer y escribir archivos en formato CSV (Comma-Separated Values).

- **Logging**: ![Logging](https://img.shields.io/badge/Logging-Latest-yellowgreen?style=flat-square)
  - **Importación**: 
    ```python
    import logging
    ```
  - **Descripción**: Biblioteca que proporciona funcionalidades para el registro de mensajes y eventos en aplicaciones Python.

- **Ultralytics**: ![Ultralytics](https://img.shields.io/badge/Ultralytics-Latest-blue?style=flat-square)
  - **Importación**: 
    ```python
    from ultralytics import YOLO
    ```
  - **Descripción**: Implementación de YOLO (You Only Look Once) para la detección de objetos en imágenes y videos.

- **Pandas**: ![Pandas](https://img.shields.io/badge/Pandas-Latest-orange?style=flat-square&logo=pandas)
  - **Importación**:
    ```python
    import pandas as pd
    ```
  - **Descripción**: Librería poderosa para la manipulación de datos en estructuras como DataFrames, útil para manejar conjuntos de datos tabulares.

- **Matplotlib**: ![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-yellow?style=flat-square&logo=matplotlib)
  - **Importación**:
    ```python
    import matplotlib.pyplot as plt
    ```
  - **Descripción**: Biblioteca de visualización gráfica usada para mostrar imágenes, gráficas y resultados de análisis.

- **PIL (Pillow)**: ![Pillow](https://img.shields.io/badge/Pillow-Latest-red?style=flat-square&logo=pillow)
  - **Importación**:
    ```python
    from PIL import Image
    ```
  - **Descripción**: Biblioteca para la apertura, manipulación y guardado de imágenes en varios formatos.

- **Glob**: ![Glob](https://img.shields.io/badge/Glob-Latest-lightgreen?style=flat-square)
  - **Importación**:
    ```python
    import glob
    ```
  - **Descripción**: Módulo que proporciona una función para encontrar todos los nombres de ruta que coinciden con un patrón específico.

---

## Índice
1. [Introducción](#introducción)
3. [Detector de matrículas (YOLO)](#detector-de-matrículas-yolo)
4. [Anonimización de transeúntes y vehículos](#anonimización-de-transeúntes-y-vehículos)
2. [Identificación de texto (OCR)](#identificación-de-texto-ocr)
5. [Detección de dirección de transeúntes y vehículos](#detección-de-dirección-de-transeúntes-y-vehículos)


## Introducción
Este proyecto se divide en 4 elementos que se unifican para dar un resultado conjunto, dos de ellas son optativas, Anonimización y detección de dirección. Las tras dos necesarias para completar la práctica Modelo de detección de matrículas y el identificador de texto (OCR).

# Detector de matrículas (YOLO)

# Anonimización de transeúntes y vehículos

# Identificación de texto (OCR)
El objetivo es detectar y extraer texto de imágenes, como matrículas de vehículos, utilizando la biblioteca EasyOCR y técnicas de procesamiento de imágenes con OpenCV. Intentando tener la mayor
probabilidad de detección.

### 1. Función mostrar_imagen(imagen):*
Esta función recibe una imagen y la muestra utilizando Matplotlib. Convierte la imagen de BGR (formato usado por OpenCV) a RGB para su correcta visualización.

### 2. Función preprocesar_imagen(imagen):
Convierte la imagen a escala de grises, aplica desenfoque y umbralización para obtener una imagen binaria que facilita la detección de texto.

### 3. Función OCR(imagen):
Función principal que toma una imagen, la preprocesa y la pasa a través de la función de detección. Devuelve el resultado final del OCR.

### 4. Función procesar_deteccion(imagen_procesada):
Esta función ejecuta un ciclo para mejorar la detección de texto en la imagen procesada, actualizando el texto y la probabilidad de detección hasta que no haya mejoras.

### 5. Función postprocesar_imagen(imagen):
Realiza el post-procesamiento de la imagen, aplicando un desenfoque gaussiano y ajustando el contraste. Se asegura de que la imagen tenga dimensiones adecuadas.

### 6. Función detectar_texto(imagen_procesada):
Utiliza EasyOCR para detectar texto en la imagen procesada. Retorna la región de interés (ROI) donde se ha detectado texto, junto con el texto y su probabilidad de detección. Si hay varias de detecciones intenta abarcar todo el area de la imagen.


Posteriormente a la hora de la detección se le da prioridad de que la longitud del texto sea la adecuada, aunque la probabilidad de la imagen sea peor. Para optimizar que el resultado sea el más cercano posible al texto de la matrícula.

![alt text](image.png)


# Detección de dirección de transeúntes y vehículos
Toda la detección de la dirección se realiza en la siguiente función:
### Función detectar_direccion
La función detectar_direccion tiene como objetivo clasificar la dirección de objetos detectados en función de su posición en el fotograma. Los objetos detectados, que pueden ser personas, bicicletas o vehículos, se clasifican en dos categorías según su dirección aparente:

* "from_front": Indica que el objeto se está acercando desde el frente.
* "to_front": Indica que el objeto se está alejando o moviendo hacia el frente de la cámara.
Parámetros
* track_id (int): Identificador único del objeto en seguimiento. Cada objeto detectado tiene un track_id para diferenciarlo de otros.
label_name (str): Nombre de la clase del objeto (p. ej., "person", "bicycle", "car").
* x (int): Coordenada X del borde derecho del objeto en el fotograma.
frame (np.array): Imagen del fotograma actual, que se utiliza para obtener el ancho del fotograma.
Funcionamiento

#### Comprobación del estado del track_id: 

La función verifica si el track_id del objeto ya ha sido clasificado previamente como "from_front" o "to_front". Para ello, recorre todas las entradas en el diccionario datos. Si el track_id ya está en cualquiera de estas listas, se establece id_inside como True y no se realiza ninguna clasificación adicional para ese objeto en este fotograma.

#### Clasificación según la posición en el fotograma:

Si el track_id no ha sido clasificado aún (id_inside es False), se procede a la clasificación basada en el tipo de objeto (label_name) y su coordenada X (x):
Para personas y bicicletas: Se considera que vienen "de frente" (to_front) si están ubicadas en el 20% más cercano del fotograma a la izquierda o en el rango entre el 70% y el 95% del ancho del fotograma en la derecha. En caso contrario, se clasifican como "from_front".

Para otros objetos (vehículos): Los vehículos se clasifican como "from_front" si su coordenada X está en el 70% de la parte izquierda del fotograma, o como "to_front" si están en el 10% de la derecha (es decir, más allá del 90% del ancho del fotograma).
Almacenamiento en el diccionario datos: Según la clasificación, el track_id se añade a la lista correspondiente (from_front o to_front) dentro de la categoría especificada en label_name.

En el conteo final solo tiene los vehiculos en circulación no contabiliza los que están aparcados

![alt text](conteo.jpeg)






[![Ver video](image-1.png)](https://drive.google.com/file/d/1DAhQNVXcXT-vgKi823JkiUI-vVsC4umz/view?usp=drive_link)