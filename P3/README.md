# Práctica 3. Detección y reconocimiento de formas

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Elena%20Morales%20Gil-brightgreen?style=flat-square&logo=github)](https://github.com/ElenaMoralesGil)

[![GitHub](https://img.shields.io/badge/GitHub-Giovanni%20León%20Corujo-yellow?style=flat-square&logo=github)](https://github.com/DevGiovanniLC)

## Tecnologias
  -  Python: ![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)

## Bibliotecas

El proyecto utiliza las siguientes bibliotecas para procesamiento de imágenes, análisis de datos y visualización:

- OpenCV: ![OpenCV](https://img.shields.io/badge/OpenCV-Latest-brightgreen?style=flat-square&logo=opencv)
  - **Importación**: 
    ```python
    import cv2
    ```
  - Descripción: Biblioteca de visión por computadora utilizada para la manipulación de imágenes y detección de características como contornos y círculos.

- NumPy: ![NumPy](https://img.shields.io/badge/NumPy-Latest-blueviolet?style=flat-square&logo=numpy)
  - **Importación**: 
    ```python
    import numpy as np
    ```
  - Descripción: Utilizada para operaciones matemáticas y manipulación de matrices, como la conversión y manejo de datos de imágenes.

- Matplotlib: ![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-yellow?style=flat-square&logo=matplotlib)
  - **Importación**:
    ```python
    import matplotlib.pyplot as plt
    ```
  - Descripción: Biblioteca de visualización gráfica usada para mostrar imágenes, gráficas y resultados de análisis.

- Pandas: ![Pandas](https://img.shields.io/badge/Pandas-Latest-orange?style=flat-square&logo=pandas)
  - **Importación**:
    ```python
    import pandas as pd
    ```
  - Descripción: Librería poderosa para la manipulación de datos en estructuras como DataFrames, útil para manejar conjuntos de datos tabulares.

- Scikit-learn: ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-blue?style=flat-square&logo=scikitlearn)
  - **Importación**:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report
    ```
  - Descripción: Conjunto de herramientas para el aprendizaje automático y minería de datos. En este proyecto se utiliza para clasificar imágenes mediante el modelo **Random Forest** y evaluar su rendimiento.

- Seaborn: ![Seaborn](https://img.shields.io/badge/Seaborn-Latest-cyan?style=flat-square&logo=seaborn)
  - **Importación**:
    ```python
    import seaborn as sns
    ```
  - Descripción: Biblioteca basada en Matplotlib, especializada en la creación de visualizaciones estadísticas atractivas, como matrices de confusión.

Estas bibliotecas son esenciales para procesar las imágenes, extraer características, construir modelos de clasificación y visualizar los resultados en este proyecto.


## Índice
1. [Introducción](#introducción)
2. [Tarea 1: Detección y Clasificación de Monedas Europeas](#tarea-1-detección-y-clasificación-de-monedas-europeas)
3. [Tarea 2: Clasificación de Imágenes con Random Forest](#tarea-2-clasificación-de-imágenes-con-random-forest)

## Introducción
Este proyecto se divide en dos tareas principales:

1. **Clasificación y detección de monedas**: Utilizando técnicas de procesamiento de imágenes, se identifica el valor de diferentes monedas presentes en una imagen, basándose en el radio y color de las mismas.
2. **Clasificación de tipos de materiales**: Se clasifican fragmentos, pellets y residuos de alquitrán (tar) utilizando un clasificador Random Forest. Para ello, se extraen características geométricas y texturales de los contornos de los objetos en las imágenes.


## Tarea 1: Detección y Clasificación de Monedas Europeas

Se aplica el algoritmo de Hough Circle Transform para detectar las monedas en la imagen. Esta técnica permite encontrar objetos circulares dentro de una imagen, filtrando el ruido mediante un desenfoque gaussiano.

Los círculos detectados representan las monedas, y cada círculo es dibujado sobre la imagen.

El usuario debe hacer clic en una moneda de 1 euro para establecer una escala de referencia. Esto permite relacionar los radios de las monedas detectadas con su tamaño real en milímetros.

Se analiza el color promedio de cada moneda dentro de un radio ajustado para identificar su tipo. Dependiendo del valor de tono en el espacio de color HSV, se determinan las características cromáticas de la moneda (tonos de cobre, oro, y plata).

Con la escala y el color promedio, el sistema clasifica la moneda haciendo una comparación con un diccionario de valores de referencia de las monedas europeas.

También se calcula el valor total de la suma de todas las monedas que hay en la imagen una vez que se consiga la escala.  Se muestra en la consola. 
![image](https://github.com/user-attachments/assets/406fce30-579e-434c-a4f1-e1434f5d536b)


## Tarea 2: Clasificación de Imágenes con Random Forest

Se cargan imágenes de tres categorías: fragmentos, pellets, y tar.
Se convierte cada imagen a escala de grises, se aplica un desenfoque gaussiano para eliminar el ruido, y luego se utiliza el método de umbralización de Otsu para segmentar las imágenes.

Para cada contorno detectado en las imágenes segmentadas, se extraen varias características geométricas:

- Área
- Perímetro
- Compacidad
- Relación de aspecto (aspect ratio)
- Relación entre ejes mayor y menor de la elipse ajustada
- Solidez (relación entre el área del contorno y el área de la envolvente convexa)

Además, se utiliza el histograma de la imagen en escala de grises para obtener características de textura, que se reducen a 32 bins.

Se utilizan técnicas de aumento de datos para mejorar la generalización del modelo, como:
- Inversión horizontal
- Rotación de la imagen
- Ajuste de brillo
- Desenfoque gaussiano

Se utiliza un RandomForestClassifier para entrenar el modelo con las características extraídas.
Se dividen los datos en conjuntos de entrenamiento y prueba (70% y 30% respectivamente), y se entrena el modelo con 100 árboles de decisión.

Para la evaluación se genera una matriz de confusión y un reporte de clasificación para evaluar el rendimiento del modelo. También se visualiza la matriz de confusión usando una gráfica de calor (heatmap), y se imprimen las métricas de precisión, recall, y F1-score.

Estos son los resultados:
![alt text](image.png)
![alt text](image-1.png)

Nos pareció que el porcentaje de acierto de 95% era suficiente. 
