# Práctica 2. Funciones básicas de OpenCV

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Elena%20Morales%20Gil-brightgreen?style=flat-square&logo=github)](https://github.com/ElenaMoralesGil)

[![GitHub](https://img.shields.io/badge/GitHub-Giovanni%20León%20Corujo-yellow?style=flat-square&logo=github)](https://github.com/DevGiovanniLC)

## Tecnologias
  -  Python: ![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)

## Librerias 
  - OpenCV: ![OpenCV](https://img.shields.io/badge/OpenCV-Latest-brightgreen?style=flat-square&logo=opencv)
  - Matplotlib: ![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-yellow?style=flat-square&logo=matplotlib)
  - NumPy: ![NumPy](https://img.shields.io/badge/NumPy-Latest-blueviolet?style=flat-square&logo=numpy)


## Índice
1. [Histograma pixeles blancos por filas](#histfilas)
2. [Umbralizado de la imagen](#umbralizado)
4. [Pop art con detección de bordes](#popart)
5. [Detección de movimiento](#movimiento)


## Histograma pixeles blancos por filas <a name="histfilas"></a>

Este ejercicio consiste en:
-   Realizar la cuenta de píxeles blancos por filas. 
-   Determinar el máximo para filas y columna
-   Mostrar el número de filas con un número de píxeles blancos mayor o igual que 0.95\*máximo.

Para contar los pixeles 
## Umbralizado de la imagen <a name="umbralizado"></a>

   Se ha aplicado umbralizado a la imagen resultante de Sobel, y posteriormente se ha realizado el conteo por filas y columnas. Se han calculado los máximos por filas y columnas, y determinado las filas y columnas por encima del 0.95\*máximo. Además se ha remarcado esas filas dentro de la imagen original con filas azules y en el histograma con filas rojas.
   
![filas](https://github.com/user-attachments/assets/febb7b14-da13-4f2e-bc0f-b98e945581b0)

![columnas](https://github.com/user-attachments/assets/16275ede-c39a-426e-87d7-683016a5e188)

## Pop art con detección de bordes <a name="popart"></a>

Al pop art de la práctica anterior se le ha agregado detección de bordes, para que sea más estético. Manteniendo la interactividad que ya tenia con las teclas "q" y "e", para aumentar o reducir el número de marcos y con "w" y "s", para modíficar el filtro del marco.

![image](https://github.com/user-attachments/assets/79b80673-89b6-47a5-8422-1be1d79912d4)


## Detección de movimiento <a name="movimiento"></a>

