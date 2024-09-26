# Práctica 1. Primeros pasos con OpenCV

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
1. [Ajedrez](#ajedrez)
2. [Mondrian](#mondrian)
    - [Sin funciones de dibujo](#mondrian_sin)
    - [Con funciones de dibujo](#mondrian_con)
4. [Modificar un plano de la imagen](#modificar-un-plano-de-la-imagen)
5. [Pintar círculos en posiciones de píxeles claros y oscuros](#pintar-círculos-en-posiciones-de-píxeles-claros-y-oscuros)
    - [Pixel más claro/oscuro](#pixel)
    - [Región 8x8 más clara/oscura](#region)
7. [Crear un efecto Pop Art personalizado](#crear-un-efecto-pop-art-personalizado)
8. [Conclusión](#conclusión)

## Ajedrez <a name="ajedrez"></a>

En este ejercicio, se busca crear una imagen que simule un tablero de ajedrez. Para lograrlo, se genera una imagen de tamaño 800x800 píxeles, donde se alternan cuadros blancos y negros en una cuadrícula. Se usaron ciclos para recorrer la imagen y se estableció que cada cuadro mida 100x100 píxeles. Se alterna el color según la posición de los cuadros en el tablero, replicando el patrón característico de un tablero de ajedrez.

![ajedrez](https://github.com/user-attachments/assets/ac0b7fce-ebc9-47d5-b56f-bad6b86fb666)


## Mondrian <a name="mondrian"></a>

Este ejercicio simula un estilo similar al del artista Piet Mondrian, en el que se utilizan bloques de colores primarios separados por líneas negras. 

![modrian](https://github.com/user-attachments/assets/f2dc2489-50f8-4090-8e4c-4768c4635661)


### Sin funciones de dibujo <a name="mondrian_sin"></a>

Se construyó la imagen utilizando matrices, donde el color y el tamaño de los rectángulos son generados aleatoriamente. Cada rectángulo tiene un ancho aleatorio y el alto varía para cada fila. Para simular las líneas negras, se dejó un pequeño espacio entre cada bloque de color, creando el efecto característico de las divisiones negras que se observan en las obras de Mondrian.



### Con funciones de dibujo <a name="mondrian_con"></a>

Este ejercicio sigue la misma estrategia que el ejercicio anterior pero para dibujar los rectangulos se utiliza las funciones de dibujo de OpenCV.


## Modificar un plano de la imagen <a name="modificar-un-plano-de-la-imagen"></a>

En este ejercicio, se toma la imagen generada en el estilo Mondrian, imagen_2.png, y se modifica uno de los planos de color (en este caso, el plano rojo). La finalidad es mostrar cómo se puede cambiar individualmente un canal de color de la imagen, alterando su apariencia de manera significativa. El plano rojo se asigna a un valor aleatorio, lo que afecta directamente el resultado final de la imagen, dándole un efecto visual diferente cada vez que se ejecuta el código.
![color](https://github.com/user-attachments/assets/42c7e396-a80d-4008-ae6f-b8cd9ef2e566)

## Pintar círculos en posiciones de píxeles claros y oscuros <a name="pintar-círculos-en-posiciones-de-píxeles-claros-y-oscuros"></a>

El objetivo de este ejercicio es encontrar los píxeles más claros y más oscuros de una imagen, y marcar su posición con círculos de diferentes colores.

### Pixel más claro/oscuro <a name="pixel"></a>

Primero se convierte la imagen a escala de grises, lo que permite identificar de forma más sencilla el píxel más brillante y el más oscuro. Una vez localizadas las posiciones de estos píxeles, se dibujan círculos en esos puntos de la imagen original. Este enfoque permite destacar visualmente los extremos de brillo y oscuridad en una imagen.

![image](https://github.com/user-attachments/assets/892f9577-ff67-414b-a6cf-ad9e32842116)


### Región 8x8 más clara/oscura <a name="region"></a>

Este ejercicio expande el anterior al enfocarse no solo en un píxel, sino en una región de 8x8 píxeles. Para lograrlo, se reduce la resolución de la imagen, lo que permite identificar fácilmente las áreas más claras y más oscuras. Una vez localizadas las regiones, se resaltan con círculos de colores en la imagen original, multiplicando las coordenadas por 8 para obtener la ubicación exacta de esas zonas en la imagen de mayor resolución.

![image](https://github.com/user-attachments/assets/2d9206f6-fe12-4238-b403-be81ba6ffd6c)

## Crear un efecto Pop Art personalizado <a name="crear-un-efecto-pop-art-personalizado"></a>

Este ejercicio utiliza la cámara web  en tiempo real para crear un collage en estilo "Pop Art". La pantalla se divide en una cuadrícula de celdas que muestran diferentes versiones del video, aplicando diferentes filtros de color en cada una. La cantidad de celdas se puede aumentar o disminuir con el teclado, apretando la "q" o "e". También al precionar la "w" o "s" se cambia la posición de los filtros.

![image](https://github.com/user-attachments/assets/25290459-7ca8-4f39-aec6-cc9715b254be)


## Conclusión <a name="conclusión"></a>

En esta práctica se exploraron diferentes técnicas de procesamiento y manipulación de imágenes. Se trabajó tanto con patrones geométricos como el ajedrez y Mondrian, como con efectos visuales avanzados en tiempo real, como el Pop Art.
