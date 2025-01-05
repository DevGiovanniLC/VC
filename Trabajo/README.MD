# Trabajo Final. Detección y seguimiento de perros (Raza, emociones, tamaño promedio, inteligencia, etc.)

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Elena%20Morales%20Gil-brightgreen?style=flat-square&logo=github)](https://github.com/ElenaMoralesGil)

[![GitHub](https://img.shields.io/badge/GitHub-Giovanni%20León%20Corujo-yellow?style=flat-square&logo=github)](https://github.com/DevGiovanniLC)
## Tecnologías
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)

---

## Descripción
El objetivo del proyecto es el de crear un sistema de detección y seguimiento de perros en un video, con la finalidad de poder identificar a los perros, obteniendo su color, su raza, su tamaño, su estado emocional.

El proyecto se ha adaptado para poder hacer esa identificación en tiempo real, gracias al uso de hilos y colas. Además, se ha usado varias optimizaciones para que el proceso de detección sea más rápido y eficiente.

## Funcionalidades
Las funcionalidades de detección son las siguientes:

* **Seguimiento de perros**: El sistema de seguimiento de perros se encarga de detectar y seguir a uno o varios perros en el video, esto se consigue gracias al uso de *YOLOv11*.

* **Detección de color**: El sistema de detección de color no es del todo preciso pero puede ser eficaz en base a la raza detectada, ya que la mayoría de los perros tienen un color dominante. Y de esta manera se puede identificar a los perros. Así funciona el sistema de detección de color, utilizando K-Means para obtener el color dominante de la región central de la imagen anteriormente recortada por yolo centrando al perro.

* **Detección de razas de perros**: El sistema de detección raza de perros se se realiza primeramente detenctando al perro en el video gracias a *YOLOv11*, y luego utilizando una red neuronal entrenada para poder identificar a los perros, *ResNet50*.

* **Detección de estado emocional**: El sistema de detección de estado emocional se realiza primeramente detenctando al perro en el video gracias a *YOLOv11* y posteriormente utilizando una red neuronal entrenada, *ResNet50* para identifica su estado emocional en base a una imagen.

* **Detección de tamaño**: El sistema de detección de tamaño se realiza primeramente detenctando al perro en el video gracias a *YOLOv11* y detectando la raza del perro a partir de la raza dell perro se puede obtener estimaciones de su tamaño promedio.

---

## Estructura del proyecto
El proyecto está dividido en diferentes scripts, los cuales se encargan de realizar diferentes tareas, empaquetando dichas funcionalidades en clases. Esto se hace, para poder subdividir el código y que sea más sencillo de visualizar y entender. Todos los scripts se encuentran en la carpeta `lib`. Todos ellos se usan entre si o se usa en el archivo `trabajo_final.ipynb`.


## Entrenamiento de modelos Resnet50


## Uso
El proyecto se puede ejecutar con el archivo `trabajo_final.ipynb` este ejecutara todos los scripts y realizará la detección y seguimiento de los perros en el video, mostrando un rectangulo donde ubica cada perro en el video además de mostrar su raza, estado emocional, color y tamaño.

---

## Adversidades tecnicas encontradas durante el desarrollo
Al hacer uso del modelo de yolo, se ha encontrado que el modelo no es muy preciso en la detección de perros, tiende a confundir constantemente a perros con vacas, asi que si en algún momento se detecta a un perro pero posteriormente se identifica como una vaca, se considera que sigue siendo un perro y se realizara el seguimiento del mismo.