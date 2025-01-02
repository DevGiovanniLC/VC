import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from  lib.RGBName import RGBName


class ColorDetector:
    def __init__(self):
        self.color_to_name = RGBName('./lib/resources/colors.json')
    
    def detect_color(self, image):
        if image is None or image.size <= 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
            return None,'Unknown'

        # Convertir a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Recortar el centro de la imagen (50% del ancho y alto)
        cropped_image = self.__crop_center(image, width_ratio=0.6, height_ratio=0.6)

        # Obtener color dominante de la región central
        dominant_color = self.__get_dominant_color(cropped_image, k=3)
        color_name = self.color_to_name.get_color_name(dominant_color)
        return  dominant_color, color_name

    def __get_dominant_color(self, image, k=3):
        """Obtiene el color dominante de una imagen usando K-Means."""
        pixels = image.reshape(-1, 3)  # Aplanar la imagen a una lista de píxeles
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return dominant_color


    def __crop_center(self, image, width_ratio=0.1, height_ratio=0.1):
        """
        Recorta la región central de la imagen.
        width_ratio y height_ratio determinan el tamaño del recorte (en proporción a la imagen original).
        """
        h, w, _ = image.shape
        new_w, new_h = int(w * width_ratio), int(h * height_ratio)
        x_start, y_start = (w - new_w) // 2, (h - new_h) // 2
        return image[y_start : y_start + new_h, x_start : x_start + new_w]


if __name__ == "__main__":
    # Ruta de la imagen
    image_path = "./Trabajo/perro.jpg"  # Cambia esto a tu archivo real
    detector = ColorDetector()
    color = detector.detect_color(image_path)
    print(f"Color: {color[1]}:{color[0]}")
