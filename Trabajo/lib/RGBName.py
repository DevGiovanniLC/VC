import json
from typing import Tuple


class RGBName:
    """
    Clase para identificar el nombre más cercano de un color en inglés a partir de valores RGB.
    """

    def __init__(self, color_file: str):
        with open(color_file, "r") as file:
            self.color_data = json.load(file)["colors"] # Carga los colores de un archivo JSON

    @staticmethod
    def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        # Calcula la distancia entre dos colores
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:

        closest_color = None
        min_distance = float("inf")

        for name, color in self.color_data.items():
            distance = self._color_distance(rgb, tuple(color))
            if distance < min_distance:
                min_distance = distance
                closest_color = name

        # Si no se encontró ningún color cercano, devuelve "Unknown"
        return closest_color or "Unknown"
