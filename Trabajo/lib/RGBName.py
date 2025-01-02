import json
from typing import Tuple


class RGBName:
    """
    Clase para identificar el nombre más cercano de un color en inglés a partir de valores RGB.
    """
    
    def __init__(self, color_file: str):
        """
        Inicializa la clase cargando los colores desde un archivo JSON.
        
        Args:
            color_file (str): Ruta al archivo JSON con los colores.
        """
        with open(color_file, 'r') as file:
            self.color_data = json.load(file)["colors"]
    
    @staticmethod
    def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        """
        Calcula la distancia entre dos colores RGB.
        
        Args:
            c1 (Tuple[int, int, int]): Primer color en formato RGB.
            c2 (Tuple[int, int, int]): Segundo color en formato RGB.
        
        Returns:
            float: Distancia ponderada entre los colores.
        """
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Obtiene el nombre del color más cercano al RGB proporcionado.
        
        Args:
            rgb (Tuple[int, int, int]): Color en formato RGB (0-255).
        
        Returns:
            str: Nombre del color más cercano.
        """
        closest_color = None
        min_distance = float('inf')
        
        for name, color in self.color_data.items():
            distance = self._color_distance(rgb, tuple(color))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return closest_color or "Unknown"


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el archivo de colores
    rgb_name = RGBName("colors.json")
    
    # Probar con un color
    rgb_color = (194, 160, 127)  # Aproximación del color marrón pastel
    print("RGB:", rgb_color)
    print("Color más cercano:", rgb_name.get_color_name(rgb_color))