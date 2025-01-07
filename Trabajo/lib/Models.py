from lib.ColorDetector import ColorDetector
from lib.BreedClassificator import BreedClassificator
from lib.EmotionClassificator import EmotionClassificator
from lib.Enums import EnumDogList


class ColorModel:
    """
    Clase que carga el modelo de detección de color
    """
    
    def __init__(self):
        self.__color_detector = ColorDetector()

    def get_characteristic(self, list_characteristics, callback):
        color_rgb, color_name = self.__color_detector.detect_color(
            list_characteristics[EnumDogList.FRAME.value]
        )
        callback(list_characteristics, EnumDogList.COLOR.value, color_name)


class BreedModel:
    """
    Clase que carga el modelo de detección de raza
    """
    
    def __init__(self):
        self.__breed_detector = BreedClassificator()

    def get_characteristic(self, list_characteristics, callback):
        result = self.__breed_detector.predict_image(list_characteristics[EnumDogList.FRAME.value])
        callback(list_characteristics, EnumDogList.BREED.value, result["class"])


class EmotionModel:
    """
    Clase que carga el modelo de detección de emociones
    """
    def __init__(self):
        self.__emotion_detector = EmotionClassificator()

    def get_characteristic(self, list_characteristics, callback):
        result = self.__emotion_detector.predict_image(list_characteristics[EnumDogList.FRAME.value])
        
        # Agrega el resultado en la lista del track id
        callback(list_characteristics, EnumDogList.EMOTION.value, result["class"])


class Models:
    """
    Clase que contiene los modelos de clasificación y clasificación de características.
    ejecuta la detección de los modelos para la detección de caracterisitcas
    """

    def __init__(self, map):
        self.__map = map
        self.__color_model = ColorModel()
        self.__breed_model = BreedModel()
        self.__emotion_model = EmotionModel()

    def set_characteristics(self, id):
        list_characteristics = self.__map[id]

        self.__color_model.get_characteristic(
            list_characteristics, self.__set_value_callback
        )

        self.__breed_model.get_characteristic(
            list_characteristics, self.__set_value_callback
        )

        self.__emotion_model.get_characteristic(
            list_characteristics, self.__set_value_callback
        )

        list_characteristics[5] += 1

    def __set_value_callback(self, list_characteristics, index, value):
        if value != "Unknown":
            list_characteristics[index] = value
