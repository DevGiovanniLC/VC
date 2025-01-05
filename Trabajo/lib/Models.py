import time
from lib.ColorDetector import ColorDetector
from lib.BreedClassificator import BreedClassificator
from lib.EmotionClassificator import EmotionClassificator


class ColorModel:
    def __init__(self):
        self.color_detector = ColorDetector()

    def get_characteristic(self, list_characteristics, callback):
        color_name, color_rgb = self.color_detector.detect_color(
            list_characteristics[0]
        )
        print(color_name)
        callback(list_characteristics, 2, color_rgb)


class BreedModel:
    def __init__(self):
        self.breed_detector = BreedClassificator()

    def get_characteristic(self, list_characteristics, callback):
        result = self.breed_detector.predict_image(list_characteristics[0])
        callback(list_characteristics, 3, result["class"])


class EmotionModel:
    def __init__(self):
        self.emotion_detector = EmotionClassificator()

    def get_characteristic(self, list_characteristics, callback):
        result = self.emotion_detector.predict_image(list_characteristics[0])
        callback(list_characteristics, 4, result["class"])


class Models:

    def __init__(self, map):
        self.map = map
        self.color_model = ColorModel()
        self.breed_model = BreedModel()
        self.emotion_model = EmotionModel()

    def set_characteristics(self, id):
        list_characteristics = self.map[id]

        self.color_model.get_characteristic(
            list_characteristics, self.set_value_callback
        )

        self.breed_model.get_characteristic(
            list_characteristics, self.set_value_callback
        )

        self.emotion_model.get_characteristic(
            list_characteristics, self.set_value_callback
        )

        list_characteristics[5] += 1

    def set_value_callback(self, list_characteristics, index, value):
        if value != "Unknown":
            list_characteristics[index] = value
