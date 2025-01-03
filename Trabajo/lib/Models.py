import time
import threading
from lib.ColorDetector import ColorDetector


class ColorModel:
    def __init__(self):
        self.color_detector = ColorDetector()
    
    def get_characteristic(self,list_characteristics, callback):
        color_name, color_rgb = self.color_detector.detect_color(list_characteristics[0])
        print(color_name)
        callback(list_characteristics, 2, color_rgb)


class BreedModel:

    def get_characteristic(list_characteristics, callback):
        time.sleep(1)
        callback(list_characteristics, 3, "DOGE")


class EmotionModel:

    def get_characteristic(list_characteristics, callback):
        time.sleep(1)
        callback(list_characteristics, 4, "HAPPY")


class Models:

    def __init__(self, map):
        self.map = map
        self.pause_event_color = threading.Event()
        self.pause_event_breed = threading.Event()
        self.pause_event_emotion = threading.Event()
        self.color_model = ColorModel()

    def set_characteristics(self, id):
        list_characteristics = self.map[id]

        self.color_model.get_characteristic(list_characteristics, self.set_value_callback)

        BreedModel.get_characteristic(list_characteristics, self.set_value_callback)

        EmotionModel.get_characteristic(list_characteristics, self.set_value_callback)

        list_characteristics[5] += 1

    def set_value_callback(self, list_characteristics, index, value):
        list_characteristics[index] = value
