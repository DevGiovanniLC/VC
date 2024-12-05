import asyncio
from abc import ABC, abstractmethod
import time


class  AsyncModel:
    def __init__(self):
        self.executing = False
    
    def is_executing(self):
        return self.executing
    
    @abstractmethod
    async def get_characteristic(self, id, callback):
        pass


class ColorModel(AsyncModel):
    def __init__(self):
        super().__init__()
    
    def get_characteristic(self, list_characteristics, callback):
        self.executing = True
        time.sleep(2) # Simula una tarea de larga duración
        self.executing = False
        callback(list_characteristics, 1,"RED")


class BreedModel(AsyncModel):
    def __init__(self):
        super().__init__()

    def get_characteristic(self, list_characteristics, callback):
        self.executing = True
        time.sleep(5)  # Simula una tarea de larga duración
        self.executing = False
        callback(list_characteristics, 2,"DOGE")


class EmotionModel(AsyncModel) :
    def __init__(self):
        super().__init__()
    
    def get_characteristic(self, list_characteristics, callback):
        self.executing = True
        time.sleep(3)  # Simula una tarea de larga duración
        self.executing = False
        callback(list_characteristics, 3,"HAPPY")



class Models:
    
    def __init__(self, dir):
        self.dir = dir
        self.colorModel = ColorModel()
        self.breedModel = BreedModel()
        self.emotionModel = EmotionModel()
    
    
    def set_characteristics(self, frame, id, score):
        list_characteristics = self.dir.get(id, [0,"","",""])
        self.dir[id] = list_characteristics
        tasks = []
        
        list_characteristics[0] = score
        print("llega")
        
        if not self.colorModel.is_executing():
            tasks.append(self.colorModel.get_characteristic(list_characteristics, self.set_value_callback))
        
        if not self.breedModel.is_executing():
            tasks.append(self.breedModel.get_characteristic(list_characteristics, self.set_value_callback))
        
        if not self.emotionModel.is_executing():
            tasks.append(self.emotionModel.get_characteristic(list_characteristics, self.set_value_callback))
        

    def set_value_callback(self, list_characteristics, index, value):
        list_characteristics[index] = value

