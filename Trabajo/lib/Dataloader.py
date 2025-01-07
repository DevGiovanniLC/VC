import pandas as pd


class DataLoader:
    """
    Clase para la cargar de datos de CSVs de inteligencia y atributos y para realizar búsquedas
    """
    
    def __init__(self):
        self.data_intelligence = pd.read_csv(
            "./data/dog_intelligence.csv", index_col=["Breed"]
        )
        self.data_atributtes = pd.read_csv(
            "./data/dog_attributes.csv", index_col=["Breed"]
        )

    def search(self, breed):
        if breed in self.data_intelligence.index:
            data1 = self.data_intelligence.loc[breed]
        else:
            print(f"El índice '{breed}' no existe en el archivo de inteligencia")
            data1 = pd.DataFrame(columns=self.data_intelligence.columns, index=[breed])

        if breed in self.data_atributtes.index:
            data2 = self.data_atributtes.loc[breed]
        else:
            print(f"El índice '{breed}' no existe en el archivo de atributos")
            data2 = pd.DataFrame(columns=self.data_atributtes.columns, index=[breed])

        return Search(pd.concat([data1, data2], axis=1))

    def get_attributes(self, item):

        search = self.search(item)

        intelligence = (
            search.get_attribute("obey").get_attribute("Classification").get()
        )
        
        height_range_cm = (search
            .get_attribute("height_low_inches")
            .get_attribute("height_high_inches")
            .get()
        )
        if height_range_cm[0] != "Unknown":
            height_range_cm = [float(x) * 2.54 for x in height_range_cm]

        weight_range_kg = (search
            .get_attribute("weight_low_lbs")
            .get_attribute("weight_high_lbs")
            .get()
        )
        
        if weight_range_kg[0] != "Unknown":
            weight_range_kg = [float(x) * 0.45359237 for x in weight_range_kg]

        return (
            intelligence,
            str(RangeType(height_range_cm, "cm")),
            str(RangeType(weight_range_kg, "kg")),
        )


class Search:
    """
    Clase para realizar búsquedas en el archivo de datos
    """
    def __init__(self, data):
        self.__data = data
        self.__attibuteList = []

    def get_attribute(self, attribute):
        try:
            column = self.__data.loc[attribute]
            self.__attibuteList.append(column.dropna().iloc[0])
        except KeyError:
            self.__attibuteList.append("Unknown")
        return self

    def get(self):
        list = self.__attibuteList.copy()
        self.__attibuteList.clear()
        return tuple(list)


class RangeType:
    """
    Clase para representar un rango de valores
    """
    
    def __init__(self, list, units):
        self.list = list
        self.units = units

    def __str__(self):
        
        if self.list[0] == "Unknown":
            return "Unknown"
        
        return f"({round(self.list[0], 2)} - {round(self.list[1], 2)}) {self.units}"

