import pandas as pd


class DataLoader:
    def __init__(self):
        self.data_intelligence = pd.read_csv(
            "Trabajo\\data\\dog_intelligence.csv", index_col=["Breed"]
        )
        self.data_atributtes = pd.read_csv(
            "Trabajo\\data\\dog_attributes.csv", index_col=["Breed"]
        )

    def search(self, breed):
        if breed in self.data_intelligence.index:
            data1 = self.data_intelligence.loc[breed]
        else:
            print(f"El índice '{breed}' no existe en el archivo de inteligencia")

        if breed in self.data_atributtes.index:
            data2 = self.data_atributtes.loc[breed]
        else:
            print(f"El índice '{breed}' no existe en el archivo de atributos")

        return Search(pd.concat([data1, data2], axis=1))

    def get_attributes(self, item):

        search = self.search(item)

        intelligence = (
            search.get_attribute("obey").get_attribute("Classification").get()
        )
        height_range_cm = (
            search.get_attribute("height_low_inches")
            .get_attribute("height_high_inches")
            .get()
        )
        height_range_cm = [float(x) * 2.54 for x in height_range_cm]

        weight_range_kg = (
            search.get_attribute("weight_low_lbs")
            .get_attribute("weight_high_lbs")
            .get()
        )
        weight_range_kg = [float(x) * 0.45359237 for x in weight_range_kg]

        return (
            intelligence,
            str(RangeType(height_range_cm, "cm")),
            str(RangeType(weight_range_kg, "kg")),
        )


class Search:
    def __init__(self, data):
        self.__data = data
        self.__attibuteList = []

    def get_attribute(self, attribute):
        column = self.__data.loc[attribute]
        self.__attibuteList.append(column.dropna().iloc[0])
        return self

    def get(self):
        list = self.__attibuteList.copy()
        self.__attibuteList.clear()
        return tuple(list)


class RangeType:

    def __init__(self, list, units):
        self.list = list
        self.units = units

    def __str__(self):
        return f"({round(self.list[0], 2)} - {round(self.list[1], 2)}) {self.units}"


if __name__ == "__main__":
    loader = DataLoader()
    # print(loader.data_atributtes)
    search = loader.get_attributes("Labrador Retriever")

    print(search)
