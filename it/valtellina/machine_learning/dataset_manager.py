import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DatasetManager:
    def __init__(self):
        self.__data = self.clean(self.scarica_dataset())

    def scarica_dataset(self):
        df = pd.read_csv("dataset/Iris.csv")
        return df

    def clean(self, df):
        df['Species'] = to_categorical(df["Species"])
        return df

    def split_data(self, target):
        X = self.__data.drop(columns=target)

        y = self.__data[target]

        # 80 train 20 test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
