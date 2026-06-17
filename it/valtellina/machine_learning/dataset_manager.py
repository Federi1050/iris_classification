import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class DatasetManager:
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.__data = self.scarica_dataset()

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_data(self):
        return self.__data



    @staticmethod
    def scarica_dataset():
        df = pd.read_csv("dataset/Iris.csv")
        return df

    @staticmethod
    def target_encoding(y):
        le = LabelEncoder()
        return le.fit_transform(y)



    def split_data(self, target):
        X = self.__data.drop(columns=target)
        y = self.__data[target]

        # label encoding
        y = self.target_encoding(y)

        # 80 train 20 test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        # one hot encoding
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def scaling(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)