import keras
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.models import Sequential

class NeuralNetwork:
    def __init__(self):
        self.model = Sequential()