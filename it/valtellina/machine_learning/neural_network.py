import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


class NeuralNetwork:
    def __init__(self):
        self.history = None
        self.model = Sequential()






    def nn_model(self, input, output_shape, target):
        input_shape = (input.shape[1],)
        self.model.add(Dense(16, activation='relu', input_shape=input_shape))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(output_shape, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.history = self.model.fit(input, target, validation_split=0.2, epochs=30)


    '''
    confronto relu-tanh: entrambe le funzioni di attivazione permettono al modello
    di raggiungere prestazioni molto elevate sul dataset Iris.
    La tanh converge più velocemente nelle prime epoche,
    mentre ReLU mostra una crescita più lenta ma più stabile nel lungo periodo.
    Con un numero sufficiente di epoche, entrambi i modelli convergono a una performance perfetta,
    indicando che il problema è facilmente apprendibile dal modello considerato.
    '''


    def predict(self, test):
        predictions = self.model.predict(test)

        return predictions

    def plot_loss(self):
        plt.figure()

        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')

        plt.legend()

        fig = plt.gcf()
        plt.show()

        return fig

    def plot_accuracy(self):
        plt.figure()

        plt.plot(self.history.history['accuracy'], label='train accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val accuracy')

        plt.legend()

        fig = plt.gcf()  # prende la figura corrente
        plt.show()

        return fig


