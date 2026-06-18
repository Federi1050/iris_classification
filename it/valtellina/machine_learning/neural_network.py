import numpy as np
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, scaler):
        self.history = None
        self.model = Sequential()
        self.scaler = scaler

    def grid_search_cv(self, X, y, activations, optimizers, n_splits=5, epochs=50):

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        all_results = []

        for activation in activations:
            for optimizer in optimizers:

                fold_accuracies = []

                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # scaling
                    scaler = StandardScaler()

                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                    # one hot encoding
                    y_train_cat = to_categorical(y_train)
                    y_val_cat = to_categorical(y_val)

                    # modello
                    model = Sequential()
                    model.add(Input(shape=(X.shape[1],)))
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(16, activation=activation))
                    model.add(Dense(y_train_cat.shape[1], activation='softmax'))

                    model.compile(
                        optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )

                    model.fit(X_train, y_train_cat, epochs=epochs, verbose=0)

                    loss, acc = model.evaluate(X_val, y_val_cat, verbose=0)
                    fold_accuracies.append(acc)

                mean_acc = np.mean(fold_accuracies)
                std_acc = np.std(fold_accuracies)

                all_results.append({
                    "activation": activation,
                    "optimizer": optimizer,
                    "mean_accuracy": mean_acc,
                    "std": std_acc
                })

                print(f"{activation} + {optimizer} → {mean_acc:.4f} ± {std_acc:.4f}")

        # ordina dal migliore al peggiore
        all_results = sorted(all_results, key=lambda x: x["mean_accuracy"], reverse=True)

        #self.grid_results = all_results

        print("\n BEST MODEL:")
        print(all_results[0])

        return all_results

    def nn_model(self, input, output_shape, target):
        input_shape = (input.shape[1],)
        self.model.add(Dense(16, activation='relu', input_shape=input_shape))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(output_shape, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.history = self.model.fit(input, target, validation_split=0.2, epochs=100)

    '''
    confronto relu-tanh: entrambe le funzioni di attivazione permettono al modello
    di raggiungere prestazioni molto elevate sul dataset Iris.
    La tanh converge più velocemente nelle prime epoche,
    mentre ReLU mostra una crescita più lenta ma più stabile nel lungo periodo.
    Con un numero sufficiente di epoche, entrambi i modelli convergono a una performance perfetta,
    indicando che il problema è facilmente apprendibile dal modello considerato.
    '''

    def predict(self, test):
        '''
        class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
        input_scaled = self.scaler.transform(test)
        probabilities = self.model.predict(input_scaled, verbose=0)
        results = []
        for prob in probabilities:
            probs_dict = {
                class_names[i]: round(float(prob[i]), 4)
                for i in range(len(class_names))
            }

            predicted_class = class_names[np.argmax(prob)]

            results.append({
                "predicted_class": predicted_class,
                "probabilities": probs_dict
            })
            '''
        results = self.model.predict(test)

        return results

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


