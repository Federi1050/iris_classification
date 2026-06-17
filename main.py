import numpy as np
from sklearn.metrics import accuracy_score

from it.valtellina.interfaccia_utente.flask_manager import FlaskManager
from it.valtellina.machine_learning.dataset_manager import DatasetManager
from it.valtellina.machine_learning.neural_network import NeuralNetwork

testing = False

if testing:
    dataset_mg = DatasetManager()
    #print("prima")
    #print(dataset_mg.get_data().head())
    dataset_mg.split_data('Species')
    #print(dataset_mg.get_X_train().head())
    dataset_mg.scaling()
    NN = NeuralNetwork()
    #NN.nn_model(dataset_mg.get_X_train(), 3,dataset_mg.get_y_train())
    y_pred = NN.predict(dataset_mg.get_X_test())
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(dataset_mg.get_y_test(), axis=1)
    print("Accuracy: ", accuracy_score(y_test_class, y_pred_class))
    NN.plot_accuracy()
    NN.plot_loss()
else:
    # flask
    app = FlaskManager()
    app.run(host='0.0.0.0', port=5000, debug=True)