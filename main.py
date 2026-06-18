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
    #dataset_mg.split_data('Species')
    #print(dataset_mg.get_X_train().head())
    #dataset_mg.scaling()

    NN = NeuralNetwork(None)

    activations = ['relu', 'tanh']
    optimizers = ['adam', 'sgd', 'RMSprop']

    data = dataset_mg.get_data()

    #preparazione grid
    X = data.drop(columns=["Species"])
    y = dataset_mg.target_encoding(data["Species"])

    results = NN.grid_search_cv(X, y, activations, optimizers)

    for r in results:
        print(r)

    '''
    risultati gridsearch 50 epochs:
    BEST MODEL:
    {'activation': 'tanh', 'optimizer': 'RMSprop', 'mean_accuracy': np.float64(0.9333333134651184), 'std': np.float64(0.05962848250986275)}
    {'activation': 'tanh', 'optimizer': 'RMSprop', 'mean_accuracy': np.float64(0.9333333134651184), 'std': np.float64(0.05962848250986275)}
    {'activation': 'tanh', 'optimizer': 'adam', 'mean_accuracy': np.float64(0.899999988079071), 'std': np.float64(0.04714044926931553)}
    {'activation': 'relu', 'optimizer': 'adam', 'mean_accuracy': np.float64(0.8733333349227905), 'std': np.float64(0.061101003022815056)}
    {'activation': 'relu', 'optimizer': 'RMSprop', 'mean_accuracy': np.float64(0.8666666626930237), 'std': np.float64(0.055777324653567245)}
    {'activation': 'tanh', 'optimizer': 'sgd', 'mean_accuracy': np.float64(0.846666669845581), 'std': np.float64(0.05811864364670431)}
    {'activation': 'relu', 'optimizer': 'sgd', 'mean_accuracy': np.float64(0.7800000071525574), 'std': np.float64(0.058118657320947315)}
    '''

    '''
    risultati gridsearch 100 epochs:
     BEST MODEL:
    {'activation': 'relu', 'optimizer': 'adam', 'mean_accuracy': np.float64(0.95333331823349), 'std': np.float64(0.03399346607355943)}
    {'activation': 'relu', 'optimizer': 'adam', 'mean_accuracy': np.float64(0.95333331823349), 'std': np.float64(0.03399346607355943)}
    {'activation': 'tanh', 'optimizer': 'adam', 'mean_accuracy': np.float64(0.95333331823349), 'std': np.float64(0.0452155428292956)}
    {'activation': 'tanh', 'optimizer': 'RMSprop', 'mean_accuracy': np.float64(0.9466666460037232), 'std': np.float64(0.04000000556310116)}
    {'activation': 'relu', 'optimizer': 'RMSprop', 'mean_accuracy': np.float64(0.9399999856948853), 'std': np.float64(0.06463573864557808)}
    {'activation': 'relu', 'optimizer': 'sgd', 'mean_accuracy': np.float64(0.8933333158493042), 'std': np.float64(0.05333333462476845)}
    {'activation': 'tanh', 'optimizer': 'sgd', 'mean_accuracy': np.float64(0.8800000071525573), 'std': np.float64(0.08844332247117753)}
    '''

    #NN.nn_model(dataset_mg.get_X_train(), 3,dataset_mg.get_y_train())

    #y_pred = NN.predict(dataset_mg.get_X_test())


    #y_pred_class = np.argmax(y_pred, axis=1)
    #y_test_class = np.argmax(dataset_mg.get_y_test(), axis=1)
    #print("Accuracy: ", accuracy_score(y_test_class, y_pred_class))

    #NN.plot_accuracy()
    #NN.plot_loss()

else:
    # flask
    app = FlaskManager()
    app.run(host='0.0.0.0', port=5000, debug=True)