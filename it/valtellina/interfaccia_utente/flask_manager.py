import base64
from io import BytesIO
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template_string
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from it.valtellina.machine_learning.dataset_manager import DatasetManager
from it.valtellina.machine_learning.neural_network import NeuralNetwork

DASHBORD_HTML = '''
<!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>Flask API - Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                background: #f4f6f8;
                color: #2c3e50;
            }

            header {
                background: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }

            .container {
                max-width: 900px;
                margin: 30px auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            .card {
                background: #eef2f7;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
            }

            code {
                background: #ddd;
                padding: 2px 5px;
                border-radius: 5px;
            }

            h2 {
                margin-top: 30px;
            }

            .endpoint {
                border-left: 5px solid #3498db;
                padding-left: 10px;
                margin: 10px 0;
            }

        </style>
    </head>

    <body>

    <header>
        <h1>Flask API - Machine Learning</h1>
        <p>Interfaccia semplice per testare la rete neurale</p>
    </header>

    <div class="container">

        <h2>Cos'è questa API?</h2>
        <div class="card">
            Questa API permette di:
            <ul>
                <li>Fare predizioni con una rete neurale</li>
                <li>Visualizzare grafici di training</li>
                <li>Visualizzare una valutazione del modello</li>
            </ul>
        </div>

        <h2>Endpoint disponibili</h2>

        <div class="card endpoint">
            <h3>/predict (POST)</h3>
            <p>Effettua una predizione usando i dati del fiore (Iris dataset)</p>
            <p><b>JSON input:</b></p>
            <code>
            {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2
            }
            </code>
        </div>

        <div class="card endpoint">
            <h3>/valutazione (GET)</h3>
            <p>Restituisce la valutazione del modello sul test set</p>
            <p><code>{"accuracy": 0.92}</code></p>
        </div>

        <div class="card endpoint">
            <h3>/grafici_valutazione (GET)</h3>
            <p>Mostra i grafici di accuracy e loss del training</p>
        </div>
    </div>
    </body>
    </html>
'''

LOADING_HTML = '''
<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body{
                display:flex;
                justify-content:center;
                align-items:center;
                height:100vh;
                font-family:Arial;
                flex-direction:column;
            }

            .loader{
                width:60px;
                height:60px;
                border:6px solid #ddd;
                border-top:6px solid #3498db;
                border-radius:50%;
                animation:spin 1s linear infinite;
            }

            @keyframes spin{
                from{transform:rotate(0deg);}
                to{transform:rotate(360deg);}
            }
        </style>
    </head>
    <body>

        <div class="loader"></div>
        <h3>Caricamento modello ML...</h3>

        <script>
            fetch('/init')
            .then(() => {
                window.location.href = '/dashboard';
            });
        </script>

    </body>
    </html>
'''

class FlaskManager:
    def __init__(self):
        print("inizializzzo flask_manager")
        self.app = Flask(__name__)
        self.__register_routes()  # inizializzazione delle varie root tutte insieme

        self.__ds_mg = None
        self.__nn = None

    def _init_ml_pipeline(self):
        print("inizializzo dataset manager")

        self.__ds_mg = DatasetManager()
        self.__ds_mg.split_data('Species')
        scaler = self.__ds_mg.scaling()

        print("inizializzo neural network")
        self.__nn = NeuralNetwork(scaler)

        print("training modello")
        X_train = self.__ds_mg.get_X_train()
        y_train = self.__ds_mg.get_y_train()

        self.__nn.nn_model(X_train, 3, y_train)

        print("modello creato e addestrato")

    def run(self, **kwargs):
        self.app.run(**kwargs)

    def __register_routes(self):
        @self.app.route('/')
        def loading():
            return LOADING_HTML

        @self.app.route('/init')
        def init():
            self._init_ml_pipeline()
            return {"status": "ok"}

        @self.app.route('/dashboard')
        def dashboard():
            return DASHBORD_HTML

        @self.app.route('/predict',methods=['POST'])
        def predict():
            if self.__nn is None: return jsonify({"ERRORE": "non hai inizializzato il modello! Usa il path '/' o '/init' per inizializzarlo"})
            input = request.get_json()
            oggetto = {
                'SepalLengthCm': input.get('SepalLengthCm'),
                'SepalWidthCm': input.get('SepalWidthCm'),
                'PetalLengthCm': input.get('PetalLengthCm'),
                'PetalWidthCm': input.get('PetalWidthCm'),
            }
            y_pred = self.__nn.predict(pd.DataFrame([oggetto]))
            return jsonify(y_pred)

        @self.app.route('/grafici_valutazione')
        def grafici():
            if self.__nn is None: return jsonify(
                {"ERRORE": "non hai inizializzato il modello! Usa il path '/' o '/init' per inizializzarlo"})
            fig1 = self.__nn.plot_accuracy()
            fig2 = self.__nn.plot_loss()

            # --- Accuracy image ---
            img1 = BytesIO()
            fig1.savefig(img1, format="png", bbox_inches="tight")
            img1.seek(0)
            encoded1 = base64.b64encode(img1.getvalue()).decode()
            plt.close(fig1)

            # --- Loss image ---
            img2 = BytesIO()
            fig2.savefig(img2, format="png", bbox_inches="tight")
            img2.seek(0)
            encoded2 = base64.b64encode(img2.getvalue()).decode()
            plt.close(fig2)

            html = """
                <h1>Plots</h1>
                <h2>Accuracy</h2>
                <img src="data:image/png;base64,{{ encoded1 }}" style="margin:10px;">
                <h2>Loss</h2>
                <img src="data:image/png;base64,{{ encoded2 }}" style="margin:10px;">
            """

            return render_template_string(html, encoded1=encoded1, encoded2=encoded2)

        @self.app.route('/valutazione')
        def valutazione():
            if self.__nn is None: return jsonify(
                {"ERRORE": "non hai inizializzato il modello! Usa il path '/' o '/init' per inizializzarlo"})
            y_pred = self.__nn.predict(self.__ds_mg.get_X_test())

            y_pred_class = np.argmax(y_pred, axis=1)
            y_test_class = np.argmax(self.__ds_mg.get_y_test(), axis=1)
            accuracy = accuracy_score(y_test_class, y_pred_class)

            return jsonify({
                'accuracy': accuracy
            })