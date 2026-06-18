# Iris_classification

## iris classification tramite neural network
Progetto di Combi Federico, Meni Gianluca.

Il progetto Iris Classifier è un'API in Flask 
progettata per analizzare il dataset iris e classificare i record. Inoltre, dato un nuovo campione, permette di classificarlo;
il tutto tramite l'utilizzo di una rete neurale creata appositamente.

## DATASET
Il dataset si trova a questo link: [https://www.kaggle.com/datasets/uciml/iris](https://www.kaggle.com/datasets/uciml/iris).

Il dataset è un file ".csv" composto dalle seguenti feature:
1. id 
2. SepalLengthCm 
3. SepalWidthCm 
4. PetalLengthCm
5. PetalWidthCm

## Funzionalità Principali

- **Analisi del dataset**: 
Esplorazione delle feature  del dataset, 
analisi delle distribuzioni e delle frequenze delle classi, 
studio delle relazioni tra variabili (EDA) tutto tramite Power BI
- **Preprocessing dei dati**: 
Scaling dei dati perchè i dati da dare in ingresso ad una rete neurale devono essere nello stesso range.
- **Utilizzo del modello di Machine Learning**:
Applicazione del modello neural network,
con valutazione delle performance tramite metriche
di Accuracy e Loss.
- **Tuning degli Ipermparametri**: 
Confronto per le varie opzioni di ottimizzazione e le funzioni di attivazione per trovare la coppia con risultati migliori.

## Avvio di Iris Classifier

### Requisiti

Prima di avviare Iris Classifier, assicurati di avere installato tutte le dipendenze necessarie. Puoi trovarle nel file `requirements.txt`.

### Avvio in locale

Per visualizzare l'API in locale è necessario recarsi al seguente indirizzo:

   ```
   http://127.0.0.1:5000/
   ```
## Avvio con Docker

### Prerequisiti

Assicurati di avere installato Docker. Puoi scaricare l'applicazione [qui](https://www.docker.com/products/docker-desktop/).

Verifica l’installazione con:

```bash
docker --version
```
### Utilizzo di Docker

1. a. Costruisci l'immagine Docker da linea di comando:
   ```bash
   docker build -t seeds-classifier .
   ```
   b. L'immagine è possibile scaricarla da docker hub al seguente link:
   [link](https://hub.docker.com/r/f3d3ri/iris_classification)
   
   
3. Si possono controllare le informazioni dell'immagine appena creata con il comando:
   ```bash
   docker image ls
   ```

4. Costruisci e avvia il container da linea di comando (copiando questo comando, verrà chiamato "iris-classifier"):
   ```bash
   docker run -d --name seeds -p 5000:5000 iris-classifier
   ```
   Se l'operazione è andata a buon fine è possibile vedere lo stavo attivo del container tramite il comando:
   ```bash
   docker ps
   ```

5. Accedi all'applicazione (nella sua route home) tramite il tuo browser all'indirizzo:
   ```
   http://127.0.0.1:5000/
   ```
## Utilizzo dell'API

L'API è consultabile direttamente da browser.

Se dovesse servire, è possibile installare dei plug-in, tra cui:

- Rest-Client (Chrome): [download](https://chromewebstore.google.com/detail/rest-client/oienkoejnhkbcibhdnpjoemdnmiokgah)
- Rested (Firefox): [download](https://addons.mozilla.org/en-US/firefox/addon/rested/)

L'utilizzo dei plug-in non permette però la restituzione delle immagini, facendo risultare "strana" la risposta di alcuni endpoint.

## API Endpoints di Seeds Classifier

Iris Classifier è provvisto di diversi endpoint GET e POST, consultabili nella route **home**.
```
http://127.0.0.1:5000/
```
Le funzionalità dell'API sono le seguenti:

### Grafici di valutazione

Endpoint che genera e restituisce i grafici di valutazione del modello in formato immagine
```
http://127.0.0.1:5000/grafici
```

### Valutazione Metriche Modello

Endpoint che valuta la rete neurale usando le metriche
```
http://127.0.0.1:5000/valutazione
```

### Previsione Classe su nuovo campione

Endpoint che esegue la previsione per un nuovo campione.
Bisogna strutturare la richiesta JSON seguendo la struttura del dataset:
```
{
  "SepalLengthCm": 4.3,
  "SepalWidthCm": 5.0,
  "PetalLengthCm": 0.87,
  "PetalWidthCm": 5.6
}
```

```
http://127.0.0.1:5000/predict
```
