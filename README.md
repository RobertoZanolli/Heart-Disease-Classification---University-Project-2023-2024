<h1>Heart Disease Classification Project</h1> 

Questo progetto si focalizza sull'analisi di un dataset relativo a malattie cardiovascolari e sulla costruzione di un modello di machine learning per prevedere il rischio di attacco cardiaco. L'obiettivo principale è identificare i fattori che influenzano maggiormente il rischio e sviluppare un modello predittivo accurato.

<h3>Gli script inclusi nel progetto sono:</h3> 
<ul>
<li> HeartDiseaseClassification.py: Questo script contiene il codice principale per l'addestramento e la valutazione di vari modelli di classificazione, tra cui la regressione logistica e le macchine a vettori di supporto (SVM). Esegue anche l'ottimizzazione degli iperparametri e calcola le metriche di performance, come accuracy, precision e recall. </li>


<li>statisticaSRS.py: Questo script esegue un'analisi statistica dei risultati del modello, generando statistiche descrittive e inferenziali. Inoltre, calcola intervalli di confidenza e altre metriche per valutare l'affidabilità dei risultati.</li>

<li> generatore_seed.py: Questo script genera numeri casuali utilizzati per inizializzare il seed in modo da mantenere costante la suddivisione del dataset tra training, validation e test set, garantendo risultati riproducibili.</li>
</ul>
<h3> I file CSV associati al progetto includono:</h3>

<li>HeartAttack.csv: Il dataset contenente le rilevazioni relative a diversi parametri medici, come età, genere, pressione sanguigna, glicemia e marker cardiaci, utilizzati per la predizione degli attacchi cardiaci.</li>
<li>metricheSRS.csv: Questo file memorizza le metriche delle prestazioni dei modelli valutati durante la fase di addestramento e validazione.</li>

<h2>Funzionalità degli script</h2>
<ol> 
<li> Preprocessing dei dati: Gli script gestiscono la pulizia e la preparazione del dataset. Questo include la rimozione di valori anomali e l'elaborazione di features numeriche per garantire che il dataset sia pronto per l'addestramento dei modelli di machine learning.</li>

<li>Exploratory Data Analysis (EDA): Viene condotta un'analisi esplorativa dei dati, con la generazione di boxplot, scatterplot e la creazione di una matrice di correlazione per comprendere meglio le relazioni tra le variabili, identificare outlier e pattern nascosti </li>
<li>Training dei modelli: L'addestramento include diversi modelli, come la regressione logistica e le Support Vector Machines (SVM) con kernel polinomiale e radiale (rbf). È stata eseguita una ricerca degli iperparametri ottimali per evitare overfitting e underfitting, garantendo al contempo le migliori performance sui dati.</li>
<li>Valutazione delle performance: Ogni modello è stato valutato utilizzando diverse metriche come misclassification error, accuracy, precision e recall. Inoltre, è stata generata una matrice di confusione per esaminare in dettaglio i falsi positivi e i falsi negativi, particolarmente importanti in un contesto medico.</li>
<li>Indagine statistica: Per valutare la stabilità del modello, sono stati generati seed casuali utilizzati per eseguire più sessioni di addestramento. Le metriche calcolate per ogni sessione sono state memorizzate in un file CSV per un'ulteriore analisi statistica, comprendente la media campionaria, la deviazione standard e l'intervallo di confidenza al 95%.</li>

</ol>
<h2>Dataset</h2>
Il dataset utilizzato in questo progetto è disponibile su Kaggle al seguente link:
https://www.kaggle.com/datasets/bharath011/heart-disease-classi cation-dataset/data.
