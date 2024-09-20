"""
Questo script implementa un modello predittivo sulla base del dataset "Hearth Disease Classification Dataset" 
trovabile su kaggle seguendo https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset/data .
Il dataset è diviso in 9 colonne di cui 1 di output (positive o negative in metito alla presenza o meno dell' attacco di cuore)
e 8 di input rispettivamente: età (int), genere (0=donna, 1=uomo), frequenza cardiaca (int), pressione sistolica (int), pressione 
diastolica (int), glucosio (float), Creatinchinasi-MB (float), Troponina(float).

@author: roberto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Carico il csv in un Dataframe Pandas
path = f"data/HeartAttack.csv"
data = pd.read_csv(path)

# 1) PREPROCESSING


#Stampo le informazioni sui dati per verificare la presenza di NaN e per controllare il tipo dei dati

print(data.info()) #Mi accorgo già che non ci sono dati mancanti
print(data.isna().any()) #conferma 


#sostituisco i valori della colonna class : da string a booleani --> negative a False e positive a True
#data.replace({'class': {'negative': False, 'positive': True}},inplace=True)


#inizializzo lista con i nomi delle variabili input
features = ["age","gender","impluse","pressurehight","pressurelow","glucose","kcm", "troponin"]
"""
#Controllo la presenza di outliers mostrando il box plot e lo scatterplot per ogni feature
print("\n---------------------PLOTTING PER ANALISI OUTLIERS----------------------")
print("---------------------BOXPLOT E SCATTERPLOT----------------------")
for feature in features:
    plt.title(label=feature)
    plt.boxplot(data[feature],sym="o")
    plt.show()
    plt.title(label=feature)
    plt.plot(data[feature],"*")
    plt.show()
"""

"""
#considerazioni sugli outlier: 
i valori di creatinchinasi (kcm) e troponina (troponin) dovrebbero tendere a 0 in paziendti sani (mediana)
il glucosio dovrebbe restare tra 77 e 99 mg/dl in pazienti sani 
in generale più il valore si discosta dalla soglia accettabile  maggiore è il rischio di attacco di cuore
unici outliers da rimuovere sono le 3 misurazioni per le quali i battiti superano i mille al minuto
"""

#Rimuovo righe dove i battiti al minuto superano i 250 
data = data[data["impluse"] <= 250]
#Rimuovo righe dove pressione minima supera pressione massima
data = data[data["pressurelow"] < data["pressurehight"]]

"""
print("\n---------------------PLOTTING RIMOZIONE OUTLIERS E DISTRIBUZIONE----------------------")
print("---------------------BOXPLOT E ISTOGRAMMA----------------------")
for feature in features:
    plt.title(label=feature)
    plt.boxplot(data[feature],sym="o")
    plt.show()
    plt.title(label=feature)
    plt.hist(data[feature], bins=50, alpha=0.7, color="r")
    plt.show()

""" 

#estraggo dati numerici da dataset
num_data = data.select_dtypes(include = "number")
#N.B. age viene considerata intera ma questo non ha molta importanza 
#in quanto non verrá utilizzata per la regressione lineare


#2) EDA
df_stats=num_data.describe()
#MATRICE DI CORRELAZIONE -> noto correlazione tra le due pressioni sistolica e diastolica 
#(se ho ipertensione sono entrambe alte non posso avere alta solo una) + leggera correlazione tra età e troponina/cheratinchinasi (
#età è fattore di rischio -> biomarker si alzano)
print("\n---------------------MATRICE DI CORRELAZIONE ----------------------")

plt.matshow(num_data.corr(), vmin=-1, vmax=1)
plt.xticks(np.arange(0, num_data.shape[1]), features, rotation=90)
plt.yticks(np.arange(0, num_data.shape[1]), features)
plt.title("Matrice di Correlazione dei valori input")
plt.colorbar()
plt.show()

#3) SPLITTING

#SPLITTING DEL DATASET in dati di training, test ed evaluation
from sklearn import model_selection
seed=np.random.seed(99)


data_train, data_test =  model_selection.train_test_split(data,train_size=1150,random_state=seed)
data_train,data_val = model_selection.train_test_split(data_train,train_size=850,random_state=seed)


#Suddivido X ed Y
x_train=data_train[features]
Y_train=data_train["class"]

x_val =data_val[features]
Y_val = data_val["class"]

x_test=data_test[features]
Y_test=data_test["class"]



#4) REGRESSIONE LINEARE DI VARIABILI CORRELATE


#CASO A (Pressure low e pressure high 

from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#Reshape dei dati 
xA_train=data_train["pressurelow"].values.reshape(-1,1)
yA_train=data_train["pressurehight"].values.reshape(-1,1)

xA_val = data_val["pressurelow"].values.reshape(-1,1)
yA_val = data_val["pressurehight"].values.reshape(-1,1)

modelA = linear_model.LinearRegression()
modelA.fit(xA_train,yA_train)
yA_pred=modelA.predict(xA_val)

print("\n---------------------PLOT REGRESSIONE LINEARE CASO A ----------------------")

plt.scatter(xA_val, yA_val)
plt.plot(xA_val,yA_pred,color="red")
plt.xlabel("pressure low")
plt.ylabel("pressure high")
plt.title("Regressione lineare A")
plt.show()

print("\n---------------------METRICHE REGRESSIONE LINEARE CASO A----------------------")

#CALCOLO DEI COEFFICIENTI DI A
print(f"Coefficiente angolare: {modelA.coef_[0]}")
print(f"Intercetta: {modelA.intercept_}")

#CALCOLO R^2 DI A
r_squaredA = r2_score(yA_val,yA_pred)
print(f"Coefficiente di determinazione (R^2): {(r_squaredA)}") 

#CALCOLO MAE DI A
maeA = mean_absolute_error(yA_val, yA_pred)
print(f"Mean Absolute Error (MAE): {maeA}")

#CALCOLO MSE DI A
mseA = mean_squared_error(yA_val, yA_pred)
print(f"Mean Squared Error (MSE): {mseA}")


#ANALISI RESIDUI A
residuals = yA_val - yA_pred 
print("\n---------------------ISTOGRAMMA RESIDUI CASO A----------------------")

sns.histplot(residuals,kde=True,bins=50),

plt.xlabel("Residuo")
plt.ylabel("Densitá di frequenza")
plt.title("Distribuzione dei residui")
plt.show()


print("\n---------------------QQ-PLOT RESIDUI CASO A----------------------")
import statsmodels.api as sm
from scipy.stats import shapiro

sm.qqplot(residuals[:, 0], line="45",fit = True)
plt.show()

print("\n---------------------TEST NORMALITÁ RESIDUI CASO A----------------------")

shapiro_test = shapiro(residuals[:,0])
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue}")

#CASO B (Pressure low e impulse)
yB_train=data_train["impluse"].values.reshape(-1,1)
xB_train=xA_train

xB_val = xA_val
yB_val = data_val["impluse"].values.reshape(-1,1)


modelB = linear_model.LinearRegression()
modelB.fit(xB_train,yB_train)
yB_pred=modelB.predict(xB_val)

print("\n---------------------PLOT REGRESSIONE LINEARE CASO B ----------------------")
plt.scatter(xB_val, yB_val)
plt.plot(xB_val,yB_pred,color="g")
plt.xlabel("pressure low")
plt.ylabel("impulse")
plt.title("Regressione lineare B")
plt.show()

print("\n---------------------METRICHE REGRESSIONE LINEARE CASO B----------------------")

#CALCOLO DEI COEFFICIENTI DI B
print(f"Coefficiente angolare: {modelB.coef_[0]}")
print(f"Intercetta: {modelB.intercept_}")

#CALCOLO R^2 DI B
r_squaredB = r2_score(yB_val,yB_pred)
print(f"Coefficiente di determinazione (R^2): {(r_squaredB)}") 
#CALCOLO MAE DI B
maeB = mean_absolute_error(yB_val, yB_pred)
print(f"Mean Absolute Error (MAE): {maeB}")

#CALCOLO MSE DI B
mseB = mean_squared_error(yB_val, yB_pred)
print(f"Mean Squared Error (MSE): {mseB}")


#ANALISI RESIDUI B
residuals = yB_val - yB_pred 
print("\n---------------------ISTOGRAMMA RESIDUI CASO B----------------------")


sns.histplot(residuals,kde=True,bins=50)
plt.xlabel("Residuo")
plt.ylabel("Frequenza")
plt.title("Distribuzione dei residui")
plt.show()


print("\n---------------------QQ-PLOT RESIDUI CASO B----------------------")

sm.qqplot(residuals[:, 0], line="45", fit = True)
plt.title("QQ-Plot Residui B")
plt.show()

print("\n---------------------TEST NORMALITÁ RESIDUI CASO B----------------------")
shapiro_test = shapiro(residuals[:,0])
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue}")



#5) ALLENAMENTO DEL MODELLO SVM E HYPERPARAMETER TUNING

from sklearn import svm

"""
print("---------------------HYPERPARAMETER TUNING SVM poly ----------------------")

for d in range(1,9): #accuratezza massima con grado 8 (seed=99)...
    model_SVM = svm.SVC(kernel="poly",degree=d)
    model_SVM.fit(x_train,Y_train) 

    y_pred = model_SVM.predict(x_val)
          

    ME = np.sum(y_pred!=Y_val)
    MR=ME/len(y_pred)
    Acc=1-MR
    print(f"ACC:{Acc} : DEGREE= {d}")

"""
    
"""    
#notato che non  varia aaccuratezza al variare di gamma
print("---------------------HYPERPARAMETER TUNING SVM rbf ----------------------")

for d in range(1,9):
    model_SVM = svm.SVC(kernel="rbf",gamma=d)
    model_SVM.fit(x_train,Y_train) 

    y_pred = model_SVM.predict(x_val)
              

    ME = np.sum(y_pred!=Y_val)
    MR=ME/len(y_pred)
    Acc=1-MR
    print(f"ACC:{Acc} : DEGREE= {d}")

"""

#6) ALLENAMENTO DEL MODELLO LOG. REGRESSION

#aggiungo al training set il validation set in quanto non mi serve piú
x_train= pd.concat([x_train, x_val])
Y_train=pd.concat([Y_train,Y_val])

#Creazione e training modello con Logistic Regression
model_LR = linear_model.LogisticRegression(solver="lbfgs", max_iter=1000)
model_LR.fit(x_train,Y_train) 

#7) VALUTAZIONE PERFORMANCE MODELLO LR
y_pred = model_LR.predict(x_test)

#Metriche LR

print("---------------------METRICHE LR ----------------------")
ME = np.sum(y_pred != Y_test)
print(f"Missclassification Error: {ME}.")
MR = np.mean(y_pred != Y_test)
print(f"Missclassification Rate: {MR}.")
Mper=100 * MR
print(f"Mper: {Mper}.")
Acc = 1-MR
print(f"Accuracy: {Acc}.")

#Valutazione Matrice di Confusione con LR 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("---------------------MATRICE DI CONFUSIONE PREDIZIONI CON LR----------------------")
cm = confusion_matrix(y_pred, Y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negativo", "Positivo"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice di Confusione LR")
plt.xlabel("Valori Predetti")
plt.ylabel("Valori Reali")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"Veri Negativi (TN): {tn}")
print(f"Falsi Positivi (FP): {fp}")
print(f"Falsi Negativi (FN): {fn}")
print(f"Veri Positivi (TP): {tp}")

print(f"Sensitivitá (TPR): {tp/(tp+fn)}")
print(f"Specificitá (TNR): {tn/(tn+fp)}")
print(f"Precisione (PPV): {tp/(tp+fp)}")

#8) ALLENAMENTO DEL MODELLO SVM

model_SVM = svm.SVC(kernel="poly",C=1,degree=8)
model_SVM.fit(x_train,Y_train) 

#7) VALUTAZIONE PERFORMANCE MODELLI SVM
y_pred = model_SVM.predict(x_test)

#Metriche SVM

print("---------------------METRICHE SVM kernel=poly----------------------")
ME = np.sum(y_pred != Y_test)
print(f"Missclassification Error: {ME}.")
MR = np.mean(y_pred != Y_test)
print(f"Missclassification Rate: {MR}.")
Mper=100 * MR
print(f"Mper: {Mper}.")
Acc = 1-MR
print(f"Accuracy: {Acc}.")
tn, fp, fn, tp
model_SVM = svm.SVC(kernel="rbf",C=1,gamma=1)
model_SVM.fit(x_train,Y_train) 

y_pred = model_SVM.predict(x_test)

print("---------------------METRICHE SVM kernel=rbf----------------------")
ME = np.sum(y_pred != Y_test)
print(f"Missclassification Error: {ME}.")
MR = np.mean(y_pred != Y_test)
print(f"Missclassification Rate: {MR}.")
Mper=100 * MR
print(f"Mper: {Mper}.")
Acc = 1-MR
print(f"Accuracy: {Acc}.")



