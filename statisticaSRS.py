"""
Script per effettuare analisi statistica delle metriche su un campione SRS(k).
Le metriche prese in esame sono ME, MR e ACC.

@author: robertozanolli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import robust
from scipy.stats import skew, kurtosis, shapiro, t
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning) 

path = "data/metricheSRS.csv"
df = pd.read_csv(path)

ME = df["LogReg_ME"]
MR = df["LogReg_MR"]
ACC = df["LogReg_Acc"]

""""
STATISTICA DESCRITTIVA
"""

#ISTOGRAMMA DI ME

sns.histplot(ME,bins=10,  kde=True, color="red")
plt.title( "Istogramma Distribuzione di ME")
plt.xlabel("Valore")
plt.ylabel("Frequenza")
plt.show()

#BOXPLOT DI ME
sns.boxplot(ME, color="red")
plt.title("Box Plot di ME")
plt.ylabel("Valore")
plt.show()

#ISTOGRAMMA DI MR
sns.histplot(MR,bins=10,  kde=True, color="green")
plt.title("Istogramma Distribuzione di MR")
plt.xlabel("Valore")
plt.ylabel("Frequenza")
plt.show()

#BOXPLOT DI MR
sns.boxplot(MR, color="green")
plt.title("Box Plot di MR")
plt.ylabel("Valore")
plt.show()

#ISTOGRAMMA DI ACC
sns.histplot(ACC,bins=10, kde=True, color="blue")
plt.title("Istogramma Distribuzione di ACC")
plt.xlabel("Valore")
plt.ylabel("Frequenza")
plt.show()

#BOXPLOT DI ACC
sns.boxplot(ACC, color="blue")
plt.title("Box Plot di ACC")
plt.ylabel("Valore")
plt.show()


#MISURE DEL CENTRO (media, mediana)

#Calcolo delle medie
media_ME = ME.mean()
media_MR = MR.mean()
media_ACC = ACC.mean()

print("\n------MEDIA-----")
print(f"Media Missclassification Error: {media_ME}")
print(f"Media Missclassification Rate: {media_MR}")
print(f"Media Accuracy: {media_ACC}")


#Calcolo delle mediane (non occorre ordinare la serie perché metodo median di pd lo fa internamente)
mediana_ME = ME.median()
mediana_MR = MR.median()
mediana_ACC = ACC.median()

print("\n------MEDIANA-----")
print(f"Mediana Missclassification Error: {mediana_ME}")
print(f"Mediana Missclassification Rate: {mediana_MR}")
print(f"Mediana Accuracy: {mediana_ACC}")

# MISURE DELLA DIFFUSIONE

# Calcolo varianza
var_ME = ME.var()
var_MR = MR.var()
var_ACC = ACC.var()

print("\n------VARIANZA-----")
print(f"Varianza di Missclassification Error: {var_ME}")
print(f"Varianza di Missclassification Rate: {var_MR}")
print(f"Varianza di Accuracy: {var_ACC}")

#Calcolo devizioni standard
std_ME = ME.std()
std_MR = MR.std()
std_ACC = ACC.std()

print("\n------DEVIAZIONE STD-----")
print(f"Deviazione Standard Missclassification Error: {std_ME}")
print(f"Deviazione Standard Missclassification Rate: {std_MR}")
print(f"Deviazione Standard Accuracy: {std_ACC}")


#Determino Range interquantile (anche quantile ordina internamente)
IQR_ME= ME.quantile(0.75) - ME.quantile(0.25)
IQR_MR= MR.quantile(0.75) - MR.quantile(0.25)
IQR_ACC= ACC.quantile(0.75) - ACC.quantile(0.25)

print("\n------RANGE INTERQUARTILE-----")
print(f"IQR Missclassification Error: {IQR_ME}")
print(f"IQR di Missclassification Rate: {IQR_MR}")
print(f"IQR di Accuracy: {IQR_ACC}")


#MISURE DELLA FORMA DI ME
print("\n------ANALISI DELLA FORMA DI ME-----")
skewness_ME = skew(ME)
kurtosis_ME = kurtosis(ME)
shapiro_test = shapiro(ME)

print(f"Simmetria della distribuzione Missclassification Error: {skewness_ME}")
print(f"Curtosi della distribuzione Missclassification Error : {kurtosis_ME}")
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue}")

#MISURE DELLA FORMA DI MR
print("\n------ANALISI DELLA FORMA DI MR-----")
skewness_MR = skew(MR)
kurtosis_MR= kurtosis(MR)
shapiro_test = shapiro(MR)

print(f"Simmetria della distribuzione Missclassification Rate: {skewness_MR}")
print(f"Curtosi della distribuzione Missclassification Rate : {kurtosis_MR}")
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue}")

#MISURE DELLA FORMA DI ACC
print("\n------ANALISI DELLA FORMA DI ACC-----")
skewness_ACC = skew(ACC)
kurtosis_ACC = kurtosis(ACC)
shapiro_test = shapiro(ACC)

print(f"Simmetria della distribuzione Accuracy: {skewness_ACC}")
print(f"Curtosi della distribuzione Accuracy: {kurtosis_ACC}")
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue}")

""""
STATISTICA INFERENZIALE
"""

#STUDIO INTERVALLI DI CONFIDENZA

#INTERVALLO DI CONFIDENZA DI ME
int_conf_ME = t.interval(confidence = 0.95, 
                       df = len(ME)-1,
                       loc = media_ME, 
                       scale = std_ME / np.sqrt(len(ME)))


print(f"\nIntervallo di confidenza di ME:  {int_conf_ME}")

#INTERVALLO DI CONFIDENZA DI MR
int_conf_MR = t.interval(confidence = 0.95, 
                       df = len(MR)-1,
                       loc = media_MR, 
                       scale = std_MR / np.sqrt(len(MR)))


print(f"Intervallo di confidenza di MR:  {int_conf_MR}")

#INTERVALLO DI CONFIDENZA DI ACC
int_conf_ACC = t.interval(confidence = 0.95, 
                       df = len(ACC)-1,
                       loc = media_ACC, 
                       scale = std_ACC / np.sqrt(len(ACC)))


print(f"Intervallo di confidenza di ACC:  {int_conf_ACC}")

