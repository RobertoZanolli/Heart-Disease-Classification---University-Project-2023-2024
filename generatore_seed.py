"""
Questo script serve per generare una sequenza di k >= 10 seed per 
effettuare lo studio statistico delle metriche del modello (eseguire il 
modello k volte con k semi diversi)

@author: robertozanolli
"""

import numpy as np

def get_user_input():
    while True:
        k = int(input("Inserisci un numero k maggiore di 10: "))
        if k >= 10:
            return k
        else:
            print("ERRORE: k<10, riprova")
      

k = get_user_input()
random_vector = np.random.randint(0, 150, size=k)
print(f"Vettore di {k} numeri casuali: {random_vector}")




