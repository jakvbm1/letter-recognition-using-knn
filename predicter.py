import os
from functions import tl2
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

choice = input("Wybierz czy chcesz podać folder czy plik [f/p]: ")

if choice == 'f':
    directory = input("Podaj ścieżkę do folderu z obrazami: ")
    for filename1 in os.listdir(directory):
        filename = os.path.join(directory, filename1)
        data = tl2(filename)
        dataf = pandas.read_csv(r'letter-recognition-new.csv')
        y = dataf.pop('letter')
        x = dataf

        maxes = []
        mins = []

        for column in x.columns:
            maxes.append(max(x[column]))
            mins.append(min(x[column]))

        to_pred = np.array(data.split(","), ndmin=2, dtype=float)
        x = x.values

        for i in range(len(maxes)):
            for j in range(len(x)):
                x[j][i] = (x[j][i] - mins[i]) / (maxes[i] - mins[i])
            to_pred[:, i] = (to_pred[:, i] - mins[i]) / (maxes[i] - mins[i])

        knn_lib=KNeighborsClassifier(6)
        knn_lib.fit(x, y)
        letter = filename[len(filename)-5]
        print(f"Wartość przewidziana: {knn_lib.predict(to_pred)[0]}")

elif choice == 'p':
    filename = input("Podaj ścieżkę do pliku obrazu: ")
    data = tl2(filename)
    dataf = pandas.read_csv(r'letter-recognition-new.csv')
    y = dataf.pop('letter')
    x = dataf

    maxes = []
    mins = []

    for column in x.columns:
        maxes.append(max(x[column]))
        mins.append(min(x[column]))

    to_pred = np.array(data.split(","), ndmin=2, dtype=float)
    x = x.values

    for i in range(len(maxes)):
        for j in range(len(x)):
            x[j][i] = (x[j][i] - mins[i]) / (maxes[i] - mins[i])
        to_pred[:, i] = (to_pred[:, i] - mins[i]) / (maxes[i] - mins[i])

    knn_lib = KNeighborsClassifier(6)
    knn_lib.fit(x, y)
    letter = filename[len(filename) - 5]
    print(f"Wartość przewidziana: {knn_lib.predict(to_pred)[0]}")
