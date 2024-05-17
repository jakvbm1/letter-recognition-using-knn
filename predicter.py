import os

from csv_creation import yed, xed, data_processing, tl2
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

filename = input("Podaj ścieżkę do pliku obrazu: ")
data = tl2(filename)
dataf = pandas.read_csv(r'letter-recognition-new.csv')
y = dataf.pop('letter')
x = dataf.values

to_pred = np.array(data.split(","), ndmin=2, dtype=float)

knn_lib=KNeighborsClassifier(6)
knn_lib.fit(x, y)
print(knn_lib.predict(to_pred))

