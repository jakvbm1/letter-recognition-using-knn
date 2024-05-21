from functions import tl2
import os

directory = input("Podaj ścieżkę do folderu z literkami, które chcesz dodać do bazy danych: ")
file_csv = open('letter-recognition-new.csv', 'a')
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        data = tl2(f)
        file_csv.write('\n'+f[len(f)-5] + ',' + data)