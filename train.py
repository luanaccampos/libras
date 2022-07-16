from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
import pickle5 as pickle

files = os.listdir('dataset')

df = pd.DataFrame()
y = []

for file in files:
    d = pd.read_csv('dataset/' + file, header=0)
    df = pd.concat([df, d])
    y.extend([file[0]]*d.shape[0])

x = df.to_numpy()
neigh = KNeighborsClassifier()
neigh.fit(x, y)

pickle.dump(neigh, open('model.sav', 'wb'))