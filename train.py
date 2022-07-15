from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

letra_a = pd.read_csv('letra_a.csv', header=0)
letra_r = pd.read_csv('letra_r.csv', header=0)

y = [0]*letra_a.shape[0] + [1]*letra_r.shape[0]
df = pd.concat([letra_a, letra_r])
x = df.to_numpy()
neigh = KNeighborsClassifier()
neigh.fit(x, y)