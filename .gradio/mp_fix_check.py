import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"c:\Users\MUHAMMAD RIZKY R\Documents\DATA SCIENCE\Lepkom Gunadarma\SEMESTER 7\Tugas\data.csv")
df = df[(df['price'] > 0) & (df['bedrooms'] > 0) & (df['bathrooms'] > 0)].copy()
features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
X = df[features]
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
model = DBSCAN(eps=0.75, min_samples=10)
labels = model.fit_predict(Xs)
vc = pd.Series(labels).value_counts().sort_index().to_dict()
print('unique', sorted(set(labels.tolist())))
print('counts', vc)
clean = df[labels != -1].copy()
clean['Cluster'] = labels[labels != -1]
print('clean rows', len(clean), 'unique clusters', sorted(set(clean['Cluster'].tolist())))
X_clean = clean[features]
scaler2 = StandardScaler(); Xs2 = scaler2.fit_transform(X_clean)
knn = KNeighborsClassifier(n_neighbors=1).fit(Xs2, clean['Cluster'])
for luas, harga in [(1200, 180000), (1500, 320000), (2500, 600000), (3000, 900000), (4200, 1400000)]:
    sample = pd.DataFrame({'price':[harga], 'bedrooms':[2], 'bathrooms':[2], 'sqft_living':[luas], 'sqft_lot':[5000]})
    x = scaler2.transform(sample[features])
    pred = knn.predict(x)[0]
    print((luas, harga), '->', pred)
