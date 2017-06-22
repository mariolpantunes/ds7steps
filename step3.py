# coding: utf-8
from sklearn import preprocessing
import numpy as np


X = np.array([[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]])

# Z-score normalization
X_scaled = preprocessing.scale(X)

#print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
print()

# Normalization
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

#print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
print()

# Polynomial features
# [1, a, b, a^2, ab, b^2]
X = np.arange(6).reshape(3, 2)

print(X)

poly = preprocessing.PolynomialFeatures(2)
X_transformed = poly.fit_transform(X)

print(X_transformed)

# PCA
