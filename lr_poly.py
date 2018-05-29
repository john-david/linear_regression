# lr_poly.py

import numpy as np 
import matplotlib.pyplot as plt 

# load data

X = []
Y = []

for line in open('data_poly.csv'):
	x, y = line.split(',')
	x = float(x)
	X.append([1, x, x*x])
	Y.append(float(y))

# convert to numpy arrays

X = np.array(X)
Y = np.array(Y)

# plot data

plt.scatter(X[:, 1], Y)
plt.show()

# calculate the weights

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# plot data with regressor line

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat), 'r-')
plt.show()

# calculate r^2

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared: ", r2)

