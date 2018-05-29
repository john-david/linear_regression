#lr_2d.py

import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

# load the data

X = []
Y = []

for line in open('data_2d.csv'):
	x1, x2, y = line.split(',')
	X.append([1, float(x1), float(x2)])
	Y.append(float(y))

# create numpy arrays

X = np.array(X)
Y = np.array(Y)

# plot the raw data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# calcuate weights

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

Yhat = np.dot(X, w) 

# plot the regressor fit

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
ax.plot(sorted(X[:, 0]), sorted(X[:, 1]), sorted(Yhat), 'r-')
plt.show()


# find r2
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared: ", r2)

