# systolic.py

# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel('mlr02.xls', encoding_override='cp1252')
X = df.as_matrix()

X_plt = X

plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y, order):
	w = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
	Yhat = X.dot(w)

	if(order == 2):
		plt.scatter(X_plt[:, 1], X_plt[:, 0])
		plt.plot( sorted(X_plt[:, 1]), sorted(Yhat), 'r-' )
		plt.show()
	
	if(order == 3):
		plt.scatter(X_plt[:, 2], X_plt[:, 0])
		plt.plot( sorted(X_plt[:, 2]), sorted(Yhat), 'r-' )
		plt.show()

	if(order == 0):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X_plt[:, 1], X_plt[:, 2], X_plt[:, 0])
		ax.plot(sorted(X_plt[:, 1]), sorted(X_plt[:, 2]), sorted(Yhat), 'r-')
		plt.show()

	d1 = Y - Yhat
	d2 = Y - Y.mean()
	r2 = 1 - d1.dot(d1) / d2.dot(d2)

	return r2



print("r2 for x2 only: ", get_r2(X2only, Y, 2) )
print("r2 for x3 only: ", get_r2(X3only, Y, 3) )
print("r2 for both: ", get_r2(X, Y, 0))



