# l2 ridge

import numpy as np 
import matplotlib.pyplot as plt 

N = 50

X = np.linspace(1, 10, N)
Y = 0.5*X + np.random.randn(N)

Y[-20] += 30
Y[-30] += 30
Y[-10] += 20

plt.scatter(X, Y)
plt.show()

X = np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml)
plt.show()

l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1], Yhat_map, label= 'map - ridge')
plt.legend()
plt.show()