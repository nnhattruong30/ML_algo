import numpy as np
import matplotlib.pyplot as plt

n_sample = 20
distr = 3
size = (2, n_sample)

# Tao mau du lieu
S1 = np.array([[1], [1]])
X1 = S1 + np.random.normal(0, distr, size)
S2 = np.array([[12], [12]])
X2 = S2 + np.random.normal(0, distr, size)

X = np.concatenate((X1, X2), axis=1)
ones = np.ones((1, 2*n_sample))
X_b = np.concatenate((ones, X))
Y = np.concatenate((np.zeros(n_sample), np.ones(n_sample)))

plt.plot(X1[0], X1[1], 'ro')
plt.plot(X2[0], X2[1], 'go')

theta = np.array([[0.], [1.], [2.]])
lr = 0.1
eps = 1e-4
x_vis = np.array([-7., 20.])
y_vis = -theta[1][0]/theta[2][0] * x_vis - theta[0][0]/theta[2][0]
plt.plot(x_vis, y_vis)

def sigmoid(x):
    return 1./(1 + np.exp(-x))

while True:
    napla = np.dot(X_b, (sigmoid(np.dot(theta.T, X_b)) - Y).T)
    theta = theta - lr * napla
    y_vis = -theta[1][0]/theta[2][0] * x_vis - theta[0][0]/theta[2][0]
    plt.plot(x_vis, y_vis)
    plt.pause(0.5)
    if abs(napla[0][0]) < eps and abs(napla[1][0]) < eps and abs(napla[2][0]) < eps: break

plt.show()
print("Theta toi uu:", theta)