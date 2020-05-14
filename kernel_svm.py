import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X = np.c_[(1, 1), (0, 0), (1, 0), (0, 1)].T
Y = [0]*2 + [1]*2

fig = plt.figure()
ax_num = 1
for kernel in ('sigmoid', 'poly', 'rbf'):
    clf = SVC(kernel= kernel, gamma=4, coef0=0)
    ax = fig.add_subplot(2, 2, ax_num)
    clf.fit(X, Y)
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='y')
    ax.plot(X[:2, 0], X[:2, 1], 'ro')
    ax.plot(X[2:, 0], X[2:, 1], 'gs')
    ax.axis('tight')
    x_min, x_max = -2, 3
    y_min, y_max = -2, 3
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    CS = ax.contourf(XX, YY, np.sign(Z), 200, cmap = 'jet', alpha = .2)
    ax.contour(XX, YY, Z, 
                colors = ['k', 'k', 'k'], 
                linestyles = ['--', '-', '--'], 
                levels = [-0.5, 0, 0.5])
    ax.set_title(kernel, fontsize = 12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax_num += 1
plt.show()