import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Tao phat sinh du lieu de train model y = 3x + 10
x = np.arange(-5, 5, 0.5)
n = len(x)      # So mau du lieu
ones = np.ones((1, n))
X = np.concatenate((ones, [x]))

std = 3 # tham so do lech chuan the hieu muc do nhieu nhieu hay it
Y = 3*x + 10 + np.random.normal(0, std, n)     # Ky vong tham so theta la [[10], [3]]
plt.plot(x, Y, 'ro')

# Khoi tao tham so
lr = 0.01
theta = tf.Variable(np.array([[2.], [-5.]]))   # Ky vong theta la: [[10], [3]]
eps = 1e-5

opt = tf.keras.optimizers.SGD(learning_rate = lr)

# Dinh nghia ham loss
@tf.function
def Loss(theta):
    return tf.reduce_mean((tf.linalg.matmul(tf.transpose(theta), X) - Y)**2) # Ham f(theta, X, Y) = mean((theta^T.X - Y)^2)

while True:
    with tf.GradientTape() as tape:
        # Tinh gia tri ham 'loss' voi ham Loss(theta) 
        loss = Loss(theta)
    # Tinh dao ham ham loss
    grads = tape.gradient(loss, [theta])
    # Truc quan hoa
    x_vis = np.array([-5, 5])
    y_vis = theta[1] * x_vis + theta[0]
    plt.plot(x_vis, y_vis)
    #plt.pause(0.5)
    # Cap nhat theta dua vao dao ham 'grads'
    opt.apply_gradients(zip(grads, [theta]))

    if abs(grads[0].numpy()[0][0]) < eps and abs(grads[0].numpy()[1][0]) < eps:
        break

plt.show()
print('Theta toi uu:', theta)
    