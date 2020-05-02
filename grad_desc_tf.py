import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

theta = tf.Variable(-4.0)
loss = lambda: theta**2 - 4* theta +10

opt = tf.keras.optimizers.SGD(learning_rate=0.1)

for iter in range(100):
    opt.minimize(loss, [theta])

print(theta)