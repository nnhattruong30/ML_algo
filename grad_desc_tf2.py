import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

theta = tf.Variable(-10.0)

@tf.function
def Loss(thete):
    return theta**2 - 4*theta + 10

alpha = 0.01
eps = 0.0001

opt = tf.optimizers.SGD(lr = alpha)

while True: 
    with tf.GradientTape() as tape:
        loss = Loss(theta)
        
    grads = tape.gradient(loss, [theta])
    opt.apply_gradients(zip(grads, [theta]))
    if abs(grads[0].numpy()) < eps: break

print(theta)