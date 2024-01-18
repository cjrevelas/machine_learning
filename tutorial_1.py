import numpy as np
import tensorflow as tf

print(tf.__version__)

weights_tf = tf.Variable(tf.random.normal([10], mean = 0.0, stddev = 0.35), name = "weights")
biases_tf = tf.Variable(tf.zeros([200]), name = "biases")
print(weights_tf)
print()
print(biases_tf)
print()

# neural network from scratch
inputs  = np.array([1, 2, 3, 2.5], dtype = float)
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                   [0.5, -0.9, 0.26, -0.5],
                   [-0.26, -0.27, 0.17, 0.87]], dtype = float)

biases = np.array([2, 3, 0.5], dtype = float)

output  = np.dot(weights, inputs) + biases
print(output)