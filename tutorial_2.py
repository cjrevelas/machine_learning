import numpy as np

np.random.seed(0)

# NEURAL NETWORK FROM SCRATCH
xx = np.array([[1, 2, 3, 2.5],
               [2, 5, -1, 2],
               [-1.5, 2.7, 3.3, -0.8]], dtype = float)

class Layer:
    # When we create a new layer we need to know the input size and the number of neurons
    # n_inputs is equal to the size of each batch of the xx input matrix
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        pass
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer(4,5) # first layer has 5 neurons receiving 4 inputs from its sample/batch
layer2 = Layer(5,2) # second layer has 2 neurons receiving 5 inputs from the previous layer

layer1.forward(xx)
print(layer1.output) # For each input sample (3 in total), we get 5 output coeffs, i.e., a 3 x 5 matrix

print()

# Now, we will feed the output of layer1 to layer2
layer2.forward(layer1.output)
print(layer2.output)
