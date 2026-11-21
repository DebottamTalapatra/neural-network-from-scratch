import numpy as np 

class Dense:
    def __init__(self, x, y, init="he"):
        if init == "he":
            std = np.sqrt(2.0 / x)
        elif init == "xavier":
            std = np.sqrt(1.0 / x)
        else:
            std = 0.01

        self.weights = std * np.random.randn(x, y)
        self.biases = np.zeros(y)

    def forward(self, input):
        self.inputs = input
        self.outputs = np.dot(input, self.weights) + self.biases
        return self.outputs

    def backward(self, inputs):
        self.dweights = np.dot(self.inputs.T, inputs)
        self.dbiases = np.sum(inputs, axis=0)
        self.dinputs = np.dot(inputs, self.weights.T)
        return self.dinputs


class Dropout:
    def forward(self, inputs, rate=0.0, training=True):
        if not training or rate <= 0.0:
            self.mask = np.ones_like(inputs)
            self.output = inputs
            return self.output

        scale = 1.0 / (1.0 - rate)
        self.mask = (np.random.rand(*inputs.shape) >= rate) * scale
        self.output = inputs * self.mask
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
        return self.dinputs
