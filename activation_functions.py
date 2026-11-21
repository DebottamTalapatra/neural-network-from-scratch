import numpy as np 

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, inputs):
        dinputs = inputs.copy()
        dinputs[self.inputs <= 0] = 0
        self.dinputs = dinputs
        return self.dinputs


class Softmax:
    def forward(self, inputs):
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
