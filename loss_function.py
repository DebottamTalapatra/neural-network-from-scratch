import numpy as np

class Categorical_Cross_Entropy_Loss:
    def forward(self, inputs, data):
        clipped = np.clip(inputs, 1e-12, 1.0 - 1e-12)
        sample_losses = -np.sum(data * np.log(clipped), axis=1)
        self.output = np.mean(sample_losses)
        return self.output

    def backward(self, inputs, data):  # Combines backward() for Categorical_Cross_Entropy_Loss + Softmax
        self.dinputs = (inputs - data) / np.shape(inputs)[0]
        return self.dinputs

