import numpy as np
from sklearn.datasets import load_digits
from layers import Dense, Dropout
from activation_functions import ReLU, Softmax
from loss_function import Categorical_Cross_Entropy_Loss

def train_and_evaluate(seed, epochs=100, learning_rate=0.1, batch_size=32, dropout_rate=0.2):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    digits = load_digits()
    X = digits.data.astype(float) / 16.0
    y = np.eye(10)[digits.target]

    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    train_size = 1400
    X_train, y_train = X[:train_size], y[:train_size]
    validation_input, validation_output = X[train_size:], y[train_size:]

    # Fresh layers & loss per seed so weights don't carry over between runs
    DenseLayer_1 = Dense(64, 64, init="he")
    Activation_ReLU_1 = ReLU()

    Dropout_1 = Dropout()

    DenseLayer_2 = Dense(64, 64, init="he")
    Activation_ReLU_2 = ReLU()

    DenseLayer_3 = Dense(64, 10, init="xavier")
    Activation_Softmax = Softmax()

    Loss_function = Categorical_Cross_Entropy_Loss()

    n_samples = X_train.shape[0]

    for _ in range(epochs):
        perm = rng.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            temp_1 = DenseLayer_1.forward(X_batch)
            relu_1_output = Activation_ReLU_1.forward(temp_1)
            dropout_layer_1 = Dropout_1.forward(relu_1_output, rate=dropout_rate, training=True)

            temp_2 = DenseLayer_2.forward(dropout_layer_1)
            relu_2_output = Activation_ReLU_2.forward(temp_2)

            temp_3 = DenseLayer_3.forward(relu_2_output)
            softmax_output = Activation_Softmax.forward(temp_3)

            Loss_function.forward(softmax_output, y_batch)

            # Backward pass
            d_output_layer = Loss_function.backward(softmax_output, y_batch)
            d_penultimate_layer = DenseLayer_3.backward(d_output_layer)
            d_relu_layer_2 = Activation_ReLU_2.backward(d_penultimate_layer)
            d_second_layer = DenseLayer_2.backward(d_relu_layer_2)
            d_dropout_layer_1 = Dropout_1.backward(d_second_layer)
            d_relu_layer_1 = Activation_ReLU_1.backward(d_dropout_layer_1)
            DenseLayer_1.backward(d_relu_layer_1)

            # Update weights & biases
            for layer in (DenseLayer_1, DenseLayer_2, DenseLayer_3):
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases

    # --- Validation inference (397 samples) ---
    temp_1 = DenseLayer_1.forward(validation_input)
    relu_1_output = Activation_ReLU_1.forward(temp_1)
    dropout_1_output = Dropout_1.forward(relu_1_output, rate=dropout_rate, training=False)

    temp_2 = DenseLayer_2.forward(dropout_1_output)
    relu_2_output = Activation_ReLU_2.forward(temp_2)

    temp_3 = DenseLayer_3.forward(relu_2_output)
    softmax_output = Activation_Softmax.forward(temp_3)

    predicted_classes = np.argmax(softmax_output, axis=1)
    expected_classes = np.argmax(validation_output, axis=1)

    correct = int(np.sum(predicted_classes == expected_classes))
    accuracy = (correct / len(expected_classes)) * 100.0
    return accuracy, correct, len(expected_classes)

