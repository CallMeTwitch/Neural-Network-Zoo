##### Imports #####
import numpy as np
 
##### Perceptron Class #####
class Perceptron:
    def __init__(self, input_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Network
        self.weights = np.zeros(input_size)
        self.bias = 0

    # Forward Propagation
    def forward(self, input):
        layer_output = np.dot(input, self.weights) + self.bias
        return np.where(layer_output > 0, 1, 0)

    # Back Propagation
    def backward(self, error, input):
        self.weights += self.learning_rate * error * np.array(input)
        self.bias += self.learning_rate * error

    # Train
    def train(self, inputs, labels):
        # Iterate Epochs
        for _ in range(self.num_epochs):
            # Iterate Pairs of Inputs and Labels
            for input, label in zip(inputs, labels):
                # Predict
                prediction = self.forward(input)
                
                # Back Propogation
                self.backward(label - prediction, input)
    
    # Test
    def test(self, inputs, labels):
        # Iterate Pairs of Inputs and Labels
        for input, label in zip(inputs, labels):
            # Predict
            prediction = self.forward(input)

            # Print
            print(f'Input: {input}, Prediction: {prediction}, Label: {label}')

# Initialize Perceptron
perceptron = Perceptron(input_size = 2, num_epochs = 1_000, learning_rate = 0.01)

##### Training #####
training_inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
training_labels = [1, 0, 0, 0]
perceptron.train(training_inputs, training_labels)

##### Testing #####
testing_inputs = [[1, 1], [0, 1]]
testing_labels = [1, 0]
perceptron.test(testing_inputs, testing_labels)
