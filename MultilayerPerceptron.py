##### Imports #####
from matplotlib import pyplot as plt
import numpy as np

##### Activation Function #####
def sigmoid(input, derivative = False):
    if derivative:
        return sigmoid(input) * (1 - sigmoid(input))
    
    return 1 / (1 + np.exp(-input))

##### Multilayer Perceptron Class #####
class MLP:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Network
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)

        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    # Forward Propagation
    def forward(self, input):
        self.layer1_output = np.dot(input, self.w1) + self.b1
        self.activation1_output = sigmoid(self.layer1_output)

        self.layer2_output = np.dot(self.activation1_output, self.w2) + self.b2
        self.activation2_output = sigmoid(self.layer2_output)

    # Backpropagation 
    def backward(self, error, input):
        error2 = error * sigmoid(self.layer2_output, derivative = True)
        dw2 = np.dot(self.activation1_output.T, error2)

        error1 = np.dot(self.w2, error2.T).T * sigmoid(self.layer1_output, derivative = True)
        dw1 = np.dot(input.T, error1)

        self.w1 += dw1 * self.learning_rate
        self.b1 += error1.flatten() * self.learning_rate

        self.w2 += dw2 * self.learning_rate
        self.b2 += error2.flatten()  * self.learning_rate

    # Train
    def train(self, inputs, labels):
        # Iterate Epochs
        for _ in range(self.num_epochs):
            # Iterate Pairs of Inputs & Labels
            for input, label in zip(inputs, labels):
                # Predict
                self.forward(input)

                # Back Propagation
                self.backward(label - self.activation2_output, input)

    # Test
    def test(self, inputs):
        # Iterate Pairs of Inputs & Labels
        for input in inputs:
            # Predict
            self.forward(input)

            # Show
            print(f'{round(max(self.activation2_output[0]) * 100, 2)}% confident Image is a {"ABC"[np.argmax(self.activation2_output)]}')

            plt.imshow(input.reshape(5, 6))
            plt.show()

# Initialize Network
multiLayerPerceptron = MLP(input_size = 30, hidden_size = 5, output_size = 3, num_epochs = 1_000, learning_rate = 0.1)

##### Training #####
a = np.array([[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]])
b = np.array([[0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0]])
c = np.array([[0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]])

y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x = [a, b, c]

multiLayerPerceptron.train(x, y)

##### Testing #####
multiLayerPerceptron.test(x)
