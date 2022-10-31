##### Imports #####
from tqdm import tqdm
import numpy as np

##### Data #####
train_X = ['good', 'bad', 'happy', 'sad', 'not good', 'not bad', 'not happy', 'not sad', 'very good', 'very bad', 'very happy', 'very sad', 'i am happy', 'this is good', 'i am bad', 'this is bad', 'i am sad', 'this is sad', 'i am not happy', 'this is not good', 'i am not bad', 'this is not sad', 'i am very happy', 'this is very good', 'i am very bad', 'this is very sad', 'this is very happy', 'i am good not bad', 'this is good not bad', 'i am bad not good', 'i am good and happy', 'this is not good and not happy', 'i am not at all good', 'i am not at all bad', 'i am not at all happy', 'this is not at all sad', 'this is not at all happy', 'i am good right now', 'i am bad right now', 'this is bad right now', 'i am sad right now', 'i was good earlier', 'i was happy earlier', 'i was bad earlier', 'i was sad earlier', 'i am very bad right now', 'this is very good right now', 'this is very sad right now', 'this was bad earlier', 'this was very good earlier', 'this was very bad earlier', 'this was very happy earlier', 'this was very sad earlier', 'i was good and not bad earlier', 'i was not good and not happy earlier', 'i am not at all bad or sad right now', 'i am not at all good or happy right now', 'this was not happy and not good earlier']
train_y = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]

test_X = ['this is happy', 'i am good', 'this is not happy', 'i am not good', 'this is not bad', 'i am not sad', 'i am very good', 'this is very bad', 'i am very sad', 'this is bad not good', 'this is good and happy', 'i am not good and not happy', 'i am not at all sad', 'this is not at all good', 'this is not at all bad', 'this is good right now', 'this is sad right now', 'this is very bad right now', 'this was good earlier', 'i was not happy and not good earlier']
test_y = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]

vocab = set([q for text in train_X for q in text.split()])
vocab_size = len(vocab)

word_to_index = {w:i for i, w in enumerate(vocab)}

##### Helper Functions #####
def oneHotEncode(text):
    inputs = []
    for q in text.split():
        vector = np.zeros((1, vocab_size))
        vector[0][word_to_index[q]] = 1
        inputs += [vector]

    return inputs

# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (input_size, output_size)) * np.sqrt(6 / (input_size + output_size))

##### Activation Functions #####
def tanh(input, derivative = False):
    if derivative:
        return 1 - (input ** 2)
    
    return np.tanh(input)

def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))

##### Recurrent Neural Network Class #####
class RNN:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Network
        self.w1 = initWeights(input_size, hidden_size)
        self.w2 = initWeights(hidden_size, hidden_size)
        self.w3 = initWeights(hidden_size, output_size)

        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, output_size))

    # Forward Propogation
    def forward(self, inputs):
        self.hidden_states = [np.zeros_like(self.b2)]

        for input in inputs:
            layer1_output = np.dot(input, self.w1)
            layer2_output = np.dot(self.hidden_states[-1], self.w2) + self.b2
            
            self.hidden_states += [tanh(layer1_output + layer2_output)]

        return np.dot(self.hidden_states[-1], self.w3) + self.b3

    # Backward Propogation
    def backward(self, error, inputs):
        d_b3 = error
        d_w3 = np.dot(self.hidden_states[-1].T, error)

        d_b2 = np.zeros_like(self.b2)
        d_w2 = np.zeros_like(self.w2)
        d_w1 = np.zeros_like(self.w1)

        d_hidden_state = np.dot(error, self.w3.T)
        for q in reversed(range(len(inputs))):
            d_hidden_state *= tanh(self.hidden_states[q + 1], derivative = True)

            d_b2 += d_hidden_state

            d_w2 += np.dot(self.hidden_states[q].T, d_hidden_state)

            d_w1 += np.dot(inputs[q].T, d_hidden_state)

            d_hidden_state = np.dot(d_hidden_state, self.w2)

        for d_ in (d_b3, d_w3, d_b2, d_w2, d_w1):
            np.clip(d_, -1, 1, out = d_)

        self.b3 += self.learning_rate * d_b3
        self.w3 += self.learning_rate * d_w3
        self.b2 += self.learning_rate * d_b2
        self.w2 += self.learning_rate * d_w2
        self.w1 += self.learning_rate * d_w1

    # Train
    def train(self, inputs, labels):
        for _ in tqdm(range(self.num_epochs)):
            for input, label in zip(inputs, labels):
                input = oneHotEncode(input)

                prediction = self.forward(input)

                error = -softmax(prediction)
                error[0][label] += 1

                self.backward(error, input)

    # Test
    def test(self, inputs, labels):
        accuracy = 0
        for input, label in zip(inputs, labels):
            print(input)

            input = oneHotEncode(input)
            prediction = self.forward(input)

            print(['Negative', 'Positive'][np.argmax(prediction)], end = '\n\n')
            if np.argmax(prediction) == label:
                accuracy += 1

        print(f'Accuracy: {round(accuracy * 100 / len(inputs), 2)}%')

# Initialize Network
rnn = RNN(input_size = vocab_size, hidden_size = 64, output_size = 2, learning_rate = 0.02, num_epochs = 1_000)

##### Training #####
rnn.train(train_X, train_y)

##### Testing #####
rnn.test(test_X, test_y)
