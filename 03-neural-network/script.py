from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Same seed every time for debugging
        random.seed(1)

        # 1 neuron with 3 connections with values between -1 and 1
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # normalizes the weights
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        for i in range(number_of_iterations):
            # pass through the neural network
            output = self.think(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output

            # adjust the weights
            adjustement = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustement

    def think(self, input):
        return self.__sigmoid(dot(input, self.synaptic_weights))

def run():
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights")
    print((neural_network.synaptic_weights))

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Training the neural network
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New weights after training")
    print((neural_network.synaptic_weights))

    # Test it in a new situation
    print("New situation: [1, 0, 0]")
    print((neural_network.think(array([1, 0, 0]))))

if __name__ == "__main__":
    run()
