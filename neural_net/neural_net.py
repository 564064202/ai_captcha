"""
Author: Rajat Gupta
Neural Network Model
"""
import math


class NeuralNetwork(object):
    """
    Neural Network Library
    """
    def dot(v, w):
        """
        v_1 * w_1 + ... + v_n * w_n
        """
        return sum(v_i * w_i for v_i, w_i in zip(v, w))

    def sigmoid(self, x):
        """
        Sigmoidal function
        """
        return 1/(1 + math.exp(-x))

    def neuron_output(self, weights, inputs):
        """
        y = x * w
        """
        return self.sigmoid(self.dot(weights, inputs))

    def feed_forward(self, neural_network, input_vector):
        """
        Feedforward training
        """
        outputs = []

        for layer in neural_network:
            input_with_bias = input_vector + [1]
            _output = [
                self.neuron_output(neuron, input_with_bias)
                for neuron in layer]
            outputs.append(_output)

            input_vector = _output

        return outputs

    def backpropagate(self, network, input_vector, targets):
        """
        Backpropagation algorithm
        """
        hidden_outputs, outputs = self.feed_forward(network, input_vector)

        output_deltas = [
            output * (1 - output) * (output - target)
            for output, target in zip(outputs, targets)]

        # Adjust weights for output layer
        for i, output_neuron in enumerate(network[-1]):
            for j, hidden_output in enumerate(hidden_outputs + [1]):
                output_neuron[j] -= output_deltas[i] * hidden_output

        # back propagate errors to hidden layer
        hidden_deltas = [
            hidden_output * (1 - hidden_output) *
            self.dot(output_deltas, [n[i] for n in outputs])
            for i, hidden_output in enumerate(hidden_outputs)]

        # Adjust weights of hidden layer
        for i, hidden_neuron in enumerate(network[0]):
            for j, inpu in enumerate(input_vector + [1]):
                hidden_neuron[j] -= hidden_deltas[i] * inpu

    def __int__(self):
        super(self.__class__, self).__init__()
