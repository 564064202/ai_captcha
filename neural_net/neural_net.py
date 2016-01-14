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

    def backpropagate(self, input_vector, targets):
        """
        Backpropagation algorithm
        """
        pass

    def __int__(self):
        super(self.__class__, self).__init__()
