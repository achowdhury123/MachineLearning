# The following python class defines a three layer artificial neural network.
# The neural network's cost function is based on the derivation found in Tariq Rashid's
# book "Make Your Own Neural Network". The neural network is optimized using mini-batch
# gradient descent.

import numpy
import pandas
import scipy.special
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot
import pickle

class neural_network:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.input_hidden_weights = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.hidden_output_weights = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    def sigmoid_function(self, inputs):
        outputs = 1/(1 + numpy.exp(-inputs))
        return outputs

    def forward_propogation(self, inputs):
        hidden_node_inputs = numpy.dot(self.input_hidden_weights, inputs)
        hidden_node_outputs = self.sigmoid_function(hidden_node_inputs)
        output_node_inputs = numpy.dot(self.hidden_output_weights, hidden_node_outputs)
        outputs = self.sigmoid_function(output_node_inputs)
        return outputs, hidden_node_outputs

    def train(self, learning_rate, training_inputs, expected_outputs, iterations, batch_size):
        training_inputs = numpy.array(training_inputs, ndmin = 2)
        expected_outputs = numpy.array(expected_outputs, ndmin = 2)
        for iterations in range(iterations):
            for i in range(0, training_inputs.shape[0], batch_size):
                input_training_batch = training_inputs[i:i + batch_size]
                expected_outputs_batch = expected_outputs[i:i + batch_size]
                hidden_output_weight_gradient = numpy.zeros((self.output_nodes, self.hidden_nodes))
                input_hidden_weight_gradient = numpy.zeros((self.hidden_nodes, self.input_nodes))
                for rows in range(input_training_batch.shape[0]):
                    input_row = numpy.array(input_training_batch[rows], ndmin = 2).T
                    expected_output_row = numpy.array(expected_outputs_batch[rows], ndmin = 2).T
                    output_row, hidden_layer_row = self.forward_propogation(input_row)
                    print(expected_output_row, output_row)
                    output_error = expected_output_row - output_row
                    hidden_layer_error = numpy.dot(self.hidden_output_weights.T, output_error)
                    hidden_output_weight_gradient+=learning_rate*numpy.dot(output_error*output_row*(1-output_row), numpy.transpose(hidden_layer_row))
                    input_hidden_weight_gradient+=learning_rate*numpy.dot(hidden_layer_error*hidden_layer_row*(1-hidden_layer_row), numpy.transpose(input_row))
                self.hidden_output_weights+=hidden_output_weight_gradient
                self.input_hidden_weights+=input_hidden_weight_gradient
        pass

    def test(self, inputs):
        inputs = numpy.array(inputs, ndmin = 2).T
        outputs, hidden_layer_row = self.forward_propogation(inputs)
        return outputs
