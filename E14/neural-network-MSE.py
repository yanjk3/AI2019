# -*- coding: utf-8 -*
# Author: Junkai-Yan
# Finished in 2019/12/12
# this file solve a classification problem using hand write NN
# loss function is MSE
# optimizer is GD

import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class NeuralNetwork:
    """
    this class indicates the entire neural network
    """
    LEARNING_RATE = 0.01

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        """
        create neural layer
        :param num_inputs: input dimension
        :param num_hidden: hidden dimension
        :param num_outputs: output dimension (num of classification)
        :param hidden_layer_weights: weights of hidden layer (default is None)
        :param hidden_layer_bias: bias of hidden layer (default is None)
        :param output_layer_weights: weights of output layer (default is None)
        :param output_layer_bias: bias of output layer (default is None)
        """
        self.num_inputs = num_inputs

        # create neural layer(generate the framework of network)
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        # init weights of neural layer
        self.init_hidden_layer(hidden_layer_weights)
        self.init_output_layer(output_layer_weights)

    def init_hidden_layer(self, hidden_layer_weights):
        """
        inti weights of hidden layer
        :param hidden_layer_weights: initial weights (default is None)
        :return: None
        """
        weight_num = 0
        # for every neuron
        for h in range(len(self.hidden_layer.neurons)):
            # for every input dimension (input_dimension)
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_output_layer(self, output_layer_weights):
        """
        inti weights of output layer
        :param output_layer_weights: initial weights (default is None)
        :return: None
        """
        weight_num = 0
        # for every neuron
        for o in range(len(self.output_layer.neurons)):
            # for every input dimension (hidden layer's quantity of neurons)
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        """
        print parameters of the network
        """
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        """
        forward function, calculate output of the network
        :param inputs: inputs vector
        :return: output of the network
        """
        # get output from hidden layer
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        # get output from output layer
        self.output = self.output_layer.feed_forward(hidden_layer_outputs)
        return self.output

    def train(self, training_inputs, training_outputs):
        """
        training function
        firstly, calculate network's output by forwarding
        secondly, calculate loss, backward and calculate derivative
        thirdly, update parameters by using gradient decrease
        :param training_inputs: input data
        :param training_outputs: label of output
        :return: None
        """
        # forward
        self.feed_forward(training_inputs)
        # calculate loss and backward
        # for output layer
        output_derivative = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # loss function's derivative * sigmoid's derivative
            output_derivative[o] = self.output_layer.neurons[o].calculate_derivative(training_outputs[o])
        # for hidden layer
        hidden_derivative = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # weighted sum's derivative * sigmoid's derivative
            weighted_sum_derivative = 0
            for o in range(len(self.output_layer.neurons)):
                weighted_sum_derivative += output_derivative[o] * self.output_layer.neurons[o].weights[h]
            hidden_derivative[h] = weighted_sum_derivative * self.hidden_layer.neurons[h].sigmoid_derivative()

        # update parameters
        # for output layer
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # derivative for every weight
                w_derivative = output_derivative[o] * self.output_layer.neurons[o].linear_derivative(w_ho)
                # gradient decrease
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * w_derivative
                # print('after', self.output_layer.neurons[o].weights[w_ho])
                # a=input()
        # for hidden layer
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # derivative for every weight
                w_derivative = hidden_derivative[h] * self.hidden_layer.neurons[h].linear_derivative(w_ih)
                # gradient decrease
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * w_derivative

        return self.calculate_total_error(training_outputs)

    def calculate_total_error(self, label):
        """
        calculate MSE loss
        :param label: label
        :return: output, loss
        """
        total_error = 0
        for o in range(len(self.output_layer.neurons)):
            total_error += 0.5*(self.output[o]-label[o])**2
        return self.output, total_error

class NeuronLayer:
    """
    this class indicates a neuron layer of a neuron network
    """
    def __init__(self, num_neurons, bias):
        """
        create layer by appending multiple neurons
        :param num_neurons: total quantity of neurons
        :param bias: bias
        """
        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        # generate neuron for layer
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        """
        print parameters of the layer
        """
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        """
        forward function, calculate output of the layer
        :param inputs: input vector
        :return: output of the layer
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        """
        get output
        :return: output of the layer
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    """
    this class indicates single neuron of a layer
    """
    def __init__(self, bias):
        """
        init function, to init bias and weights
        :param bias: the bias of this neuron
        """
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        """
        calculate the output of the neuron
        :param inputs: inputs array
        :return: output array
        """
        self.inputs = inputs
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        """
        calculate input .* weight + bias
        :return: result
        """
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def sigmoid(self, total_net_input):
        """
        non-linear function, active function is sigmoid, normalize output into interval (0, 1)
        :param total_net_input: output before normalization
        :return: output after normalization
        """
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_error(self, target_output):
        """
        loss function mean square error(MSE)
        :param target_output: label
        :return: MSE between label and output
        """
        return 0.5 * (target_output - self.output) ** 2

    def calculate_derivative(self, target_output):
        """
        calculate derivative for output layer by using chain rule
        :param target_output: label
        :return: derivative of output layer
        """
        return self.mse_derivative(target_output) * self.sigmoid_derivative()

    def mse_derivative(self, target_output):
        """
        function to calculate derivative of MSE, the derivative formula calculated by hand
        :param target_output: label
        :return: derivative of MSE
        """
        return -(target_output - self.output)

    def sigmoid_derivative(self):
        """
        function to calculate derivative of sigmoid, the derivative formula calculated by hand
        :return: derivative of sigmoid
        """
        return self.output * (1 - self.output)

    def linear_derivative(self, index):
        """
        function to calculate derivative of linear, the derivative formula calculated by hand
        :param index: the position of x in w*x
        :return: x, the coefficient of w
        """
        return self.inputs[index]


def get_data(filename):
    """
    :param filename: name of file
    :return: data set (list)
    """
    data_set = pd.read_csv(filename)
    data_set = data_set.values
    data_set_n = np.zeros(data_set.shape)
    print(data_set.shape)
    for i in range(data_set.shape[1] - 1):
        mean = data_set.mean()
        d = data_set.std()
        data_set_n[:,i] = (data_set[:,i]-mean)/d
    data_set_n[:, -1] = data_set[:,-1]
    data = list()
    for i in data_set_n:
        data.append(list(i))
    return data

if __name__=="__main__":
    training_data = get_data('horse-colic-data.csv')
    testing_data = get_data('horse-colic-test.csv')
    # neural network with 35 input attribution and 3 classification
    nn = NeuralNetwork(35, 15, 3)

    loss_in_train = []
    loss_in_test = []
    acc_in_train = []
    acc_in_test = []
    for epoch in range(300):
        if epoch == 250:
            nn.LEARNING_RATE = 0.001
        # train
        loss = 0.0
        count = 0
        for data in training_data:
            out=[0]*3
            out[ int(data[-1])-1 ]=1
            out_, loss_single = nn.train(data[:-1], out)
            loss += loss_single
            if out_.index(max(out_))+1 == int(data[-1]):
                count += 1
        loss_in_train.append(loss/len(training_data))
        acc_in_train.append(count/len(training_data))
        random.shuffle(training_data)

        # test
        loss = 0.0
        count = 0
        for data in testing_data:
            out=[0]*3
            out[ int(data[-1])-1 ]=1
            nn.feed_forward(data[:-1])
            out_, loss_single = nn.calculate_total_error(out)

            loss += loss_single
            if out_.index(max(out_))+1 == data[-1]:
                count += 1
        loss_in_test.append(loss/len(testing_data))
        acc_in_test.append(count/len(testing_data))

    plt.figure('Loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.plot(loss_in_train, label='training loss')
    plt.plot(loss_in_test, label='testing loss')
    plt.legend()

    plt.figure('Accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.plot(acc_in_train, label='training acc')
    plt.plot(acc_in_test, label='testing acc')
    plt.legend()
    plt.show()

