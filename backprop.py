# backprop.py
# author: Robert M. Keller
# date: 31 January 2016

# description:
# FFnet is a basic multilevel neural network class
# The number of layers is arbitrary.
# Each layer has a function, as specified using the class ActivationFunction.
# Each layer has its own learning rate.

import math
import random
from random import shuffle
import copy

import hw3data
xorSamples = hw3data.xorSamples
vhSamples = hw3data.vhSamples
sineSamples = hw3data.sineSamples
toBinarySamples = hw3data.toBinarySamples
letterSamples = hw3data.letterSamples
cancerTrainingSamples = hw3data.cancerTrainingSamples
cancerTestSamples = hw3data.cancerTestSamples


class FFnet:
    def __init__(nn, name, size, function, rate):
        """ Feedforward Neural Network                                    """
	""" nn is the 'self' reference, used in each method               """
        """ name is a string naming this network.                         """
        """ size is a list of the layer sizes:                            """
        """     The first element of size is understood as the number of  """
        """         inputs to the network.                                """
        """     The remaining elements of size are the number of neurons  """
        """         in each layer.                                        """
        """     Therefore the last element is the number of outputs       """
        """         of the network.                                       """
        """ function is a list of the activation functions in each layer. """
        """ deriv is a list of the corresponding function derivatives.    """
        """ rate is a list of the learning rate for each layer.           """

        nn.name = name
        nn.size = size
        nn.output = [[0 for i in range(s)] # output values for all layers, 
                        for s in size]     # counting the input as layer 0

        nn.range1 = range(1, len(size))    # indices excluding the input layer
        nn.lastLayer = len(size)-1         # index of the last layer
        size1 = size[1:]                   # layer sizes excluding the input layer

        # dummy is used because the input layer does not have weights
        # but we want the indices to conform.
        dummy = [[]]
        nn.function = dummy + function
        nn.rate = dummy + rate

        # initialize weights and biases
        nn.weight = dummy + [[[randomWeight() for synapse in range(size[layer-1])]
                                              for neuron in range(size[layer])] 
                                              for layer in nn.range1]

        nn.bias = dummy+[[randomWeight() for neuron in range(layer)] 
                                         for layer in size1]

        nn.sensitivity = dummy + [[0 for neuron in range(layer)] 
                                     for layer in size1]

        nn.act = dummy + [[0 for i in range(layer)] 
                             for layer in size1]


        # implementation for momentum
        nn.momentumW = dummy + [[[0 for synapse in range(size[layer-1])]
                                              for neuron in range(size[layer])] 
                                              for layer in nn.range1]

        nn.momentumB = dummy+[[0 for neuron in range(layer)] 
                                         for layer in size1]

        # implementation for VLR
        nn.prevW = copy.deepcopy(nn.weight)
        nn.prevB = copy.deepcopy(nn.bias)
        nn.prevSens = copy.deepcopy(nn.sensitivity)
        nn.prevAct = copy.deepcopy(nn.act)
        nn.prevOut = copy.deepcopy(nn.output)

    def describe(nn, noisy):
        """ describe prints a description of this network. """
        print "---------------------------------------------------------------"
        print "network", nn.name + ":"
        print "size =", nn.size
        print "function =", map(lambda x:x.name, nn.function[1:])
        print "learning rate =", nn.rate[1:]
        if noisy:    
            print "weight =", roundall(nn.weight[1:], 3)
            print "bias =", roundall(nn.bias[1:], 3)

    def forward(nn, input):
        """ forward runs the network, given an input vector. """
        """ All act values and output values are saved.      """
        """ The output of the last layer is returned as a    """
        """ convenience for later testing.                   """

        nn.output[0] = input # set input layer

        # Iterate over all neurons in all layers.

        for layer in nn.range1:
            fun = nn.function[layer].fun
            for neuron in range(nn.size[layer]):
                # compute and save the activation
                nn.act[layer][neuron] = nn.bias[layer][neuron] \
                           + inner(nn.weight[layer][neuron], nn.output[layer-1])
                # compute the output
                nn.output[layer][neuron] = fun(nn.act[layer][neuron])

        return nn.output[-1]

    def backward(nn, desired):
        """ backward runs the backpropagation step, """
        """ computing and saving all sensitivities  """
        """ based on the desired output vector.     """

        # Iterate over all neurons in the last layer.        
        # The sensitivites are based on the error and derivatives
        # evaluated at the activation values, which were saved during forward.

        deriv = nn.function[nn.lastLayer].deriv
        for neuron in range(nn.size[nn.lastLayer]):
            error = desired[neuron] - nn.output[nn.lastLayer][neuron]
            nn.sensitivity[nn.lastLayer][neuron] = \
                error*deriv(nn.act[nn.lastLayer][neuron], \
                            nn.output[nn.lastLayer][neuron])

        # Iterate backward over all layers except the last.
        # The sensitivities are computed from the sensitivities in the following
        # layer, weighted by the weight from a neuron in this layer to the one
        # in the following, times this neuron's derivative.

        for layer in range(nn.lastLayer-1, 0, -1):
            deriv = nn.function[layer].deriv
            # preNeuron is the neuron from which there is a connection
	    # postNeuron is the neuron to which there is a connection
            for preNeuron in range(nn.size[layer]):
                factor = deriv(nn.act[layer][preNeuron], nn.output[layer][preNeuron])
                sum = 0
                for postNeuron in range(nn.size[layer+1]):
                    sum += nn.weight[layer + 1][postNeuron][preNeuron] \
                          *nn.sensitivity[layer+1][postNeuron]
                nn.sensitivity[layer][preNeuron] = sum*factor

    def update(nn, alpha):
        """ update updates all weights and biases based on the       """
        """ sensitivity values learning rate, and inputs to          """
        """ this layer, which are the outputs of the previous layer. """

        for layer in nn.range1:
            for neuron in range(nn.size[layer]):
                factor = nn.rate[layer]*nn.sensitivity[layer][neuron]
                # nn.bias[layer][neuron] += factor
                deltaB = factor + alpha * nn.momentumB[layer][neuron]
                nn.bias[layer][neuron] += deltaB
                nn.momentumB[layer][neuron] = deltaB
                for synapse in range(nn.size[layer-1]):
                    # nn.weight[layer][neuron][synapse] \
                    #     += factor*nn.output[layer-1][synapse]
                    deltaW = factor*nn.output[layer-1][synapse] + alpha * nn.momentumW[layer][neuron][synapse]
                    nn.weight[layer][neuron][synapse] += deltaW 
                    nn.momentumW[layer][neuron][synapse] = deltaW

    def learn(nn, input, desired, alpha):
        """ learn learns by forward propagating input,  """
        """ back propagating error, then updating.      """
        """ It returns the output vector and the error. """

        nn.forward(input)
        nn.backward(desired)
        nn.update(alpha)
        output = nn.output[-1]
        error = subtract(desired, output)
        return [output, error]

    def assess(nn, sample, noisy):
        """ Assess the classification performance of a sample.             """
        """ returns 1 or 0 indicating correct or incorrect classification. """

        [input, desired] = sample
        nn.forward(input)
        output = nn.output[-1]
        error = subtract(desired, output)
        wrong = countWrong(error, 0.5)
        if noisy:
            print nn.name, "input =", input, \
                  "desired =", desired, \
                  "output =", roundall(output, 3), \
                  "error =", roundall(error, 3), \
                  "wrong =", wrong
        return wrong
    
    def train(nn, samples, epochs, displayInterval, noisy, \
        alpha=0, VLR=False, max_perf_inc=1.04, lr_dec=0.7, lr_inc=1.05):
        """ Trainsthe network using the specified set of samples,    """
        """ for the specified number of epochs.                      """
        """ displayInterval indicates how often to display progress. """
        """ If using as a classifier, assumes the first component in """
        """ the output vector is the classification.                 """

        previousMSE = float("inf")
        for epoch in range(epochs):
            shuffle(samples)
            SSE = 0
            wrong = 0
            for [x, y] in samples:
                [output, error] = nn.learn(x, y, alpha)
                SSE += inner(error, error)/len(output)
                wrong += countWrong(error, 0.5)
            MSE = SSE/len(samples)
            wrongpc = 100.0*wrong/(len(samples)*len(output))
            if wrong == 0:
                break   # stop if classification is correct
            # implementation of VLR
            direction = "decreasing" if MSE < previousMSE else "increasing"
            if VLR:
                if MSE >= max_perf_inc * previousMSE:
                    # discard weights and biases
                    nn.weight = copy.deepcopy(nn.prevW)
                    nn.bias = copy.deepcopy(nn.prevB)
                    nn.sensitivity = copy.deepcopy(nn.prevSens)
                    nn.act = copy.deepcopy(nn.prevAct)
                    nn.output = copy.deepcopy(nn.prevOut)
                    # decrement rates
                    for index in range(1, len(nn.rate)):
                        nn.rate[index] *= lr_dec
                else:
                    # keep new weights and biases
                    nn.prevW = copy.deepcopy(nn.weight)
                    nn.prevB = copy.deepcopy(nn.bias)
                    nn.prevSens = copy.deepcopy(nn.sensitivity)
                    nn.prevAct = copy.deepcopy(nn.act)
                    nn.prevOut = copy.deepcopy(nn.output)
                    if direction == "decreasing":
                        # increment rates
                        for index in range(1, len(nn.rate)):
                            nn.rate[index] *= lr_inc
                
            if epoch%displayInterval == 0:
                print nn.name, "epoch", epoch, "MSE =", round(MSE, 3), "wrong =", \
                    str(wrong) + " (" + str(round(wrongpc, 3)) + "%)", direction
            previousMSE = MSE

        if noisy:
            print nn.name, "final weight =", roundall(nn.weight[1:], 3)
            print nn.name, "final bias =", roundall(nn.bias[1:], 3)
        wrong = 0
        for sample in samples:
            wrong += nn.assess(sample, noisy)
        wrongpc = 100.0*wrong/(len(samples)*len(output))
        print nn.name, "final MSE =", round(MSE, 3), "final wrong =", \
                    str(wrong) + " (" + str(round(wrongpc, 3)) + "%)"

    def assessAll(nn, samples):
        """ Assess the network using the specified set of samples.   """
        """ Primarily used for testing an already-trained network.   """
        """ If using as a classifier, assumes the first component in """
        """ the output vector is the classification.                 """

        SSE = 0
        wrong = 0
        for [x, y] in samples:
            output = nn.forward(x)
            error = subtract(y, output)
            # print "input: {}, desired: {}, output:{}".format(x, y, \
            #     [round(val, 3) for val in output])
            SSE += inner(error, error)/len(output)
            wrong += countWrong(error, 0.5)
        MSE = SSE/len(samples)
        wrongpc = 100.0*wrong/(len(samples)*len(output))
        print nn.name, "test MSE =", round(MSE, 3), "test wrong =", \
                    str(wrong) + " (" + str(round(wrongpc, 3)) + "%)"

class ActivationFunction:
    """ ActivationFunction packages a function together with its derivative. """
    """ This prevents getting the wrong derivative for a given function.     """
    """ Because some derivatives are computable from the function's value,   """
    """ the derivative has two arguments: one for the argument and one for   """
    """ the value of the corresponding function. Typically only one is use.  """

    def __init__(af, name, fun, deriv):
        af.name = name
        af.fun = fun
        af.deriv = deriv

    def fun(af, x):
        return af.fun(x)

    def deriv(af, x, y):
        return af.deriv(x, y)

logsig = ActivationFunction("logsig",
                            lambda x: 1.0/(1.0 + math.exp(-x)),
                            lambda x,y: y*(1.0-y))

tansig = ActivationFunction("tansig",
                            lambda x: math.tanh(x),
                            lambda x,y: 1.0 - y*y)

purelin = ActivationFunction("purelin",
                             lambda x: x,
                             lambda x,y: 1)

def randomWeight():
    """ returns a random weight value between -0.5 and 0.5 """
    return random.random()-0.5

def inner(x, y):
    """ Returns the inner product of two equal-length vectors. """
    n = len(x)
    assert len(y) == n
    sum = 0
    for i in range(0, n):
        sum += x[i]*y[i]
    return sum

def subtract(x, y):
    """ Returns the first vector minus the second. """
    n = len(x)
    assert len(y) == n
    return map(lambda i: x[i]-y[i], range(0, n))

def countWrong(L, tolerance):
    """ Returns the number of elements of L with an absolute """
    """ value above the specified tolerance. """
    return reduce(lambda x,y:x+y, \
                  map(lambda x:1 if abs(x)>tolerance else 0, L))

def roundall(item, n):
    """ Round a list, list of lists, etc. to n decimal places. """
    if type(item) is list:
        return map(lambda x:roundall(x, n), item)
    return round(item, n)

# various examples

def xor():
    nnet = FFnet("xor", [2,3,1], [tansig, logsig], [0.5, 0.2])
    nnet.describe(False)
    nnet.train(xorSamples, 1000, 100, False)

def xor2():
    nnet = FFnet("xor2", [2,2,1], [tansig, logsig], [0.5, 0.2])
    nnet.describe(False)
    nnet.train(xorSamples, 1000, 100, False)

def vh():
    nnet = FFnet("vh", [16,6,1], [tansig, logsig], [0.5, 0.2])
    nnet.describe(False)
    nnet.train(vhSamples, 1000, 100, False)

def toBinary():
    nnet = FFnet("toBinary", [1, 16, 3], [tansig, logsig], [0.2, 0.1])
    nnet.describe(False)
    nnet.train(toBinarySamples, 10000, 500, False)

def letters():
    nnet = FFnet("letters", [35,12,26], [logsig, logsig], [0.1, 0.1])
    nnet.describe(False)
    nnet.train(letterSamples, 2000, 100, False)

def sine():
    nnet = FFnet("sine", [1,16, 4, 1], [tansig, tansig, purelin], [0.4, 0.2, 0.1])
    nnet.describe(False)
    nnet.train(sineSamples, 100000, 500, True)

def cancer():
    nnet = FFnet("cancer", [9, 5, 1], [logsig, logsig], [0.5, 0.2])
    nnet.describe(False)
    nnet.train(cancerTrainingSamples, 2000, 100, False, VLR=False)
    nnet.assessAll(cancerTestSamples)

def iff():
    iffSamples = [[[0, 0], [1]], [[0, 1], [0]], [[1, 0], [0]], [[1, 1], [1]]]
    nnet = FFnet("iff", [2,2,1], [logsig, logsig], [0.5, 0.45])
    nnet.describe(False)
    nnet.train(iffSamples, 10000, 100, False)

def autoencoder():
    AEsamples = [
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    ]

    nnet = FFnet("autoencoder", [16,4,16], [logsig, logsig], [0.5, 0.2])
    nnet.describe(False)
    nnet.train(AEsamples, 10000, 100, False, alpha = 0.3)

def wine():
    import wine
    wineSamples = wine.data
    shuffle(wineSamples)
    wineTrainingSamples = wineSamples[:len(wineSamples) * 2 / 3]
    wineTestSamples = wineSamples[len(wineSamples) * 2 / 3:]
    nnet = FFnet("wine", [13,15,15,15,3], [logsig, logsig, logsig, logsig], [0.5, 0.5, 0.5, 0.2])
    nnet.describe(False)
    nnet.train(wineTrainingSamples, 10000, 10, False)
    nnet.assessAll(wineTestSamples)

def main():
    # xor( )
    # xor2()
    # vh()
    # letters()
    # toBinary()
    # sine()
    # cancer()
    iff()
    # autoencoder()
    # wine()
    
main()