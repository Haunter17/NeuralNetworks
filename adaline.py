# code based on Prof. Keller's starter code
import cancer
import math
def threshold(weights, input):
    """ The transfer function for a Perceptron """
    """     weights is the weight vector """
    """     input is the input vector """
    return 1 if inner(weights, input) > 0.5 else 0

def inner(x, y):
    """ Returns the inner product of two vectors with the same number of components """
    n = len(x)
    assert len(y) == n
    val = 0
    for i in range(0, n):
        val = val + x[i]*y[i]
    return val

def normalize(vec):
    """ Taking a vector and normalize it """
    l2norm = math.sqrt(inner(vec, vec))
    if l2norm == 0:
        return vec
    return [num * 1.0 / l2norm for num in vec]

def MSE(weights, samples):
    accum = 0
    m = len(samples)
    for sample in samples:
        inputSamp = [1] + sample[0]
        desired = sample[1]
        output = inner(weights, inputSamp)
        diff = desired - output
        accum += (diff ** 2)
    return accum * 1.0 / m

def train(samples, limit, verbose=False, freq=100, norm=True, rate=0.1, regression=False, epsilon=1.0):
    """ Train a perceptron with the set of samples. """
    """ samples is a list of pairs of the form [input vector, desired output] """
    """ limit is a limit on the number of epochs """
    """ Returns a list of the number of epochs used, """
    """                   the number wrong at the end, and """
    """                   the final weights """
    import random
    weights = [0] + map(lambda x:0, samples[0][0]) # initialize weights to all 0
    n = len(weights)   
    nsamples = len(samples)
    wrong = nsamples # initially assume every classification is wrong
    epoch = 1

    while epoch <= limit:
        if verbose and epoch % freq == 0:
            print 'epoch {0} : weights = {1}'.format(epoch, [round(x, 3) for x in weights])
        for sample in samples:
            inputSamp = [1] + sample[0]
            # normalize sample if necessary
            if norm:
                inputSamp = normalize(inputSamp)
            desired = sample[1]
            output = inner(weights, inputSamp)
            error = desired - output

            for i in range(0, n):
                weights[i] = weights[i] + rate * error * inputSamp[i]    # update
            if verbose and error != 0 and epoch % freq == 0:
                    print 'input = {} desired = {} output = {} error = {} new weights = {}'\
                    .format([round(x, 3) for x in inputSamp], desired, round(output, 3), \
                        round(error, 3), [round(x, 3) for x in weights])
        epoch = epoch + 1
        # validate weights for classification
        if not regression:
            wrong = 0
            for sample in samples:
                inputSamp = [1] + sample[0]
                if norm:
                    inputSamp = normalize(inputSamp)
                desired = sample[1]
                output = inner(weights, inputSamp)
                predError = desired - threshold(weights, inputSamp)
                if predError != 0:
                    wrong = wrong + 1
            if wrong == 0:
                break
        else: # validate weights for regression
            newMSE = MSE(weights, samples)
            if  verbose and epoch % freq == 0:
                print "epoch {} : MSE is {}".format(epoch, round(newMSE, 3))
            if newMSE < epsilon:
                print "epoch {} : MSE is {}".format(epoch, round(newMSE, 3))
                break

    print 'number of epochs: {}, final weights = {}'.format(epoch - 1, [round(x, 3) for x in weights])
    return [epoch-1, wrong, weights]

def test(samples, weights, verbose=False, norm=True, regression=False):
    """ Test a perceptron with the set of samples. """
    n = len(weights)   
    nsamples = len(samples)

    if regression:
        mse = MSE(weights, samples)
        for sample in samples:
            inputSamp = [1] + sample[0]
            desired = sample[1]
            output = inner(weights, inputSamp)
            error = desired - output
            if verbose:
                print 'input = {} desired = {} output = {}'\
                    .format([round(x, 3) for x in inputSamp], desired, round(output, 3))
        print "MSE: {}".format(round(mse, 3))
        return mse
    else:
        wrong = 0
        for sample in samples:
            inputSamp = [1] + sample[0]
            if norm:
                inputSamp = normalize(inputSamp)
            desired = sample[1]
            output, error = 0, 0
            output = threshold(weights, inputSamp)
            error = desired - output
            if error != 0:
                wrong = wrong + 1
                if verbose:
                    print '{} : input = {} desired = {} output = {} error = {}'\
                    .format(wrong, [round(x, 3) for x in inputSamp], desired, output, error)
        print 'percentage of errors = {}'.format(round(wrong * 1.0 / nsamples, 4))
        return wrong

implies = [[[0, 0], 1], [[0, 1], 1], [[1, 0], 0], [[1, 1], 1]]
nandSamples = [[[0, 0], 1], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]
iffSamples = [[[0, 0], 1], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]]

widrowHoffTrain = [[[1,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0], 60], [[1,1,1,0,1,0,0,0,1,1,1,0,1,1,1,0], 0], \
[[1,1,1,0,1,1,0,0,1,0,0,0,1,0,0,0], -60]]
widrowHoffTest = [[[0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,0], 60], [[0,1,1,1,0,1,0,0,0,1,1,1,0,1,1,1], 0], \
[[0,1,1,1,0,1,1,0,0,1,0,0,0,1,0,0], -60]]

hvTest = [[[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0], 0], [[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0], 0], \
[[0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0], 0], [[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], 0], \
[[1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], 1],[[0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0], 1], \
[[0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0], 1], [[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1], 1]]
# import random
# random.shuffle(hvTest)
hvTrain = hvTest[:2] + hvTest[3:4] + hvTest[4:6] + hvTest[7:]
# weights = train(iffSamples, 500, verbose=True, norm=False, rate=0.001)
# weights = train(cancer.cancertrainingSamples, 200, verbose=False, norm=False, rate=0.01)[2]
# print test(cancer.cancertrainingSamples, weights, verbose=False, norm=False)
# print test(cancer.cancertestSamples, weights, verbose=False, norm=False)

weights = train(widrowHoffTrain + widrowHoffTest, 200, verbose=True, freq=50, norm=False, \
    rate=0.03, regression=True, epsilon=1.0)[2]
test(widrowHoffTrain + widrowHoffTest, weights, verbose=True, norm=False, regression=True)

# weights = train(hvTrain, 3000, verbose=False, freq=1000, norm=False, rate=0.002, regression=True, epsilon=0.05)[2]
# test(hvTest, weights, verbose=True, norm=False, regression=True)
