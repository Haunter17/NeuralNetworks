# code based on Prof. Keller's starter code
import cancer
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
    import math
    l2norm = math.sqrt(inner(vec, vec))
    if l2norm == 0:
        return vec
    return [num * 1.0 / l2norm for num in vec]


def train(samples, limit, verbose=False, norm=True, rate=0.1):
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
        if verbose:
            print 'epoch {0} : weights = {1}'.format(epoch, [round(x, 3) for x in weights])
        # random.shuffle(samples)
        for sample in samples:
            inputSamp = [1] + sample[0]
            # normalize sample if necessary
            if norm:
                inputSamp = normalize(inputSamp)

            desired = sample[1]
            output = inner(weights, inputSamp)
            error = desired - output
            predError = desired - threshold(weights, inputSamp)

            for i in range(0, n):
                weights[i] = weights[i] + rate * error * inputSamp[i]    # update
            if verbose and error != 0:
                    print 'input = {} desired = {} output = {} error = {} new weights = {}'\
                    .format([round(x, 3) for x in inputSamp], desired, round(output, 3), round(error, 3), [round(x, 3) for x in weights])
        epoch = epoch + 1
        # validate weights
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
    print 'final weights = {}'.format([round(x, 3) for x in weights])
    return [epoch-1, wrong, weights]

def test(samples, weights, verbose=False, norm=True):
    """ Test a perceptron with the set of samples. """
    n = len(weights)   
    nsamples = len(samples)

    wrong = 0
    for sample in samples:
        inputSamp = [1] + sample[0]
        if norm:
            inputSamp = normalize(inputSamp)
        desired = sample[1]
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
majoritySamples = [[[0, 0, 0], 0], [[0, 0, 1], 0], [[0, 1, 0], 0], \
[[1, 0, 0], 0], [[0, 1, 1], 1], [[1, 0, 1], 1], \
[[1, 1, 0], 1], [[1, 1, 1], 1]]

# weights = train(iffSamples, 500, verbose=True, norm=False, rate=0.001)
# print weights
weights = train(cancer.cancertrainingSamples, 173, verbose=False, norm=False, rate=0.01)
print weights
# test(cancer.cancertrainingSamples, weights, verbose=False, norm=False)
# test(cancer.cancertestSamples, weights, verbose=False, norm=False)
