#RTRL Learning by Zhepei Wang
import numpy as np
import copy
def parseIO(X):
    X = np.matrix(X).transpose()
    return X

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def initW(row, col):
    return np.matrix(np.random.rand(row, col))

# config
X = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1],\
[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], \
[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], \
[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
y = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], \
[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], \
[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], \
[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

X = parseIO(X)
y = parseIO(y)
inputDim = X.shape[0]
numSample = X.shape[1]
outputDim = y.shape[0]
hiddenDim = outputDim
print outputDim
X = np.hstack((X, np.matrix([[0] for i in range(inputDim)])))
W = initW(hiddenDim, inputDim + hiddenDim)

def forwardFeed(W, Z, t):
    # Z supposed to be updated
    Z[inputDim:, t] = sigmoid(W * Z[:, t - 1])
    return Z[inputDim: , t]

def train(X, y, rate=0.3, max_iter=400, threshold=1.0, displayInterval=50, noisy=False):
    epoch = 0
    global W
    for epoch in range(max_iter):
        H = np.matrix([[0. for col in range(numSample + 1)] for row in range(hiddenDim)])
        Z = np.vstack((X, H))
        MSE = 0
        gradWT = [[[0. for col in range(inputDim + hiddenDim)] \
        for row in range(hiddenDim)] for t in range(numSample + 1)]
        gradW = copy.deepcopy(gradWT[0])

        pTL = [[[[0. for col in range(inputDim + hiddenDim)] \
        for row in range(hiddenDim)] \
        for k in range(hiddenDim)] for t in range(numSample + 1)]
        
        for t in range(1, numSample + 1):
            pred = forwardFeed(W, Z, t)
            e = y[:, t - 1] - pred
            MSE += 0.5 * ((e.transpose() * e)).item(0)
            # calculate p and gradW for sample T
            for i in range(hiddenDim):
                for j in range(inputDim + hiddenDim):
                    accumK = 0
                    for k in range(hiddenDim):
                        accumL = 0
                        for l in range(hiddenDim):
                            accumL += W.item(k, inputDim + l) * pTL[t - 1][l][i][j]
                        if i == k:
                            accumL += Z.item(j, t - 1)
                        pTL[t][k][i][j] = Z.item(inputDim + k, t) * (1. - Z.item(inputDim + k, t)) \
                        * accumL
                        accumK += e.item(k) * pTL[t][k][i][j]
                    gradWT[t][i][j] += accumK
        for t in range(len(gradWT)):
            for i in range(hiddenDim):
                for j in range(inputDim + hiddenDim):
                    gradW[i][j] = gradWT[t][i][j]
        MSE /= numSample
        if epoch % displayInterval == 0:
            print "MSE for epoch {}: {}".format(epoch, MSE)
        gradW = np.matrix(gradW)
        W = W + rate * gradW
    return W

def assess(W, X, y):
    H = np.matrix([[0. for col in range(numSample + 1)] for row in range(hiddenDim)])
    Z = np.vstack((X, H))
    for t in range(1, numSample + 1):
        Z[inputDim:, t] = sigmoid(W * Z[:, t - 1])
    return Z[inputDim:, -1]

Wopt = train(X, y)
print assess(Wopt, X, y)
print y
