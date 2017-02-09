#RTRL Learning by Zhepei Wang
import numpy as np
def parseIO(X):
    X = np.matrix(X).transpose()
    return X

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def initW(row, col):
    return np.matrix(np.random.rand(row, col))

# config
X = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
y = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

X = parseIO(X)
y = parseIO(y)
inputDim = X.shape[0]
numSample = X.shape[1]
outputDim = y.shape[1]
hiddenDim = outputDim


X = np.hstack((X, np.matrix([[0] for i in range(inputDim)])))

W = initW(hiddenDim, inputDim + hiddenDim)
H = np.matrix([[0. for col in range(numSample + 1)] for row in range(hiddenDim)])
Z = np.vstack((X, H))

def forwardFeed(t):
    # H = np.matrix([[0. for col in range(numSample + 1)] for row in range(hiddenDim)])
    # Z = np.vstack((X, H))
    # for t in range(1, numSample + 1):
    #     Z[inputDim:, t] = sigmoid(W * Z[:, t - 1])
    # return Z[inputDim:, 1:]
    Z[inputDim:, t] = sigmoid(W * Z[:, t - 1])
    return Z[inputDim: , t]

