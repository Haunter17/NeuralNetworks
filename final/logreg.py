import numpy as np
from sklearn import linear_model
import time

# module to load data
print('Start loading data...')
import loadData as data
X_train = data.X_train
y_train = data.y_train
X_test = data.X_test
y_test = data.y_test
print('Number of training samples: {0:4d}'.format(len(y_train)))
print('Number of test samples: {0:4d}'.format(len(y_test)))

print('Setting up model for logistic regression...')
# set up logistic regression model
model = linear_model.LogisticRegression(C=5.0)
t_begin = time.time()
model.fit(X_train, y_train)
t_end = time.time()
print('Time elapsed for training: {t:4.2f} seconds'.format(t = t_end - t_begin))

# accuracy report
print('Testing the model...')
accuracy = model.score(X_test, y_test)
print('Logistic regression accuracy: {a:4.4f}'.format(a = accuracy))
