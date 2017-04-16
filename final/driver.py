import numpy as np
from sklearn import linear_model, svm
import time

# module to load data
print('Start loading data...')
import loadData as data
'''
	X_train is an m_train x n array
	y_train is a 1 x m_train array
	X_test is an m_test x n array
	y_test is a 1 x m_test array
'''
X_train = data.X_train
y_train = data.y_train
X_test = data.X_test
y_test = data.y_test
print('Number of trainimng samples: {0:4d}'.format(len(y_train)))
print('Number of test samples: {0:4d}'.format(len(y_test)))

def training(model, modelName, X_train, y_train, X_test, y_test):
	'''
		Produce the training and testing accuracy for a given model
	'''
	print('Start training the {} model...'.format(modelName))
	t_begin = time.time()
	model.fit(X_train, y_train)
	t_end = time.time()
	print('================')
	print('Time elapsed for training: {t:4.2f} seconds'.format(t = t_end - t_begin))
	# accuracy report
	print('Testing the {} model...'.format(modelName))
	print('Training accuracy: {a:4.4f}'.format(a = model.score(X_train, y_train)))
	print('Testing accuracy: {a:4.4f}'.format(a = model.score(X_test, y_test)))
	print('================')

def logreg(X_train, y_train, X_test, y_test):
	''' 
		Produce logistic regression accuracy based on the training set and
		the test set
	'''
	print('Setting up model for logistic regression...')
	model = linear_model.LogisticRegression(C=5.0, verbose = False)
	training(model, 'logistic regression', X_train, y_train, X_test, y_test)

def linearSVM(X_train, y_train, X_test, y_test):
	''' 
		Produce the linear support vector machine report based on the training set and 
		the test set
	'''
	print('Setting up model for linear support vector machine...')
	model = svm.LinearSVC(C = 5.0, verbose = 0)
	training(model, 'linear SVM', X_train, y_train, X_test, y_test)

def kernelSVM(X_train, y_train, X_test, y_test):
	''' 
		Produce the rbf-kernel support vector machine report based on the training set and 
		the test set
	'''
	print('Setting up model for rbf-kernel support vector machine...')
	model = svm.SVC(C = 5.0, verbose = 0)
	training(model, 'rbf-kernel SVM', X_train, y_train, X_test, y_test)


# main driver function
if __name__ == '__main__':
	logreg(X_train, y_train, X_test, y_test)
	linearSVM(X_train, y_train, X_test, y_test)
	kernelSVM(X_train, y_train, X_test, y_test)
