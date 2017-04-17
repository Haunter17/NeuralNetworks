import numpy as np
from sklearn import linear_model, svm, neural_network
import time
import matplotlib.pyplot as plt

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

def plot_learning_curve(model, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    model : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the model is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    from sklearn.model_selection import learning_curve
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    print('Generating learning curve...')
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def learning_curve_wrapper(model, title, X, y, train_sizes = np.array([0.2, 0.5, 0.8, 1])):
	n_samples = min(35000, len(X))
	X = X[:n_samples, :]
	y = y[:n_samples]
	plot_learning_curve(model, title, X, y, train_sizes = train_sizes)
	plt.show()

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

def logreg(X_train, y_train, X_test, y_test, reg = 0.2):
	''' 
		Produce logistic regression accuracy based on the training set and
		the test set
	'''
	print('Setting up model for logistic regression...')
	model = linear_model.LogisticRegression(C= 1.0 / reg, verbose = False)
	training(model, 'logistic regression', X_train, y_train, X_test, y_test)
	title = 'Learning Curve (Logistic Regression, $\lambda = {}$)'.format(reg)
	# learning_curve_wrapper(model, title, X_train, y_train)

def linearSVM(X_train, y_train, X_test, y_test, reg = 0.2):
	''' 
		Produce the linear support vector machine report based on the training set and 
		the test set
	'''
	print('Setting up model for linear support vector machine...')
	model = svm.LinearSVC(C = 1.0 / reg, verbose = 0)
	training(model, 'linear SVM', X_train, y_train, X_test, y_test)
	title = 'Learning Curve (Linear SVM, $\lambda = {}$)'.format(reg)
	learning_curve_wrapper(model, title, X_train, y_train)

def kernelSVM(X_train, y_train, X_test, y_test, reg = 0.2):
	''' 
		Produce the rbf-kernel support vector machine report based on the training set and 
		the test set
	'''
	print('Setting up model for rbf-kernel support vector machine...')
	model = svm.SVC(C = 1.0 / reg, verbose = 0)
	training(model, 'rbf-kernel SVM', X_train, y_train, X_test, y_test)

def MLP(X_train, y_train, X_test, y_test, reg = 0.01):
	''' 
		Produce the multilayer perceptron training report based on the training set and 
		the test set
	'''
	print('Setting up MLP model')
	model = neural_network.MLPClassifier(alpha = reg, hidden_layer_sizes = (100, 100, 100,))
	training(model, 'MLP', X_train, y_train, X_test, y_test)
	title = 'Learning Curve (MLP, $\lambda = {}$, hidden = [100, 100, 100])'.format(reg)
	learning_curve_wrapper(model, title, X_train, y_train)

# main driver function
if __name__ == '__main__':
	# logreg(X_train, y_train, X_test, y_test)
	# linearSVM(X_train, y_train, X_test, y_test)
	# kernelSVM(X_train, y_train, X_test, y_test)
	MLP(X_train, y_train, X_test, y_test)
