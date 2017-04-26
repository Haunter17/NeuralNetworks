import numpy as np
from sklearn import linear_model, svm, neural_network
import time
import matplotlib.pyplot as plt

# module to load data
print('==> Start to load data...')
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
print('-- Number of trainimng samples: {0:4d}'.format(len(y_train)))
print('-- Number of test samples: {0:4d}'.format(len(y_test)))

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
    print('==> Generating learning curve...')
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

def learning_curve_wrapper(model, fname, title, X, y, \
	train_sizes = np.linspace(.1, 1.0, 6), if_show = False):
	n_samples = min(35000, len(X))
	X = X[:n_samples, :]
	y = y[:n_samples]
	plot_learning_curve(model, title, X, y, train_sizes = train_sizes)
	if if_show:
		plt.show()
	else:
		plt.savefig('{}.png'.format(fname), format = 'png')
		plt.close()
	print('==> Plotting completed')

def training(model, modelName, X_train, y_train, X_test, y_test):
	'''
		Produce the training and testing accuracy for a given model
	'''
	print('==> Start training the {} model...'.format(modelName))
	t_begin = time.time()
	model.fit(X_train, y_train)
	t_end = time.time()
	print('================')
	try:
		getattr(model, 'n_iter_')
	except AttributeError:
		print('-- Model {} is not iterative'.format(modelName))
	else:
		print('-- Actual number of iterations: {}'.format(model.n_iter_))
	print('-- Time elapsed for training: {t:4.2f} seconds'.format(t = t_end - t_begin))
	# accuracy report
	print('-- Testing the {} model...'.format(modelName))
	print('-- Training accuracy: {a:4.4f}'.format(a = model.score(X_train, y_train)))
	print('-- Testing accuracy: {a:4.4f}'.format(a = model.score(X_test, y_test)))
	print('================')

def logreg(X_train, y_train, X_test, y_test, reg = 0.2, lc = False):
	''' 
		Produce logistic regression accuracy based on the training set and
		the test set
	'''
	print('********************************')
	print('==> Setting up model for logistic regression...')
	model = linear_model.LogisticRegression(C= 1.0 / reg, verbose = False)
	training(model, 'logistic regression', X_train, y_train, X_test, y_test)
	if lc:
		title = 'Learning Curve (Logistic Regression, $\lambda = {}$)'.format(reg)
		save_file_name = 'logreg'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	print('********************************')

def linearSVM(X_train, y_train, X_test, y_test, reg = 0.2, lc = False):
	''' 
		Produce the linear support vector machine report based on the training set and 
		the test set
	'''
	print('********************************')
	print('==> Setting up model for linear support vector machine...')
	model = svm.LinearSVC(C = 1.0 / reg, verbose = 0)
	training(model, 'linear SVM', X_train, y_train, X_test, y_test)
	if lc:
		title = 'Learning Curve (Linear SVM, $\lambda = {}$)'.format(reg)
		save_file_name = 'linsvm'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	print('********************************')

def kernelSVM(X_train, y_train, X_test, y_test, reg = 0.2, lc = False):
	''' 
		Produce the rbf-kernel support vector machine report based on the training set and 
		the test set
	'''
	print('********************************')
	print(' ==> Setting up model for rbf-kernel support vector machine...')
	model = svm.SVC(C = 1.0 / reg, verbose = 0)
	training(model, 'rbf-kernel SVM', X_train, y_train, X_test, y_test)
	title = 'Learning Curve (rbf-kernel SVM, $\lambda = {}$)'.format(reg)
	if lc:
		save_file_name = 'rbfsvm'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	print('********************************')

def MLP(X_train, y_train, X_test, y_test, reg = 0.01, lc = False):
	''' 
		Produce the multilayer perceptron training report based on the training set and 
		the test set
	'''
	print('********************************')
	print('==> Setting up MLP model')
	model = neural_network.MLPClassifier(alpha = reg, hidden_layer_sizes = (100, 100, 100,))
	training(model, 'MLP', X_train, y_train, X_test, y_test)
	if lc:
		title = 'Learning Curve (MLP, $\lambda = {}$, hidden = [100, 100, 100])'.format(reg)
		save_file_name = 'MLP'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	print('********************************')

def PCA(X, target_pct = 0.99, k = -1):
	'''
		X has dimension m x n.
		Generate principal components of the data.
	'''
	# zero out the mean
	m, n = X.shape
	mu = X.mean(axis = 0).reshape(1, -1)
	X = X - np.repeat(mu, m, axis = 0)
	# unit variance
	var = np.multiply(X, X).sum(axis = 0)
	std = np.sqrt(var).reshape(1, -1)
	X = np.nan_to_num(np.divide(X, np.repeat(std, m, axis = 0)))
	# svd
	U, S, V = np.linalg.svd(X.T @ X)
	if k == -1:
		# calculate target k
		total_var = sum(S ** 2)
		accum = 0.
		k = 0
		while k < len(S):
			accum += S[k] ** 2
			if accum / total_var >= target_pct:
				break
			k += 1
	# projection
	X_rot = X @ U[:, :k + 1]
	return X_rot, S ** 2, k

def PCA_analysis(D, title = 'Relative Variance Preservation', savename = 'PCA.png'):
	'''
		Generate variance preservation analysis of the PCA.
	'''
	total_var = sum(D ** 2)
	D /= total_var
	plt.style.use('ggplot')
	plt.title(title)
	plt.plot(range(len(D)), D)
	plt.xlabel("Order of eigenvalue")
	plt.ylabel("Percentage of variance")
	plt.savefig(savename, format = 'png')
	plt.close()

def alg_batch(X_train, y_train, X_test, y_test):
	logreg(X_train, y_train, X_test, y_test)
	linearSVM(X_train, y_train, X_test, y_test)
	kernelSVM(X_train, y_train, X_test, y_test)
	MLP(X_train, y_train, X_test, y_test)
# main driver function
if __name__ == '__main__':
	print('==> Running Algorithms on multiclass data...')
	# alg_batch(X_train, y_train, X_test, y_test)
	print('============================================')
	print('==> Running PCA multiclass data...')
	X_train_rot, D_train, k_train = PCA(X_train)
	# PCA_analysis(D_train, title = 'PCA Analysis for Training Data', \
		# savename = 'PCA_train.png')
	# X_test_rot, D_test, k_test = PCA(X_test, k = k_train)
	# alg_batch(X_train_rot, y_train, X_test_rot, y_test)
