import numpy as np
from sklearn import linear_model, svm, neural_network
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

def plot_confusion_matrix(cmat, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          savename = 'confusion.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    print('==> Plotting confusion matrix...')
    plt.imshow(cmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cmat)

    thresh = cmat.max() / 2.
    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        plt.text(j, i, cmat[i, j],
                 horizontalalignment="center",
                 color="white" if cmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(savename, format = 'png')
    plt.close()

def training(model, modelName, X_train, y_train, X_test, y_test, \
	bin_clf = False):
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
	print('==> Testing the {} model...'.format(modelName))
	print('-- Training accuracy: {a:4.4f}'.format(a = model.score(X_train, y_train)))
	print('-- Testing accuracy: {a:4.4f}'.format(a = model.score(X_test, y_test)))
	if bin_clf:
		y_pred = model.predict(X_test)
		precision, recall, f1 = bin_clf_analysis(y_test, y_pred)
		print('-- Testing precision: {a:4.4f}'.format(a = precision))
		print('-- Testing recall: {a:4.4f}'.format(a = recall))
		print('-- Testing F1-score: {a:4.4f}'.format(a = f1))
	print('================')

def bin_clf_analysis(y, y_pred):
	'''
		Analyze result of binary classification.
		Return precision, recall, F1-score.
	'''
	m = len(y)
	count_p, count_r = 0, 0
	total_p, total_r = 0, 0
	for index in range(m):
		actual, pred = y.item(index), y_pred.item(index)
		if actual == 1:
			total_r += 1
			if pred == 1:
				count_r += 1
		if pred == 1:
			total_p += 1
			if actual == 1:
				count_p += 1
	precision = 1. * count_p / total_p
	recall = 1. * count_r / total_r
	f1 = 2. * precision * recall / (precision + recall)
	return precision, recall, f1

def logreg(X_train, y_train, X_test, y_test, reg = 0.2, lc = False, \
	bin_clf = False, cm = True):
	''' 
		Produce logistic regression accuracy based on the training set and
		the test set
	'''
	print('********************************')
	print('==> Setting up model for logistic regression...')
	model = linear_model.LogisticRegression(C= 1.0 / reg, verbose = False)
	training(model, 'logistic regression', X_train, y_train, X_test, y_test, \
		bin_clf = bin_clf)
	if lc:
		title = 'Learning Curve (Logistic Regression, $\lambda = {}$)'.format(reg)
		save_file_name = 'logreg'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for Logistic Regression'
		save_file_name = 'logreg_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)
	print('********************************')

def linearSVM(X_train, y_train, X_test, y_test, reg = 0.2, lc = False, \
	bin_clf = False, cm = True):
	''' 
		Produce the linear support vector machine report based on the training set and 
		the test set
	'''
	print('********************************')
	print('==> Setting up model for linear support vector machine...')
	model = svm.LinearSVC(C = 1.0 / reg, verbose = 0)
	training(model, 'linear SVM', X_train, y_train, X_test, y_test, \
		bin_clf = bin_clf)
	if lc:
		title = 'Learning Curve (Linear SVM, $\lambda = {}$)'.format(reg)
		save_file_name = 'linsvm'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for Linear SVM'
		save_file_name = 'linsvm_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)
	print('********************************')

def kernelSVM(X_train, y_train, X_test, y_test, reg = 0.2, lc = False, \
	bin_clf = False, cm = True):
	''' 
		Produce the rbf-kernel support vector machine report based on the training set and 
		the test set
	'''
	print('********************************')
	print(' ==> Setting up model for rbf-kernel support vector machine...')
	model = svm.SVC(C = 1.0 / reg, verbose = 0)
	training(model, 'rbf-kernel SVM', X_train, y_train, X_test, y_test, \
		bin_clf = bin_clf)
	title = 'Learning Curve (rbf-kernel SVM, $\lambda = {}$)'.format(reg)
	if lc:
		save_file_name = 'rbfsvm'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for Kernel SVM'
		save_file_name = 'kersvm_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)
	print('********************************')

def MLP(X_train, y_train, X_test, y_test, reg = 0.01, lc = False, \
	bin_clf = False, cm = True):
	''' 
		Produce the multilayer perceptron training report based on the training set and 
		the test set
	'''
	print('********************************')
	print('==> Setting up MLP model')
	model = neural_network.MLPClassifier(alpha = reg, \
		hidden_layer_sizes = (100, 100, 100,))
	training(model, 'MLP', X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	if lc:
		title = 'Learning Curve (MLP, $\lambda = {}$, hidden = [100, 100, 100])'.format(reg)
		save_file_name = 'MLP'
		learning_curve_wrapper(model, save_file_name, title, X_train, y_train)
	if cm:
		y_pred = model.predict(X_test)
		num_classes = int(y_train.max()) + 1
		classes = ['class {}'.format(i) for i in range(num_classes)]
		cmat = confusion_matrix(y_test, y_pred)
		title = 'Confusion Matrix for MLP'
		save_file_name = 'MLP_cm.png'
		plot_confusion_matrix(cmat, classes, title = title, \
			savename = save_file_name)
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
	var = np.multiply(X, X).mean(axis = 0)
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

def alg_batch(X_train, y_train, X_test, y_test, bin_clf = False):
	logreg(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	linearSVM(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	kernelSVM(X_train, y_train, X_test, y_test, bin_clf = bin_clf)
	MLP(X_train, y_train, X_test, y_test, bin_clf = bin_clf)

# main driver function
if __name__ == '__main__':
	# multiclass original
	print('==> Running algorithms on multiclass data with full dimension...')
	alg_batch(X_train, y_train, X_test, y_test)
	print('============================================')
	# # multiclass PCA
	# print('==> Running PCA multiclass data...')
	# X_train_rot, D_train, k_train = PCA(X_train)
	# PCA_analysis(D_train, title = 'PCA Analysis for Training Data', \
	# 	savename = 'PCA_train.png')
	# X_test_rot, D_test, k_test = PCA(X_test, k = k_train)
	# alg_batch(X_train_rot, y_train, X_test_rot, y_test)
	# binary original
	# print('==> Running algorithms on binary classification with full dimension...')
	# y_train_bin = (y_train != 11).astype(int)
	# y_test_bin = (y_test != 11).astype(int)
	# alg_batch(X_train, y_train_bin, X_test, y_test_bin, bin_clf = True)
