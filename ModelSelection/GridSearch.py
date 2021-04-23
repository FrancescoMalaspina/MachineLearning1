import numpy as np
import pvml

def load_set(filename):
	data = np.loadtxt(filename)
	X = data[:, :-1]
	Y = data[:, -1]
	return X, Y

X, Y 		= load_set("page-blocks-train.txt")
Xval, Yval 	= load_set("page-blocks-val.txt")
Xtest, Ytest = load_set("page-blocks-test.txt")

lambdas = 	[1e-7, 3e-7, 1e-6, 3e-6]
gammas  = 	[1e-6, 3e-6, 1e-5, 3e-5]

best_accuracy = 0
best_alpha = None
best_b = None
best_lambda = None
best_gamma = None

for lambda_ in lambdas:
	for gamma in gammas:
		alpha, b = pvml.ksvm_train (X, Y, "rbf", gamma, lambda_, lr = 0.01, steps = 1000)
		labels, logits = pvml.ksvm_inference(Xval, X, alpha, b, "rbf", gamma)
		accuracy = (labels == Yval).mean() * 100
		print ("lambda = ", lambda_, ", gamma = ", gamma, ", accuracy = ", accuracy)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_alpha = alpha
			best_b = b
			best_lambda = lambda_
			best_gamma = gamma

print("\n TEST")
labels, logits = pvml.ksvm_inference(Xval, X, best_alpha, best_b, "rbf", best_gamma)
accuracy = (labels == Ytest).mean() * 100
print ("lambda = ", best_lambda, ", gamma = ", best_gamma, ", accuracy = ", accuracy)

