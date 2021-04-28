import numpy as np
import pvml

train_data = np.loadtxt("train.txt.gz")
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]

print("train data loaded")

w, b = pvml.svm_train(Xtrain, Ytrain, 1e-5, 1e-3, 10000)

print("model trained")

test_data = np.loadtxt("test.txt.gz")
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]

print("test data loaded")


#----------------------------------------------
# Accuracy testing
train_predictions = pvml.svm_inference(Xtrain, w, b)
accuracy = (train_predictions == Ytrain).mean() * 100
print ("Training accuracy = ", accuracy)

test_predictions = pvml.svm_inference(Xtest, w, b)
accuracy = (test_predictions == Ytest).mean() * 100
print ("Test accuracy = ", accuracy)


# def svm_train(X, Y, lambda_=1e-5, lr=1e-3, steps=1000): 
# 	m, n = X.shape
# 	w = np.zeros(n)
# 	b = 0
# 	for step in range(steps):
# 		z = X @ w + b
# 		hinge_diff = -(Y == 1) * (z < 1) + (Y == 0) * (z > -1) 
# 		grad_w = (hinge_diff @ X) / m + lambda_ * w
# 		grad_b = hinge_diff.mean()
# 		w -= lr * grad_w
# 		b -= lr * grad_b 
# 	return w, b

# def svm_inference(X, w, b): 
# 	z = X @ w + b
# 	labels = (z > 0).astype(int) 
# 	return labels