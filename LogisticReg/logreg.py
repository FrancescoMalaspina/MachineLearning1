import numpy as np
import matplotlib.pyplot as plt

def logreg_inference(X, w, b):
	logits = X @ w + b
	probabilities = 1 / (1 + np.exp(-logits))
	return probabilities

def cross_entropy(P, Y):
	return (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()

def logreg_train (X,Y):
	''' X has n rows -> number of training samples
	and m cols -> number of features'''
	m, n = X.shape
	b = 0
	w = np.zeros(n)
	lr = 0.005 #learning rate
	steps = 100000

	for i in range (steps):
		P = logreg_inference(X, w, b) 	#probabilities estimates
		loss = cross_entropy(P, Y)
		if i % 1000 == 0:
			print (i, loss)
		grad_b = (P - Y).mean() 		#derivative with respect to b
		grad_w = (X.T @ (P - Y)) / m
		#only now update the parameters
		b -= lr * grad_b
		w -= lr * grad_w
	return w, b

data = np.loadtxt("exam.txt")
X = data[:, :2]
Y = data[:,  2]
# plt.scatter (X[:, 0], X[:, 1], c=Y)
# plt.show()

w, b = logreg_train(X, Y)
print("w = ", w,"b = ", b)

P = logreg_inference(X, w, b)
predictions = (P > 0.5)
accuracy = (predictions == Y).mean()
print ("Accuracy = ", accuracy * 100)




