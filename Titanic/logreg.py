import numpy as np
import matplotlib.pyplot as plt

def load_file(filename):
	data = np.loadtxt(filename)
	X = data [:, :-1]
	Y = data [:,  -1]
	return X, Y

def logreg_inference(X, w, b):
	logits = X @ w + b
	probabilities = 1 / (1 + np.exp(-logits))
	return probabilities

def cross_entropy(P, Y):
	return (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()

def logreg_train (X, Y, lambda_=0, lr=1e-3, steps= 1000):
	''' X has n rows -> number of training samples
	and m cols -> number of features'''
	m, n = X.shape
	b = 0
	w = np.zeros(n)
	accs = []
	losses = []
	lr = 0.001 #learning rate
	steps = 100000

	for step in range (steps):
		P = logreg_inference(X, w, b) 	#probabilities estimates
		loss = cross_entropy(P, Y)
		if step % 1000 == 0:
			prediction = (P > 0.5)
			accuracy = (prediction == Y).mean()
			accs.append (accuracy)
			losses.append(loss)
			print (step, loss, accuracy * 100)
		grad_b = (P - Y).mean() 		#derivative with respect to b
		grad_w = (X.T @ (P - Y)) / m + 2 * lambda_ * w
		#only now update the parameters
		b -= lr * grad_b
		w -= lr * grad_w
	return w, b, accs, losses

X, 		Y 		= load_file("titanic-train.txt")
Xrnd = X + np.random.randn(X.shape[0], X.shape[1]) / 15 
# plt.scatter (Xrnd[:, 0], Xrnd[:, 1], c=Y)
# plt.xlabel("Class of the passengers")
# plt.ylabel("Sex of the passengers")
# plt.title("Scatter plot of outcomes")
# plt.show()

w, b, accs, losses = logreg_train(X, Y)
# plt.plot(accs, label = 'Accuracy')
# plt.plot(losses, label = 'Losses')
# plt.legend()
# plt.xlabel("Iterations e+03")
# plt.title("Training curve")
# plt.show()
print("Weights = ", w,"\nBias = ", b)

np.savez("model.npz", w, b)

P_train = logreg_inference(X, w, b)
train_predictions = (P_train > 0.5)
train_accuracy = (train_predictions == Y).mean()
print ("Train Accuracy = ", train_accuracy * 100)


# my = np.array([2, 1, 21, 1, 0, 35])
# P_my = logreg_inference(my, w, b)
# print("My probability of Surviving was: ", P_my*100)