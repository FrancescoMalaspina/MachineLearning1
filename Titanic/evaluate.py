import numpy as np
import matplotlib.pyplot as plt

data = np.load("model.npz")
w = data["arr_0"]
b = data["arr_1"]
print("Weights = ", w,"\nBias = ", b)

def load_file(filename):
	data = np.loadtxt(filename)
	X = data [:, :-1]
	Y = data [:,  -1]
	return X, Y

def logreg_inference(X, w, b):
	logits = X @ w + b
	probabilities = 1 / (1 + np.exp(-logits))
	return probabilities

X_test, Y_test  = load_file("titanic-test.txt")
# X, 		Y 		= load_file("titanic-train.txt")

# P_train = logreg_inference(X, w, b)
# train_predictions = (P_train > 0.5)
# train_accuracy = (train_predictions == Y).mean()
# print ("Train Accuracy = ", train_accuracy * 100)

P_test = logreg_inference(X_test, w, b)
test_predictions = (P_test > 0.5)
test_accuracy = (test_predictions == Y_test).mean()
print ("Test Accuracy = ", test_accuracy * 100)

my = np.array([2, 0, 21, 1, 0, 35])
P_my = logreg_inference(my, w, b)
print("My probability of Surviving was: ", P_my*100)