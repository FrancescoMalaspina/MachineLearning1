import numpy as np

NumberOfFiles = 10

def train_nb(X, Y):
	n = X.shape [1]
	m = X.shape [0]
	pos_counter = X[Y == 1, :].sum(0)
	pi_pos = (1 + pos_counter) / (pos_counter.sum() + n)
	neg_counter = X[Y == 0, :].sum(0)
	pi_neg = (1 + neg_counter) / (neg_counter.sum() + n)
	prior_pos = Y.sum() / m
	prior_neg = 1 - prior_pos
	w = np.log(pi_pos) - np.log(pi_neg)
	b = np.log(prior_pos) - np.log(prior_neg)
	return w, b

def inference_nb(X, w, b):
	scores = X @ w +b
	labels = (scores > 0).astype(int)
	return labels, scores

def wrongly_classified_reviews(predictions, Y, scores, filenames):
	""" This function prints the names of the files that get misclassified with the worst scores"""
	mask = (predictions != Y)
	scores_masked = scores[mask]
	filenames_masked = filenames[mask]
	print("")
	indexes_ordered = np.argsort( scores_masked )
	for i in indexes_ordered[ : -NumberOfFiles : -1]:
		print ("negative review classified as good, score: %3.1f, filename: %s" % (scores_masked[i], filenames_masked[i]))
	print("")
	for i in indexes_ordered[ : NumberOfFiles ]:
		print ("positive review classified as bad, score: %3.1f, filename: %s" % (scores_masked[i], filenames_masked[i]))

train_data = np.loadtxt("train.txt.gz")
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]
train_filenames = np.loadtxt("train_names.txt.gz", dtype = str)
w, b = train_nb(Xtrain, Ytrain)
#print(w, b)

test_data = np.loadtxt("test.txt.gz")
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]
test_filenames = np.loadtxt("test_names.txt.gz", dtype = str)


#----------------------------------------------
# Accuracy testing
train_predictions, train_scores = inference_nb(Xtrain, w, b)
accuracy = (train_predictions == Ytrain).mean() * 100
print ("Training accuracy = ", accuracy)

test_predictions, test_scores = inference_nb(Xtest, w, b)
accuracy = (test_predictions == Ytest).mean() * 100
print ("Test accuracy = ", accuracy)


#-----------------------------------------------
#analysis of the inner workings of the algorithm
f = open("vocabulary.txt")
vocabulary = f.read().split()
f.close()

print("\nTRAIN DATA")
wrongly_classified_reviews(train_predictions, Ytrain, train_scores, train_filenames)

print("\nTEST DATA")
wrongly_classified_reviews(test_predictions, Ytest, test_scores, test_filenames)

# indices = np.argsort(w)
# print("NEGATIVE WORDS")
# for i in indices[:30]:
# 	print ( vocabulary[i], w[i])
# print("\n")

# print("POSITIVE WORDS")
# for i in indices[:-30:-1]:
# 	print ( vocabulary[i], w[i])

# print(scores_wrong.shape, "\n \n")

# print(test_scores_wrong.shape, "\n \n")
