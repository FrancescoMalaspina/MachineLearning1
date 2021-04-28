import numpy as np
import os
import porter

PUNCT = '!#$%&()?=^.*:,;@[]?}{|/§><1234567890_§+-"'
TABLE = str.maketrans(PUNCT, " " * len(PUNCT))
#SETS = ("train", "test", "validation")
SETS = ("train", "test")


def load_vocabulary(filename):
	f = open (filename, encoding = "utf8")
	text = f.read()
	f.close()
	words = text.split()
	voc = {}
	index = 0
	for word in words:
		voc[word] = index
		index +=1
	return voc

def read_document (filename, voc):
	f = open(filename, encoding = "utf8")
	text = f.read() #read the file as a string
	f.close()
	text = text.lower()
	text = text.translate(TABLE)
	words = text.split() #create a list with the words
	stem = list(map(porter.stem, words))
	bow = np.zeros(len(voc)) #bag of word
	# for word in stem:
	for word in words:
		if word in voc:
			index = voc[word]
			bow[index] += 1
	return bow

vocabulary = load_vocabulary("vocabulary.txt")


for set_ in SETS:
	documents = []
	labels = []
	filenames = [] # names of the files containing the reviews
	pos = set_ + "/pos/"
	neg = set_ + "/neg/"
	savefile = set_ + ".txt.gz"
	for f in os.listdir(pos):
		path = pos + f
		bow = read_document(path, vocabulary)
		documents.append(bow)
		labels.append(1)
		# filenames.append(int(f.replace('.txt', '')))
		filenames.append(f)
	for f in os.listdir(neg):
		path = neg + f
		bow = read_document(path, vocabulary)
		documents.append(bow)
		labels.append(0)
		# filenames.append(int(f.replace('.txt', '')))
		filenames.append(f)
	print("Una parte fatta")

	X = np.stack(documents) #righe: documenti, colonne: parole del vocabolario
	Y = np.array(labels)
	N = np.array(filenames)
	# data = np.concatenate([X,Y[:, None], N[:, None]], 1)
	data = np.concatenate([X,Y[:, None]], 1)
	np.savetxt(savefile, data)
	np.savetxt(set_ + "_names.txt.gz", N, fmt="%s")
