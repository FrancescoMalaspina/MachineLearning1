import os
import collections
import porter

VOC_LEN = 100

PUNCT = '!#$%&()?=^.*:,;@[]?}{|/§><1234567890_§+-"'
TABLE = str.maketrans(PUNCT, " " * len(PUNCT))

def read_document(filename):
	f = open(filename, encoding = "utf8")
	text = f.read() #read the file as a string
	f.close()
	text = text.lower()
	text = text.translate(TABLE)
	words = text.split() #create a list with the words
	# stem = list(map(porter.stem, words))
	# return stem
	return words

vocabulary = collections.Counter()
stopwords = read_document("stopwords.txt")
for f in os.listdir("train/pos"):
	path = "train/pos/" + f
	words = read_document(path)
	vocabulary.update(words)
	#print (vocabulary)

for f in os.listdir("train/neg"):
	path = "train/neg/" + f
	words = read_document(path)
	vocabulary.update(words)
	#print (vocabulary)

# for word, count in vocabulary.most_common(VOC_LEN):
# 	print(word)

# for word in stopwords:
# 	del vocabulary[word]

f = open ("vocabulary.txt", "w")
for word, count in vocabulary.most_common(VOC_LEN):
	print(word, file = f)

f.close()

