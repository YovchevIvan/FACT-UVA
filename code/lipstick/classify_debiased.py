import pickle
import operator
from random import shuffle
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle as shuffle_parallel
from matplotlib import pyplot as plt
from random import seed
import numpy as np
import argparse
import gensim
seed(10)

def load_limited_vocab(embeddings_file):
	# load embeddings file limited in size created by the debiasing script (see paper)
	# vocab: lists of vocabularies
	# wv: word vectors
	# w2i: dictionaries of words and their corresponding index

	#load model
	model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True)

	# model has been loaded
	assert (model is not None)

	vocab = [w for w in model.vocab]

	print("Number of words: ", len(vocab))

	wv = np.array([model[w] for w in vocab], dtype='float32')
	w2i = {w: i for i, w in enumerate(vocab)}
	return vocab, wv, w2i

def compute_bias_by_projection(space_to_tag, vocab,	 wv, w2i):
	# create a dictionary of the bias, before and after
	males = wv[space_to_tag].dot(wv[space_to_tag][w2i[space_to_tag]['he'],:])
	females = wv[space_to_tag].dot(wv[space_to_tag][w2i[space_to_tag]['she'],:])
	d = {}
	for w,m,f in zip(vocab[space_to_tag], males, females):
		d[w] = m-f
	return d

def most_biased(gender_bias_bef, fname):
	# extract nost biased words (in case it is not stored in the goven folder)

	size_train = 500
	size_test = 2000
	size = size_train + size_test
	sorted_g = sorted(gender_bias_bef.items(), key=operator.itemgetter(1))
	females = [item[0] for item in sorted_g[:size]]
	males = [item[0] for item in sorted_g[-size:]]
	for f in females:
		assert(gender_bias_bef[f] < 0)
	for m in males:
		assert(gender_bias_bef[m] > 0)
	shuffle(females)
	shuffle(males)
	with open(fname + '_females.data', 'wb') as filehandle:
		# store the data as binary data stream
		pickle.dump(females, filehandle)
	with open(fname + '_males.data', 'wb') as filehandle:
		# store the data as binary data stream
		pickle.dump(males, filehandle)

def report_bias(gender_bias):
	# calculate the avg bias of the vocabulary (abs) before and after debiasing (sanity check)

	bias = 0.0
	for k in gender_bias:
		bias += np.abs(gender_bias[k])
	print (bias/len(gender_bias))

def train_and_predict(space_train, space_test, clf, portion, wv, w2i, fname):
	# take 5000 most biased words, split each polarity randomly to train (1/5) and test (4/5), and predict

	## loading male and female most biased words

	size_train = 500
	size_test = 2000
	size = size_train + size_test
	with open(fname + '_females.data', 'rb') as filehandle:
		# store the data as binary data stream
		females = pickle.load(filehandle)
	with open(fname + '_males.data', 'rb') as filehandle:
		# store the data as binary data stream
		males = pickle.load(filehandle)

	X_train = [wv[space_train][w2i[space_train][w],:] for w in males[:size_train]+females[:size_train]]
	Y_train = [1]*size_train + [0]*size_train
	X_test = [wv[space_test][w2i[space_test][w],:] for w in males[size_train:]+females[size_train:]]
	Y_test = [1]*size_test + [0]*size_test
	
	X_train, Y_train = shuffle_parallel(X_train, Y_train)
	X_test, Y_test = shuffle_parallel(X_test, Y_test)
	# get portion of data
	
	train_split_point = int(len(X_train)*portion)
	X_train = X_train[:train_split_point]
	Y_train = Y_train[:train_split_point]
	
	
	clf.fit(X_train, Y_train)

	print ('\ttrain with', space_train)
	print ('\ttest with', space_test)

	preds = clf.predict(X_test)

	accuracy = [1 if y==z else 0 for y,z in zip(preds, Y_test)]
	acc =  float(sum(accuracy))/len(accuracy)
	print ('\taccuracy:',acc)
	return acc

def run_all_classifiers(wv, w2i, fname):
	# define classifier
	# RBF SvM
	clf_svm_rbf = svm.SVC(25)
	# Linear SVM
	clf_svm_linear = svm.SVC(kernel = 'linear')
	# Random Forest
	clf_forest = RandomForestClassifier()
	# Gradient boosting
	clf_boost = GradientBoostingClassifier()
	# Logistic regression
	clf_logreg = LogisticRegression()
	# Linear MLP
	clf_mlp_linear = MLPClassifier(hidden_layer_sizes=(50), activation='identity')
	# MLP
	clf_mlp = MLPClassifier(hidden_layer_sizes=(50), activation='tanh' )
	# Deep net
	clf_deep = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10), activation='relu')



	classifiers = [clf_svm_rbf, clf_logreg, clf_mlp]
	classifier_names = ["SVM - radial basis", "Logistic regression","MLP"]
	seeds = [1,2,3,4,5,6,7,8,9,10]

	splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	accuracies_bef = []
	accuracies_aft = []
	acc_diffs = []

	for clf_ind in range(len(classifiers)):
		clf = classifiers[clf_ind]
		print("================" + classifier_names[clf_ind] + "================")
		clf_acc_bef = []
		clf_acc_aft = []
		acc_diff = []
		for split in splits:
			print("\nData used (portion): " + str(split) + "\n")
			# classification before debiasing
			acc_bef = 0
			acc_aft = 0
			for se in seeds:
				seed(se)
				acc_bef += train_and_predict('bef', 'bef', clf, split, wv, w2i, fname)

				# classification after debiasing

				acc_aft += train_and_predict('aft', 'aft', clf, split, wv, w2i, fname)

			acc_bef/= len(seeds)
			acc_aft/= len(seeds)
			clf_acc_bef.append(acc_bef)
			clf_acc_aft.append(acc_aft)
			acc_diff.append(acc_bef-acc_aft)
		accuracies_bef.append(clf_acc_bef)
		accuracies_aft.append(clf_acc_aft)
		acc_diffs.append(acc_diff)
	
	
	#### Plotting

	fig,a =	 plt.subplots(1, 3)
	plt.suptitle("Classification results - " + config.embedding)
	a[0].set_title("Classification on original embeddings")
	a[0].set_xlabel('training data used (portion)')
	a[0].set_ylabel('accuracy')
	a[0].set_ylim(0.7, 1.05)
	a[1].set_title("Classification on debiased embeddings")
	a[1].set_xlabel('training data used (portion)')
	a[1].set_ylabel('accuracy')
	a[1].set_ylim(0.7, 1.05)
	a[2].set_title("Difference: original - debiased")
	a[2].set_xlabel('training data used (portion)')
	a[2].set_ylabel('accuracy difference')
	for accuracy in accuracies_bef:
		a[0].plot(splits, accuracy)
	for accuracy in accuracies_aft:
		a[1].plot(splits, accuracy)	 
	for accuracy in acc_diffs:
		a[2].plot(splits, accuracy)
	plt.legend(classifier_names)
	plt.show()

############################# MAIN #####################################
if __name__ == "__main__":

	# Param parser
	parser = argparse.ArgumentParser()

	# Model params
	parser.add_argument('--embeddings_original', type=str, default="../../embeddings/bias_word2vec.bin", help="Path to original embeddings file")
	parser.add_argument('--embeddings_debiased', type=str, default="../../embeddings/debiased_word2vec.bin", help="Path to debiased embeddings file")
	parser.add_argument('--fname', type=str, default="results", help="Prefix name of output file")
	config = parser.parse_args()

	emb_path_bef = config.embeddings_original
	emb_path_aft = config.embeddings_debiased

	#loading limited embeddings
	vocab = {} 
	wv = {}
	w2i = {}
	vocab['bef'], wv['bef'], w2i['bef'] = load_limited_vocab(emb_path_bef)
	vocab['aft'], wv['aft'], w2i['aft'] = load_limited_vocab(emb_path_aft)
	print("loading done")

	# assigning biases
	gender_bias_bef =  compute_bias_by_projection('bef', vocab,wv,w2i)
	gender_bias_aft =  compute_bias_by_projection('aft', vocab,wv,w2i)

	# creating list of most biased word (if not present)
	most_biased(gender_bias_bef, config.fname)

	#check average bias before and after
	report_bias(gender_bias_bef)
	report_bias(gender_bias_aft)

	# run classification with several classifiers before and after debiasing
	# classifiaction aims to tell which words are originally masculine and feminine
	# if the debiasing works as intended, after debiasing the model should not be able to learn separation
	run_all_classifiers(wv, w2i, config.fname)
	
