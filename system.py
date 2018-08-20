from __future__ import print_function
import numpy as np
from numpy import zeros, newaxis

from keras import regularizers
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input
from keras.layers import LSTM, SimpleRNN, GRU, RepeatVector, Permute, merge, Flatten, Lambda, Concatenate
from keras.layers import Bidirectional
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras import backend as K
import sys
import tensorflow as tf
import h5py

#------------------------------------------------------------------------------------------------------------
if len(sys.argv) < 5 :
	print('[usage] python system.py [usage of data] [embedding] [class weights] [lexicons features]')
	print('usage of data : train-18, train-all, train')
	print('embedding : glove-t, glove-g, acl2015, word2vec, self')
	print('class weights : True / False')
	print('lexicons features : True / False')
	sys.exit(1)

batch_size = 32
UsageOfData = sys.argv[1]
Embedding = sys.argv[2]
ClassWeights = sys.argv[3]
LexiconsFeatures = sys.argv[4]

#check
if not (UsageOfData == 'train-18' or UsageOfData == 'train-all' or UsageOfData == 'train') :
	print('The "usage of data" is wrong!!!')
	sys.exit(1)
if not (Embedding == 'glove-t' or Embedding == 'glove-g' or Embedding == 'acl2015' or Embedding == 'word2vec' or Embedding == 'self') :
	print('The "embedding" is wrong!!!')
	sys.exit(1)
if not (ClassWeights == 'True' or ClassWeights == 'False') :
	print('The "class weights" is wrong!!!')
	sys.exit(1)
if not (LexiconsFeatures == 'True' or LexiconsFeatures == 'False') :
	print('The "lexicons features" is wrong!!!')
	sys.exit(1)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading word list...')
word_list = {}
if UsageOfData == 'train-18' :
	f = open('./Data/wordList/wordList-2018.txt', 'r')
else :
	f = open('./Data/wordList/wordList-2017-2018.txt', 'r')
for line in f.readlines():
    values = line.split()
    coefs = values[0]
    word = values[1]
    word_list[word] = coefs
f.close()
print('word list :', len(word_list))
#------------------------------------------------------------------------------------------------------------
if LexiconsFeatures == 'True' :
	print('--------------------------------------------------')
	print('Load lexicons...')
	lexicon = [{},{},{},{}]
	fileName = ['normalize_afinn_score.txt', 'normalize_Sentiment140_score.txt', 'normalize_sentistrength_score.txt', 'normalize_vader_score.txt']
	for i in range(len(fileName)) :
		LexiconFile = open('./Lexicons/' + fileName[i], 'r')
		for line in LexiconFile.readlines() :
		    token = line.split('\t')
		    if lexicon[i].get(token[0]) is None : 
		        lexicon[i][token[0]] = float(token[1].split('\n')[0])
		LexiconFile.close()
	print('AFINN lexicon :', len(lexicon[0]))
	print('Sentiment140 lexicon :', len(lexicon[1]))
	print('Sentistrength lexicon :', len(lexicon[2]))
	print('Vader lexicon :', len(lexicon[3]))
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading Data...')

def LoadData(name, LexiconsFeatures) :
	#Load data
	data=[]
	score=[]
	f = open('./Data/processed/'+name, 'r')
	for line in f.readlines():
		temp=[]
		tempScore=[]
		sp=line.split()
		for word in sp:
			if word in word_list :
				temp.append(int(word_list[word]))
			if LexiconsFeatures == 'True' :
				s=[]
				if word in lexicon[0] :
					s.append(float(lexicon[0][word]))
				else :
					s.append(float(0.0))
				if word in lexicon[1] :
					s.append(float(lexicon[1][word]))
				else :
					s.append(float(0.0))
				if word in lexicon[2] :
					s.append(float(lexicon[2][word]))
				else :
					s.append(float(0.0))
				if word in lexicon[3] :
					s.append(float(lexicon[3][word]))
				else :
					s.append(float(0.0))
				tempScore.append(s)
		data.append(temp)
		if LexiconsFeatures == 'True' :
			score.append(tempScore)
	f.close()
	X = np.asarray(data)
	if LexiconsFeatures == 'True' :
		Score = np.asarray(score)
		return X, Score
	return X
if LexiconsFeatures == 'True' :
	XTrain18, ScoreTrain18 = LoadData('2018-Valence-oc-En-train-data.tok', LexiconsFeatures)
	if UsageOfData != 'train-18' :
		XTrain17, ScoreTrain17 = LoadData('2017-semEval-en-train-data.tok', LexiconsFeatures)
	XDev18, ScoreDev18 = LoadData('2018-Valence-oc-En-dev-data.tok', LexiconsFeatures)
	XTest18, ScoreTest18 = LoadData('2018-Valence-oc-En-test-data.tok', LexiconsFeatures)
else :
	ScoreTrain18 = ScoreTrain17 = ScoreDev18 = ScoreTest18 = 0
	XTrain18 = LoadData('2018-Valence-oc-En-train-data.tok', LexiconsFeatures)
	if UsageOfData != 'train-18' :
		ScoreTrain = 0
		XTrain17 = LoadData('2017-semEval-en-train-data.tok', LexiconsFeatures)
	XDev18 = LoadData('2018-Valence-oc-En-dev-data.tok', LexiconsFeatures)
	XTest18 = LoadData('2018-Valence-oc-En-test-data.tok', LexiconsFeatures)

print('Padding sequences...')
if UsageOfData != 'train-18' :
	XTrain = np.concatenate((XTrain18, XTrain17), axis=0)		
	maxlen = 99
	XTrain18 = sequence.pad_sequences(XTrain18, maxlen=maxlen)
	XTrain = sequence.pad_sequences(XTrain, maxlen=maxlen)
	XDev18 = sequence.pad_sequences(XDev18, maxlen=maxlen)
	XTest18 = sequence.pad_sequences(XTest18, maxlen=maxlen)
	if LexiconsFeatures == 'True' :
		ScoreTrain18 = sequence.pad_sequences(ScoreTrain18, maxlen=maxlen)
		ScoreTrain = np.concatenate((ScoreTrain18, ScoreTrain17), axis=0)
		ScoreTrain = sequence.pad_sequences(ScoreTrain, maxlen=maxlen)
		ScoreDev18 = sequence.pad_sequences(ScoreDev18, maxlen=maxlen)
		ScoreTest18 = sequence.pad_sequences(ScoreTest18, maxlen=maxlen)
else :
	maxlen = 56
	XTrain18 = sequence.pad_sequences(XTrain18, maxlen=maxlen)
	XDev18 = sequence.pad_sequences(XDev18, maxlen=maxlen)
	XTest18 = sequence.pad_sequences(XTest18, maxlen=maxlen)
	if LexiconsFeatures == 'True' :
		ScoreTrain18 = sequence.pad_sequences(ScoreTrain18, maxlen=maxlen)
		ScoreDev18 = sequence.pad_sequences(ScoreDev18, maxlen=maxlen)
		ScoreTest18 = sequence.pad_sequences(ScoreTest18, maxlen=maxlen)

print('train18 data :', XTrain18.shape)
if UsageOfData != 'train-18' :
	print('trainAll data :', XTrain.shape)
print('dev data :', XDev18.shape)
print('test data :', XTest18.shape)
if LexiconsFeatures == 'True' :
	print('Score Train :', ScoreTrain18.shape)
	if UsageOfData != 'train-18' :
		print('Score TrainAll :', ScoreTrain.shape)
	print('Score Dev :', ScoreDev18.shape)
	print('Score Test :', ScoreTest18.shape)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading Label...')

YTrain18 = np.loadtxt('./Data/processed/2018-Valence-oc-En-train-label.txt')
YDev18 = np.loadtxt('./Data/processed/2018-Valence-oc-En-dev-label.txt')
YTest18 = np.loadtxt('./Data/processed/2018-Valence-oc-En-test-label.txt')

if UsageOfData == 'train-18' :
	YTrainThree    = [0 if x < 0 else 2 if x > 0 else 1 for x in YTrain18]
	YTrainThree    = to_categorical(YTrainThree, num_classes=3)
	YTrainNegative = [3 if x > 0 else x+3 for x in YTrain18]
	YTrainNegative = to_categorical(YTrainNegative, num_classes=4)
	YTrainNeutral  = [0 if x == 0 else 1 for x in YTrain18]
	YTrainNeutral  = to_categorical(YTrainNeutral, num_classes=2)
	YTrainPositive = [0 if x < 0 else x for x in YTrain18]
	YTrainPositive = to_categorical(YTrainPositive, num_classes=4)
	YTrainSeven    = [x+3 for x in YTrain18]
	YTrainSeven    = to_categorical(YTrainSeven, num_classes=7)
elif UsageOfData != 'train-18' :
	YTrain17 = np.loadtxt('./Data/processed/2017-semEval-en-train-label.txt')
	#   The labels of SemEval-2017 are [-1,0,1] in SemEval-2018
	#YTrain = np.concatenate((YTrain18, YTrain17), axis=0)
	#   The labels of SemEval-2017 are [-3,0,3] in SemEval-2018
	YTrain17 = [-3 if x < 0 else 3 if x > 0 else 0 for x in YTrain17]
	YTrain = np.concatenate((YTrain18, YTrain17), axis=0)

	YTrainThree = [0 if x < 0 else 2 if x > 0 else 1 for x in YTrain]
	YTrainThree = to_categorical(YTrainThree, num_classes=3)
	if UsageOfData == 'train-all' :
		YTrainNegative = [3 if x > 0 else x+3 for x in YTrain]
		YTrainNegative = to_categorical(YTrainNegative, num_classes=4)
		YTrainNeutral  = [0 if x == 0 else 1 for x in YTrain]
		YTrainNeutral  = to_categorical(YTrainNeutral, num_classes=2)
		YTrainPositive = [0 if x < 0 else x for x in YTrain]
		YTrainPositive = to_categorical(YTrainPositive, num_classes=4)
		YTrainSeven    = [x+3 for x in YTrain]
		YTrainSeven    = to_categorical(YTrainSeven, num_classes=7)
	else :
		YTrainNegative = [3 if x > 0 else x+3 for x in YTrain18]
		YTrainNegative = to_categorical(YTrainNegative, num_classes=4)
		YTrainNeutral  = [0 if x == 0 else 1 for x in YTrain18]
		YTrainNeutral  = to_categorical(YTrainNeutral, num_classes=2)
		YTrainPositive = [0 if x < 0 else x for x in YTrain18]
		YTrainPositive = to_categorical(YTrainPositive, num_classes=4)
		YTrainSeven    = [x+3 for x in YTrain18]
		YTrainSeven    = to_categorical(YTrainSeven, num_classes=7)

YDevThree    = [0 if x < 0 else 2 if x > 0 else 1 for x in YDev18]
YDevThree    = to_categorical(YDevThree, num_classes=3)
YDevNegative = [3 if x > 0 else x+3 for x in YDev18]
YDevNegative = to_categorical(YDevNegative, num_classes=4)
YDevNeutral  = [0 if x == 0 else 1 for x in YDev18]
YDevNeutral  = to_categorical(YDevNeutral, num_classes=2)
YDevPositive = [0 if x < 0 else x for x in YDev18]
YDevPositive = to_categorical(YDevPositive, num_classes=4)
YDevSeven    = [x+3 for x in YDev18]
YDevSeven    = to_categorical(YDevSeven, num_classes=7)

YTestThree    = [0 if x < 0 else 2 if x > 0 else 1 for x in YTest18]
YTestThree    = to_categorical(YTestThree, num_classes=3)
YTestNegative = [3 if x > 0 else x+3 for x in YTest18]
YTestNegative = to_categorical(YTestNegative, num_classes=4)
YTestNeutral  = [0 if x == 0 else 1 for x in YTest18]
YTestNeutral  = to_categorical(YTestNeutral, num_classes=2)
YTestPositive = [0 if x < 0 else x for x in YTest18]
YTestPositive = to_categorical(YTestPositive, num_classes=4)
YTestSeven    = [x+3 for x in YTest18]
YTestSeven    = to_categorical(YTestSeven, num_classes=7)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading word vectors...')

embeddings_index = {}
if Embedding == 'glove-t' :
	f = open('./vector/glove.twitter.27B.200d.2017.2018.txt') #glove twitter 200d
elif Embedding == 'glove-g' :
	f = open('./vector/glove.840B.300d-2017-2018.txt') #glove common crawl 300d
elif Embedding == 'acl2015' :
	f = open('./vector/word2vec-2017-2018.txt') #ACL W-NUT 2015 400d
elif Embedding == 'word2vec' :
	f = open('./vector/GoogleNews-vectors-negative300-2017-2018.txt') #GoogleNews 300
elif Embedding == 'self' :
	f = open('./vector/selfWordVector.txt') #Self train word2vec 400d
for line in f:
    values = line.split()
    if len(values) < 30 :
        continue
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('word vectors :', len(embeddings_index))
print('dimintion of word vectors :', len(embeddings_index.values()[0]))
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('word embedding...')

#get the embedding dimension
EMBEDDING_DIM = len(embeddings_index.values()[0])
embedding_matrix = np.zeros((len(word_list) + 1, EMBEDDING_DIM))
for word, i in word_list.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[int(i)] = embedding_vector
'''
count = -1 #because embedding_matrix[0] is zero vector
for i in range(len(word_list)+1) :
    for j in range(EMBEDDING_DIM) :
        if embedding_matrix[i][j] != 0 :
            break
        if j == EMBEDDING_DIM-1 :
            count += 1
'''
print('embedding_matrix :', embedding_matrix.shape)
#print('Number of zero embedding :', count)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('input to embedding...')

def word_embedding(data, score, dim, LexiconsFeatures):
	a=[]
	for idx1, i in enumerate(data):
		b=[]
		for idx2, j in enumerate(i):
			c = list( np.zeros(dim) )
			c = embedding_matrix[j]
			if LexiconsFeatures == 'True' :
				c = np.concatenate((c, score[idx1][idx2]), axis=0)
			b.append(c)
		a.append(b)
	a=np.asarray(a)
	return a

XTrain18 = word_embedding(XTrain18, ScoreTrain18, EMBEDDING_DIM, LexiconsFeatures)
print('shape of train18 :', XTrain18.shape)
if UsageOfData != 'train-18' :
	XTrain = word_embedding(XTrain, ScoreTrain, EMBEDDING_DIM, LexiconsFeatures)
	print('shape of trainAll :', XTrain.shape)
XDev18 = word_embedding(XDev18, ScoreDev18, EMBEDDING_DIM, LexiconsFeatures)
print('shape of dev :', XDev18.shape)
XTest18 = word_embedding(XTest18, ScoreTest18, EMBEDDING_DIM, LexiconsFeatures)
print('shape of test :', XTest18.shape)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Training...')

def PCC(y_true, y_pred) :
    pred_mean = K.mean(y_pred)
    label_mean = K.mean(y_true)
    covariance = K.sum(np.dot(y_pred-pred_mean, y_true-label_mean))
    standard_deviation_pred = K.sqrt(K.sum(np.power(y_pred-pred_mean, 2)))
    standard_deviation_label = K.sqrt(K.sum(np.power(y_true-label_mean, 2)))
    pearson = covariance / (standard_deviation_pred * standard_deviation_label)
    return pearson

# H-Parameter
adamC = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, clipnorm=5.0)
_input = Input(shape=[XTrain18.shape[1], XTrain18.shape[2]], dtype='float32')

monitors = ['val_loss', 'val_loss', 'val_loss', 'val_loss', 'val_loss']
patiences = [10, 15, 15, 15, 15]
modelName = ['Three', 'Negative', 'Neural', 'Positive', 'Seven']
earlyStopping1 = EarlyStopping(monitor=monitors[0], min_delta=0, patience=patiences[0], verbose=0, mode='auto')
checkpoint1 = ModelCheckpoint('./saveModel/'+modelName[0]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb1 = [earlyStopping1, checkpoint1]
earlyStopping2 = EarlyStopping(monitor=monitors[1], min_delta=0, patience=patiences[1], verbose=0, mode='auto')
checkpoint2 = ModelCheckpoint('./saveModel/'+modelName[1]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb2 = [earlyStopping2, checkpoint2]
earlyStopping3 = EarlyStopping(monitor=monitors[2], min_delta=0, patience=patiences[2], verbose=0, mode='auto')
checkpoint3 = ModelCheckpoint('./saveModel/'+modelName[2]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb3 = [earlyStopping3, checkpoint3]
earlyStopping4 = EarlyStopping(monitor=monitors[3], min_delta=0, patience=patiences[3], verbose=0, mode='auto')
checkpoint4 = ModelCheckpoint('./saveModel/'+modelName[3]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb4 = [earlyStopping4, checkpoint4]
earlyStopping5 = EarlyStopping(monitor=monitors[4], min_delta=0, patience=patiences[4], verbose=0, mode='auto')
checkpoint5 = ModelCheckpoint('./saveModel/'+modelName[4]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb5 = [earlyStopping5, checkpoint5]

if ClassWeights == 'True' :
	kw1 = 14389.0
	three_class_weight = {0:kw1/9037.0, 1:kw1/18527.0, 2:kw1/15603.0}
	kw2 = 175.0
	negative_class_weight = {0:kw2/129.0, 1:kw2/249.0, 2:kw2/78.0, 3:kw2/725.0}
	kw3 = 495.0
	neutral_class_weight = {0:kw3/341.0, 1:kw3/840.0}
	kw4 = 175.0
	positive_class_weight = {0:kw4/797.0, 1:kw4/167.0, 2:kw4/92.0, 3:kw4/125.0}
	kw5 = 140.0
	seven_class_weight = {0:kw5/129.0, 1:kw5/249.0, 2:kw5/78.0, 3:kw5/341.0, 4:kw5/167.0, 5:kw5/92.0, 6:kw5/125.0}
	print('three_class_weight = ', three_class_weight)
	print('negative_class_weight = ', negative_class_weight)
	print('neutral_class_weight = ', neutral_class_weight)
	print('positive_class_weight = ', positive_class_weight)
	print('seven_class_weight = ', seven_class_weight)

#  Three class model : {-1, 0, 1}------------------------------------------------------------------------------------------
ThreeLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='ThreeLstm')(_input)
ThreeLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(ThreeLstm)
ThreeDense = Dense(200, activation='tanh', name='ThreeRepre')(ThreeLstmSum)
ThreeOutput = Dense(3, activation='softmax')(ThreeDense)

model1 = Model(inputs=_input, outputs=ThreeOutput)
model1.summary()
model1.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
if ClassWeights == 'True' :
	if UsageOfData == 'train-18' :
		model1.fit(XTrain18, YTrainThree, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevThree), callbacks=cb1, verbose=1, class_weight=three_class_weight)
	else :
		model1.fit(XTrain, YTrainThree, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevThree), callbacks=cb1, verbose=1, class_weight=three_class_weight)
else :
	if UsageOfData == 'train-18' :
		model1.fit(XTrain18, YTrainThree, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevThree), callbacks=cb1, verbose=1)
	else :
		model1.fit(XTrain, YTrainThree, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevThree), callbacks=cb1, verbose=1)
model1.load_weights('./saveModel/' + modelName[0] + '.hdf5')
w1 = model1.get_layer(name='ThreeLstm').get_weights()

#  Negative class model : {-3, -2, -1, other}----------------------------------------------------------------------------------
NegativeLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='NegativeLstm1')(_input)
NegativeLstm = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='NegativeLstm2')(NegativeLstm)
NegativeLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(NegativeLstm)
NegativeDense = Dense(200, activation='tanh', name='NegativeRepre')(NegativeLstmSum)
NegativeOutput = Dense(4, activation='softmax')(NegativeDense)

model2 = Model(inputs=_input, outputs=NegativeOutput)
model2.summary()
model2.get_layer(name='NegativeLstm1').set_weights(w1)
model2.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
if ClassWeights == 'True' :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model2.fit(XTrain18, YTrainNegative, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNegative), callbacks=cb2, verbose=1, class_weight=negative_class_weight)
	else :
		model2.fit(XTrain, YTrainNegative, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNegative), callbacks=cb2, verbose=1, class_weight=negative_class_weight)
else :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model2.fit(XTrain18, YTrainNegative, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNegative), callbacks=cb2, verbose=1)
	else :
		model2.fit(XTrain, YTrainNegative, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNegative), callbacks=cb2, verbose=1)
model2.load_weights('./saveModel/' + modelName[1] + '.hdf5')
w2 = model2.get_layer(name='NegativeLstm2').get_weights()

#  Neutral class model : {0, other}-----------------------------------------------------------------------------------------
NeuralLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='NeuralLstm1')(_input)
NeuralLstm = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='NeuralLstm2')(NeuralLstm)
NeuralLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(NeuralLstm)
NeuralDense = Dense(100, activation='tanh', name='NeuralRepre')(NeuralLstmSum)
NeuralOutput = Dense(2, activation='softmax')(NeuralDense)

model3 = Model(inputs=_input, outputs=NeuralOutput)
model3.summary()
model3.get_layer(name='NeuralLstm1').set_weights(w1)
model3.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
if ClassWeights == 'True' :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model3.fit(XTrain18, YTrainNeutral, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNeutral), callbacks=cb3, verbose=1, class_weight=neutral_class_weight)
	else :
		model3.fit(XTrain, YTrainNeutral, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNeutral), callbacks=cb3, verbose=1, class_weight=neutral_class_weight)
else :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model3.fit(XTrain18, YTrainNeutral, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNeutral), callbacks=cb3, verbose=1)
	else :
		model3.fit(XTrain, YTrainNeutral, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevNeutral), callbacks=cb3, verbose=1)
model3.load_weights('./saveModel/' + modelName[2] + '.hdf5')
w3 = model3.get_layer(name='NeuralLstm2').get_weights()

#  Positive class model : {1, 2, 3, other}-----------------------------------------------------------------------------------
PositiveLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='PositiveLstm1')(_input)
PositiveLstm = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='PositiveLstm2')(PositiveLstm)
PositiveLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(PositiveLstm)
PositiveDense = Dense(200, activation='tanh', name='PositiveRepre')(PositiveLstmSum)
PositiveOutput = Dense(4, activation='softmax')(PositiveDense)

model4 = Model(inputs=_input, outputs=PositiveOutput)
model4.summary()
model4.get_layer(name='PositiveLstm1').set_weights(w1)
model4.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
if ClassWeights == 'True' :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model4.fit(XTrain18, YTrainPositive, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevPositive), callbacks=cb4, verbose=1, class_weight=positive_class_weight)
	else :
		model4.fit(XTrain, YTrainPositive, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevPositive), callbacks=cb4, verbose=1, class_weight=positive_class_weight)
else :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model4.fit(XTrain18, YTrainPositive, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevPositive), callbacks=cb4, verbose=1)
	else :
		model4.fit(XTrain, YTrainPositive, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevPositive), callbacks=cb4, verbose=1)
model4.load_weights('./saveModel/' + modelName[3] + '.hdf5')
w4 = model4.get_layer(name='PositiveLstm2').get_weights()

#  Seven class model : {-3, -2, -1, 0, 1, 2, 3}---------------------------------------------------------------------------
SevenLstm1 = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='SevenLstm1')(_input)
SevenLstm2 = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='SevenLstm2')(SevenLstm1)
SevenLstm3 = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='SevenLstm3')(SevenLstm1)
SevenLstm4 = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='SevenLstm4')(SevenLstm1)
SevenLstm = merge([SevenLstm2, SevenLstm3, SevenLstm4], mode='concat')
SevenLstm = Bidirectional(LSTM(200, dropout=0.3, return_sequences=True), merge_mode='concat')(SevenLstm)

attention = Dense(200, activation='tanh')(SevenLstm) #200
attention = Dense(1, bias=False)(attention)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(400)(attention) #400
attention = Permute([2, 1])(attention)
representation = merge([SevenLstm, attention], mode='mul')
representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)

SevenDense = Dense(200, activation='tanh', name='SevenRepre')(representation) #200
SevenOutput = Dense(7, activation='softmax')(SevenDense)

model5 = Model(inputs=_input, outputs=SevenOutput)
model5.summary()
model5.get_layer(name='SevenLstm1').set_weights(w1)
model5.get_layer(name='SevenLstm2').set_weights(w2)
model5.get_layer(name='SevenLstm3').set_weights(w3)
model5.get_layer(name='SevenLstm4').set_weights(w4)
model5.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
if ClassWeights == 'True' :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model5.fit(XTrain18, YTrainSeven, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevSeven), callbacks=cb5, verbose=1, class_weight=seven_class_weight)
	else :
		model5.fit(XTrain, YTrainSeven, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevSeven), callbacks=cb5, verbose=1, class_weight=seven_class_weight)
else :
	if UsageOfData == 'train-18' or UsageOfData == 'train' :
		model5.fit(XTrain18, YTrainSeven, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevSeven), callbacks=cb5, verbose=1)
	else :
		model5.fit(XTrain, YTrainSeven, batch_size=batch_size, epochs=200, validation_data=(XDev18, YDevSeven), callbacks=cb5, verbose=1)
#--------------predict--------------
predict = [[], [], [], [], []]
model1.load_weights('./saveModel/' + modelName[0] + '.hdf5')
predict[0] = model1.predict(XTest18, batch_size=batch_size)

model2.load_weights('./saveModel/' + modelName[1] + '.hdf5')
predict[1] = model2.predict(XTest18, batch_size=batch_size)

model3.load_weights('./saveModel/' + modelName[2] + '.hdf5')
predict[2] = model3.predict(XTest18, batch_size=batch_size)

model4.load_weights('./saveModel/' + modelName[3] + '.hdf5')
predict[3] = model4.predict(XTest18, batch_size=batch_size)

model5.load_weights('./saveModel/' + modelName[4] + '.hdf5')
predict[4] = model5.predict(XTest18, batch_size=batch_size)
np.savetxt('./predict/systemPredictProbability_Data_'+ UsageOfData + '_Embedding_' + Embedding + '_ClassWeights_' + ClassWeights + '_LexiconsFeatures_' + LexiconsFeatures + '.txt', predict[4])
#--------------Metric--------------
matrix = [np.zeros((3, 3)), np.zeros((4, 4)), np.zeros((2, 2)), np.zeros((4, 4)), np.zeros((7, 7))]
pred_str = []
pred = []
#Calculate confusion matrix
for l in range(len(predict)) :
    if l == 0 :
        Target = YTestThree
    elif l == 1 :
        Target = YTestNegative
    elif l == 2 :
        Target = YTestNeutral
    elif l == 3 :
        Target = YTestPositive
    elif l == 4 :
        Target = YTestSeven
    for i, (tar, Label) in enumerate( zip(Target, predict[l]) ) :
        m = np.max(Label)
        for j, value in enumerate(Label) :
            if value == m :
                if l == 4 :
                    pred_str.append(str(j - 3))
                    pred.append(int(j - 3))
                for k, num in enumerate(tar) :
                    if num == 1 :
                        matrix[l][k][j] += 1
                        break
                break
#--------------Save Predict--------------
f = open('./predict/systemPredict_Data_'+ UsageOfData + '_Embedding_' + Embedding + '_ClassWeights_' + ClassWeights + '_LexiconsFeatures_' + LexiconsFeatures + '.txt', 'w')
f.write('\n'.join(pred_str))
f.close()
#------------------------------------------------------------------------------------------------------------
average_recall = [np.zeros((3)), np.zeros((4)), np.zeros((2)), np.zeros((4)), np.zeros((7))]
ar = [0, 0, 0, 0, 0]
acc = [0, 0, 0, 0, 0]
for i in range(len(matrix)) :
    if i == 0 :
        average_recall[i][0] = matrix[i][0][0] / (matrix[i][0][0] + matrix[i][0][1] + matrix[i][0][2])
        average_recall[i][1] = matrix[i][1][1] / (matrix[i][1][0] + matrix[i][1][1] + matrix[i][1][2])
        average_recall[i][2] = matrix[i][2][2] / (matrix[i][2][0] + matrix[i][2][1] + matrix[i][2][2])
        ar[i] = (average_recall[i][0]+average_recall[i][1]+average_recall[i][2]) / 3
        acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2]) / len(YTestThree)
        print('--------------------------------'+ modelName[i] +'---------------------------------------')
        print('Average Recall : ', ar[i])
        print('Acc. : ', acc[i])
    elif i == 1 or i == 3 :
        average_recall[i][0] = matrix[i][0][0] / (matrix[i][0][0] + matrix[i][0][1] + matrix[i][0][2] + matrix[i][0][3])
        average_recall[i][1] = matrix[i][1][1] / (matrix[i][1][0] + matrix[i][1][1] + matrix[i][1][2] + matrix[i][1][3])
        average_recall[i][2] = matrix[i][2][2] / (matrix[i][2][0] + matrix[i][2][1] + matrix[i][2][2] + matrix[i][2][3])
        average_recall[i][3] = matrix[i][3][3] / (matrix[i][3][0] + matrix[i][3][1] + matrix[i][3][2] + matrix[i][3][3])
        ar[i] = (average_recall[i][0]+average_recall[i][1]+average_recall[i][2]+average_recall[i][3]) / 4
        
        print('--------------------------------'+ modelName[i] +'---------------------------------------')
        print('Average Recall : ', ar[i])
        if i == 1 :
            acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2] + matrix[i][3][3]) / len(YTestNegative)
        if i == 3 :
            acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2] + matrix[i][3][3]) / len(YTestPositive)
        print('Acc. : ', acc[i])
    elif i == 2 :
        average_recall[i][0] = matrix[i][0][0] / (matrix[i][0][0] + matrix[i][0][1])
        average_recall[i][1] = matrix[i][1][1] / (matrix[i][1][0] + matrix[i][1][1])
        ar[i] = (average_recall[i][0]+average_recall[i][1]) / 2
        acc[i] = (matrix[i][0][0] + matrix[i][1][1]) / len(YTestNeutral)
        print('--------------------------------'+ modelName[i] +'---------------------------------------')
        print('Average Recall : ', ar[i])
        print('Acc. : ', acc[i])
    else :
        average_recall[i][0] = matrix[i][0][0] / (matrix[i][0][0] + matrix[i][0][1] + matrix[i][0][2] + matrix[i][0][3] + matrix[i][0][4] + matrix[i][0][5] + matrix[i][0][6])
        average_recall[i][1] = matrix[i][1][1] / (matrix[i][1][0] + matrix[i][1][1] + matrix[i][1][2] + matrix[i][1][3] + matrix[i][1][4] + matrix[i][1][5] + matrix[i][1][6])
        average_recall[i][2] = matrix[i][2][2] / (matrix[i][2][0] + matrix[i][2][1] + matrix[i][2][2] + matrix[i][2][3] + matrix[i][2][4] + matrix[i][2][5] + matrix[i][2][6])
        average_recall[i][3] = matrix[i][3][3] / (matrix[i][3][0] + matrix[i][3][1] + matrix[i][3][2] + matrix[i][3][3] + matrix[i][3][4] + matrix[i][3][5] + matrix[i][3][6])
        average_recall[i][4] = matrix[i][4][4] / (matrix[i][4][0] + matrix[i][4][1] + matrix[i][4][2] + matrix[i][4][3] + matrix[i][4][4] + matrix[i][4][5] + matrix[i][4][6])
        average_recall[i][5] = matrix[i][5][5] / (matrix[i][5][0] + matrix[i][5][1] + matrix[i][5][2] + matrix[i][5][3] + matrix[i][5][4] + matrix[i][5][5] + matrix[i][5][6])
        average_recall[i][6] = matrix[i][6][6] / (matrix[i][6][0] + matrix[i][6][1] + matrix[i][6][2] + matrix[i][6][3] + matrix[i][6][4] + matrix[i][6][5] + matrix[i][6][6])
        ar[i] = (average_recall[i][0]+average_recall[i][1]+average_recall[i][2]+average_recall[i][3]+average_recall[i][4]+average_recall[i][5]+average_recall[i][6]) / 7
        acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2] + matrix[i][3][3] + matrix[i][4][4] + matrix[i][5][5] + matrix[i][6][6]) / len(YTestSeven)
        print('--------------------------------'+ modelName[i] +'---------------------------------------')
        print('Average Recall : ', ar[i])
        print('Acc. : ', acc[i])

#------------------------------------------------------------------------------------------------------------ 
print('--------------------------------------------------------------------------')
print('pearson correlation coefficient')

pred_mean = np.mean(pred, axis=0)
label_mean = np.mean(YTest18, axis=0)
print('pred_mean = ', pred_mean)
print('label_mean = ', label_mean)

covariance = np.sum(np.dot(pred-pred_mean, YTest18-label_mean))
print('covariance = ', covariance)

standard_deviation_pred = np.sqrt(np.sum(np.power(pred-pred_mean, 2)))
standard_deviation_label = np.sqrt(np.sum(np.power(YTest18-label_mean, 2)))
print('standard_deviation_pred = ', standard_deviation_pred)
print('standard_deviation_label = ', standard_deviation_label)

pearson = covariance / (standard_deviation_pred * standard_deviation_label)
print('pearson = ', pearson)
