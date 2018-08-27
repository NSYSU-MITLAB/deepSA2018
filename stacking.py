from __future__ import print_function
import numpy as np

from keras import regularizers
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input
from keras.layers import LSTM, RepeatVector, Permute, merge, Flatten, Lambda, Concatenate
from keras.layers import Bidirectional
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Reading labels...')

y_train = []
f = open('./Data/processed/2018-Valence-oc-En-train-label.txt', 'r')
for line in f.readlines():
	temp = [0,0,0,0,0,0,0]
	temp[int(line.split('\n')[0])+3] = 1
	y_train.append(temp) 
f.close()
y_train = np.asarray(y_train)

y_dev = []
f = open('./Data/processed/2018-Valence-oc-En-dev-label.txt', 'r')
for line in f.readlines():
	temp = [0,0,0,0,0,0,0]
	temp[int(line.split('\n')[0])+3] = 1
	y_dev.append(temp) 
f.close()
y_dev = np.asarray(y_dev)

target = np.loadtxt('./Data/processed/2018-Valence-oc-En-test-label.txt')
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Reading representations...')

representationNameTrain = [['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_train_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_train_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_train_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_train_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_train_glove-t.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_train_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_train_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_train_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_train_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_train_glove-g.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_train_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_train_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_train_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_train_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_train_acl2015.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_train_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_train_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_train_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_train_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_train_word2vec.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_train_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_train_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_train_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_train_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_train_self.txt']]

representationNameDev = [['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_dev_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_dev_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_dev_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_dev_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_dev_glove-t.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_dev_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_dev_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_dev_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_dev_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_dev_glove-g.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_dev_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_dev_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_dev_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_dev_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_dev_acl2015.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_dev_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_dev_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_dev_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_dev_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_dev_word2vec.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_dev_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_dev_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_dev_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_dev_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_dev_self.txt']]

representationNameTest = [['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_test_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_test_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_test_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_test_glove-t.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_test_glove-t.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_test_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_test_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_test_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_test_glove-g.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_test_glove-g.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_test_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_test_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_test_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_test_acl2015.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_test_acl2015.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_test_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_test_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_test_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_test_word2vec.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_test_word2vec.txt'],
						   ['./pre-trainedResults/representation/sysemRepresentation_threeclassmodel_test_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_negativeclassmodel_test_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_neutralclassmodel_test_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_positiveclassmodel_test_self.txt',
							'./pre-trainedResults/representation/sysemRepresentation_sevenclassmodel_test_self.txt']]

DeepMojirepresentationName = ['./pre-trainedResults/representation/DeepMojiRepresentation_chain-thaw_train.txt',
							  './pre-trainedResults/representation/DeepMojiRepresentation_chain-thaw_dev.txt',
							  './pre-trainedResults/representation/DeepMojiRepresentation_chain-thaw_test.txt']

#  Representation of the second last layer of DeepMoji. (Dimension: 2304)
DeepMojiRTrain = np.loadtxt(DeepMojirepresentationName[0])
DeepMojiRDev = np.loadtxt(DeepMojirepresentationName[1])
DeepMojiRTest = np.loadtxt(DeepMojirepresentationName[2])

#  Concatenating output of the second last layer of five class models in the system. (Dimension: 900)
representationTrain = []
for i in range(len(representationNameTrain)) :
	Repre = []
	for j in range(len(representationNameTrain[i])) :
		R = np.loadtxt(representationNameTrain[i][j])
		if j == 0 :
			Repre = R
		else :
			Repre = np.concatenate((Repre, R), axis = 1)
	representationTrain.append(Repre)
representationTrain = np.asarray(representationTrain)
print(representationTrain.shape)

representationDev = []
for i in range(len(representationNameDev)) :
	Repre = []
	for j in range(len(representationNameDev[i])) :
		R = np.loadtxt(representationNameDev[i][j])
		if j == 0 :
			Repre = R
		else :
			Repre = np.concatenate((Repre, R), axis = 1)
	representationDev.append(Repre)
representationDev = np.asarray(representationDev)
print(representationDev.shape)

representationTest = []
for i in range(len(representationNameTest)) :
	Repre = []
	for j in range(len(representationNameTest[i])) :
		R = np.loadtxt(representationNameTest[i][j])
		if j == 0 :
			Repre = R
		else :
			Repre = np.concatenate((Repre, R), axis = 1)
	representationTest.append(Repre)
representationTest = np.asarray(representationTest)
print(representationTest.shape)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')

earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
checkpoint = ModelCheckpoint('./saveModel/stacking.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb = [earlyStopping, checkpoint]
adamC = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, clipnorm=5.0)
_input1 = Input(shape=[representationTrain[0].shape[1]], dtype='float32')
_input2 = Input(shape=[representationTrain[1].shape[1]], dtype='float32')
_input3 = Input(shape=[representationTrain[2].shape[1]], dtype='float32')
_input4 = Input(shape=[representationTrain[3].shape[1]], dtype='float32')
_input5 = Input(shape=[representationTrain[4].shape[1]], dtype='float32')
_input6 = Input(shape=[DeepMojiRTrain.shape[1]], dtype='float32')
R1 = Dense(50, activation='tanh')(_input1)
R2 = Dense(50, activation='tanh')(_input2)
R3 = Dense(50, activation='tanh')(_input3)
R4 = Dense(50, activation='tanh')(_input4)
R5 = Dense(50, activation='tanh')(_input5)
R6 = Dense(200, activation='tanh')(_input6)
R = Concatenate()([R1, R2, R3, R4, R5, R6])
Output = Dense(7, activation='softmax')(R)
model = Model(inputs=[_input1, _input2, _input3, _input4, _input5, _input6], outputs=Output)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy'])
model.fit([representationTrain[0], representationTrain[1], representationTrain[2], representationTrain[3], representationTrain[4], DeepMojiRTrain], y_train, batch_size=64, nb_epoch=1000, validation_data=([representationDev[0], representationDev[1], representationDev[2], representationDev[3], representationDev[4], DeepMojiRDev], y_dev), callbacks=cb, verbose=1)

model.load_weights('./saveModel/stacking.hdf5')
predMean = model.predict([representationTest[0], representationTest[1], representationTest[2], representationTest[3], representationTest[4], DeepMojiRTest], batch_size=64)		
#--------------Metric--------------
matrix = np.zeros((7, 7))
pred_str = []
pred = []
#Calculate confusion matrix
for i, (tar, Label) in enumerate( zip(target, predMean) ) :
	m = np.max(Label)
	for j, value in enumerate(Label) :
		if value == m :
			pred.append(int(j - 3))
			matrix[int(tar)+3][j] += 1
			break
#------------------------------------------------------------------------------------------------------------
acc = 0
acc = (matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3] + matrix[4][4] + matrix[5][5] + matrix[6][6]) / len(predMean)
print('Acc. : ', acc)
#------------------------------------------------------------------------------------------------------------ 
pred_mean = np.mean(pred, axis=0)
label_mean = np.mean(target, axis=0)
covariance = np.sum(np.dot(pred-pred_mean, target-label_mean))
standard_deviation_pred = np.sqrt(np.sum(np.power(pred-pred_mean, 2)))
standard_deviation_label = np.sqrt(np.sum(np.power(target-label_mean, 2)))
pearson = covariance / (standard_deviation_pred * standard_deviation_label)
print('pearson = ', pearson)

