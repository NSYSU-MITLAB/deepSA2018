from __future__ import print_function
import numpy as np
from numpy import zeros, newaxis

#np.random.seed(1337)  # for reproducibility

from keras import regularizers
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input, GlobalMaxPooling1D
from keras.layers import LSTM, SimpleRNN, GRU, RepeatVector, Permute, merge, Flatten, Lambda, Concatenate
from keras.layers import Bidirectional, MaxPooling1D, Conv1D, AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
#from keras.utils.visualize_util import plot

from keras import backend as K
import tensorflow as tf
import h5py

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
#------------------------------------------------------------------------------------------------------------
batch_size = 32
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading word list...')

word_list = {}
#f = open('SemEval-2017-2018-wordList-earlystopping.txt', 'r')
#f = open('SemEval-2018-wordList.txt', 'r')
#f = open('./wordList/wordList-2018.txt', 'r')
f = open('./wordList/wordList-2017-2018.txt', 'r')
for line in f.readlines():
    values = line.split()
    coefs = values[0]
    #word = values[2]
    word = values[1]
    word_list[word] = coefs
f.close()

print('word list :', len(word_list))
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Load Lexicons...')

lexicon = [{},{},{},{}]
fileName = ['normalize_afinn_score.txt', 'normalize_Sentiment140_score.txt', 'normalize_sentistrength_score.txt', 'normalize_vader_score.txt']
for i in range(len(fileName)) :
    LexiconFile = open('./Lexicon_method/sentiment_score/' + fileName[i], 'r')
    for line in LexiconFile.readlines() :
        token = line.split('\t')
        if lexicon[i].get(token[0]) is None : 
            lexicon[i][token[0]] = float(token[1].split('\n')[0]) ########### normalize [-1,1]
    LexiconFile.close()

print('afinn lexicon :', len(lexicon[0]))
print('Sentiment140 lexicon :', len(lexicon[1]))
print('sentistrength lexicon :', len(lexicon[2]))
print('vader lexicon :', len(lexicon[3]))
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading Data...')

#Load train data
data=[]
score=[]
#f = open('./train_data/new/2017-2018-Valence-oc-En-train-data.tok', 'r')
#f = open('./train_data/new/2018-Valence-oc-En-subtrain-data.tok', 'r')
f = open('./train_data/new/all-Valence-oc-En-train-data.tok', 'r')
for line in f.readlines():
    temp=[]
    tempScore=[]
    sp=line.split()
    for word in sp:
	if word in word_list :
            temp.append(int(word_list[word]))
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
    score.append(tempScore)
X_train = np.asarray(data)
ScoreTrain = np.asarray(score)
f.close()

#Load train data2
data=[]
score=[]
f = open('./train_data/new/2018-Valence-oc-En-train-data.tok', 'r')
#f = open('./train_data/new/2018-Valence-oc-En-subtrain-data.tok', 'r')
for line in f.readlines():
    temp=[]
    tempScore=[]
    sp=line.split()
    for word in sp:
	if word in word_list :
            temp.append(int(word_list[word]))
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
    score.append(tempScore)
X_train2 = np.asarray(data)
ScoreTrain2 = np.asarray(score)
f.close()
'''
#Load dev data
data=[]
#f = open('./dev_data/2018-Valence-oc-En-dev-data.tok', 'r')
#f = open('./train_data/SemEval-2018-subdev.tok', 'r')
#f = open('./dev_data/2018-Valence-oc-En-dev-data.tok', 'r')
f = open('./train_data/new/2018-Valence-oc-En-subdev-data.tok', 'r')
for line in f.readlines():
    temp=[]
    sp=line.split()
    for word in sp:
	if word in word_list :
            temp.append(int(word_list[word]))
    data.append(temp)
X_dev = np.asarray(data)
f.close()
'''
#Load test data
data=[]
score=[]
#f = open('./dev_data/2018-Valence-oc-En-dev-data.tok', 'r')
f = open('./test_data/2018-Valence-oc-En-test-data.tok', 'r')
for line in f.readlines():
    temp=[]
    tempScore=[]
    sp=line.split()
    for word in sp:
        if word in word_list :
            temp.append(int(word_list[word]))
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
    score.append(tempScore)
X_test = np.asarray(data)
ScoreTest = np.asarray(score)
f.close()

print('Padding sequences...')
X_train = sequence.pad_sequences(X_train, maxlen=60)#, maxlen=99, maxlen=56
X_train2 = sequence.pad_sequences(X_train2, maxlen=60)
#X_dev = sequence.pad_sequences(X_dev, maxlen=56)
X_test = sequence.pad_sequences(X_test, maxlen=60)
print('train data :', X_train.shape)
print('train data2 :', X_train2.shape)
#print('dev data :', X_dev.shape)
print('test data :', X_test.shape)

ScoreTrain = sequence.pad_sequences(ScoreTrain, maxlen=60)
ScoreTrain2 = sequence.pad_sequences(ScoreTrain2, maxlen=60)
ScoreTest = sequence.pad_sequences(ScoreTest, maxlen=60)
print('Score Train :', ScoreTrain.shape)
print('Score Train2 :', ScoreTrain2.shape)
print('Score Test :', ScoreTest.shape)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading Label...')

#Load train label
train_label_count = [0,0,0]
#train_label_count = [0,0,0,0,0,0,0]
y_train = []
f = open('./train_data/new/all-Valence-oc-En-train-label-3.txt', 'r')
#f = open('./train_data/new/2017-2018-Valence-oc-En-train-label.txt', 'r')
#f = open('./train_data/new/2018-Valence-oc-En-subtrain-label.txt','r')
for line in f.readlines():
    if int(line)==3:
	train_label_count[2] += 1
        y_train.append([0,0,1])
        #y_train.append([0,0,0,0,0,0,1])
    if int(line)==2:
	train_label_count[2] += 1
        y_train.append([0,0,1])
        #y_train.append([0,0,0,0,0,1,0])
    if int(line)==1:
	train_label_count[2] += 1
        y_train.append([0,0,1])
        #y_train.append([0,0,0,0,1,0,0])
    if int(line)==0:
	train_label_count[1] += 1
        y_train.append([0,1,0])
        #y_train.append([0,0,0,1,0,0,0])
    if int(line)==-1:
	train_label_count[0] += 1
        y_train.append([1,0,0])
        #y_train.append([0,0,1,0,0,0,0])
    if int(line)==-2:
	train_label_count[0] += 1
        y_train.append([1,0,0])
        #y_train.append([0,1,0,0,0,0,0])
    if int(line)==-3:
	train_label_count[0] += 1
        y_train.append([1,0,0])
        #y_train.append([1,0,0,0,0,0,0])
f.close()

y_train = np.asarray(y_train)

#Load train label
train_label_count1 = [0,0,0]
train_label_count2 = [0,0,0,0]
train_label_count3 = [0,0]
train_label_count4 = [0,0,0,0]
train_label_count5 = [0,0,0,0,0,0,0]
y_train1=[]
y_train2=[]
y_train3=[]
y_train4=[]
y_train5=[]

#f = open('./train_data/semEval-2017-2018--train-label--3-0-3-earlystopping.txt','r')
#f = open('./train_data/SemEval-2018-subtrainlabel.txt','r')
#f = open('./train_data/new/2018-Valence-oc-En-subtrain-label.txt','r')
#f = open('./train_data/new/2017-2018-Valence-oc-En-train-label.txt', 'r')
#f = open('./train_data/new/all-Valence-oc-En-train-label-1.txt', 'r')
#f = open('./train_data/new/all-Valence-oc-En-train-label-3.txt', 'r')
f = open('./train_data/new/2018-Valence-oc-En-train-label.txt', 'r')
for line in f.readlines():
    if int(line)==3:
	train_label_count1[2] += 1
	train_label_count2[3] += 1
	train_label_count3[1] += 1
	train_label_count4[3] += 1
	train_label_count5[6] += 1
        y_train1.append([0,0,1])
        y_train2.append([0,0,0,1])
        y_train3.append([0,1])
        y_train4.append([0,0,0,1])
        y_train5.append([0,0,0,0,0,0,1])
    if int(line)==2:
	train_label_count1[2] += 1
	train_label_count2[3] += 1
	train_label_count3[1] += 1
	train_label_count4[2] += 1
	train_label_count5[5] += 1
        y_train1.append([0,0,1])
        y_train2.append([0,0,0,1])
        y_train3.append([0,1])
        y_train4.append([0,0,1,0])
        y_train5.append([0,0,0,0,0,1,0])
    if int(line)==1:
	train_label_count1[2] += 1
	train_label_count2[3] += 1
	train_label_count3[1] += 1
	train_label_count4[1] += 1
	train_label_count5[4] += 1
        y_train1.append([0,0,1])
        y_train2.append([0,0,0,1])
        y_train3.append([0,1])
        y_train4.append([0,1,0,0])
        y_train5.append([0,0,0,0,1,0,0])
    if int(line)==0:
	train_label_count1[1] += 1
	train_label_count2[3] += 1
	train_label_count3[0] += 1
	train_label_count4[0] += 1
	train_label_count5[3] += 1
        y_train1.append([0,1,0])
        y_train2.append([0,0,0,1])
        y_train3.append([1,0])
        y_train4.append([1,0,0,0])
        y_train5.append([0,0,0,1,0,0,0])
    if int(line)==-1:
	train_label_count1[0] += 1
	train_label_count2[2] += 1
	train_label_count3[1] += 1
	train_label_count4[0] += 1
	train_label_count5[2] += 1
        y_train1.append([1,0,0])
        y_train2.append([0,0,1,0])
        y_train3.append([0,1])
        y_train4.append([1,0,0,0])
        y_train5.append([0,0,1,0,0,0,0])
    if int(line)==-2:
	train_label_count1[0] += 1
	train_label_count2[1] += 1
	train_label_count3[1] += 1
	train_label_count4[0] += 1
	train_label_count5[1] += 1
        y_train1.append([1,0,0])
        y_train2.append([0,1,0,0])
        y_train3.append([0,1])
        y_train4.append([1,0,0,0])
        y_train5.append([0,1,0,0,0,0,0])
    if int(line)==-3:
	train_label_count1[0] += 1
	train_label_count2[0] += 1
	train_label_count3[1] += 1
	train_label_count4[0] += 1
	train_label_count5[0] += 1
        y_train1.append([1,0,0])
        y_train2.append([1,0,0,0])
        y_train3.append([0,1])
        y_train4.append([1,0,0,0])
        y_train5.append([1,0,0,0,0,0,0])
f.close()

y_train1 = np.asarray(y_train1)
y_train2 = np.asarray(y_train2)
y_train3 = np.asarray(y_train3)
y_train4 = np.asarray(y_train4)
y_train5 = np.asarray(y_train5)
'''
#Load dev label
dev_label_count1 = [0,0,0]
dev_label_count2 = [0,0,0,0]
dev_label_count3 = [0,0]
dev_label_count4 = [0,0,0,0]
dev_label_count5 = [0,0,0,0,0,0,0]
y_dev1=[]
y_dev2=[]
y_dev3=[]
y_dev4=[]
y_dev5=[]

#f = open('./dev_data/SemEval-2018-Task1-V-oc-dev-label.txt', 'r')
#f = open('./train_data/SemEval-2018-subdevlabel.txt','r')
#f = open('./dev_data/2018-Valence-oc-En-dev-label.txt', 'r')
f = open('./train_data/new/2018-Valence-oc-En-subdev-label.txt', 'r')
for line in f.readlines():
    if int(line)==3:
	dev_label_count1[2] += 1
	dev_label_count2[3] += 1
	dev_label_count3[1] += 1
	dev_label_count4[3] += 1
	dev_label_count5[6] += 1
        y_dev1.append([0,0,1])
        y_dev2.append([0,0,0,1])
        y_dev3.append([0,1])
        y_dev4.append([0,0,0,1])
        y_dev5.append([0,0,0,0,0,0,1])
    if int(line)==2:
	dev_label_count1[2] += 1
	dev_label_count2[3] += 1
	dev_label_count3[1] += 1
	dev_label_count4[2] += 1
	dev_label_count5[5] += 1
        y_dev1.append([0,0,1])
        y_dev2.append([0,0,0,1])
        y_dev3.append([0,1])
        y_dev4.append([0,0,1,0])
        y_dev5.append([0,0,0,0,0,1,0])
    if int(line)==1:
	dev_label_count1[2] += 1
	dev_label_count2[3] += 1
	dev_label_count3[1] += 1
	dev_label_count4[1] += 1
	dev_label_count5[4] += 1
        y_dev1.append([0,0,1])
        y_dev2.append([0,0,0,1])
        y_dev3.append([0,1])
        y_dev4.append([0,1,0,0])
        y_dev5.append([0,0,0,0,1,0,0])
    if int(line)==0:
	dev_label_count1[1] += 1
	dev_label_count2[3] += 1
	dev_label_count3[0] += 1
	dev_label_count4[0] += 1
	dev_label_count5[3] += 1
        y_dev1.append([0,1,0])
        y_dev2.append([0,0,0,1])
        y_dev3.append([1,0])
        y_dev4.append([1,0,0,0])
        y_dev5.append([0,0,0,1,0,0,0])
    if int(line)==-1:
	dev_label_count1[0] += 1
	dev_label_count2[2] += 1
	dev_label_count3[1] += 1
	dev_label_count4[0] += 1
	dev_label_count5[2] += 1
        y_dev1.append([1,0,0])
        y_dev2.append([0,0,1,0])
        y_dev3.append([0,1])
        y_dev4.append([1,0,0,0])
        y_dev5.append([0,0,1,0,0,0,0])
    if int(line)==-2:
	dev_label_count1[0] += 1
	dev_label_count2[1] += 1
	dev_label_count3[1] += 1
	dev_label_count4[0] += 1
	dev_label_count5[1] += 1
        y_dev1.append([1,0,0])
        y_dev2.append([0,1,0,0])
        y_dev3.append([0,1])
        y_dev4.append([1,0,0,0])
        y_dev5.append([0,1,0,0,0,0,0])
    if int(line)==-3:
	dev_label_count1[0] += 1
	dev_label_count2[0] += 1
	dev_label_count3[1] += 1
	dev_label_count4[0] += 1
	dev_label_count5[0] += 1
        y_dev1.append([1,0,0])
        y_dev2.append([1,0,0,0])
        y_dev3.append([0,1])
        y_dev4.append([1,0,0,0])
        y_dev5.append([1,0,0,0,0,0,0])
f.close()

y_dev1 = np.asarray(y_dev1)
y_dev2 = np.asarray(y_dev2)
y_dev3 = np.asarray(y_dev3)
y_dev4 = np.asarray(y_dev4)
y_dev5 = np.asarray(y_dev5)
'''
#Load test label
test_label_count1 = [0,0,0]
test_label_count2 = [0,0,0,0]
test_label_count3 = [0,0]
test_label_count4 = [0,0,0,0]
test_label_count5 = [0,0,0,0,0,0,0]
y_test1=[]
y_test2=[]
y_test3=[]
y_test4=[]
y_test5=[]

target = []
#f = open('./dev_data/2018-Valence-oc-En-dev-label.txt', 'r')
f = open('./test_data/2018-Valence-oc-En-test-label.txt', 'r')
for line in f.readlines():
    target.append(int(line.split('\n')[0]))
    if int(line)==3:
	test_label_count1[2] += 1
	test_label_count2[3] += 1
	test_label_count3[1] += 1
	test_label_count4[3] += 1
	test_label_count5[6] += 1
        y_test1.append([0,0,1])
        y_test2.append([0,0,0,1])
        y_test3.append([0,1])
        y_test4.append([0,0,0,1])
        y_test5.append([0,0,0,0,0,0,1])
    if int(line)==2:
	test_label_count1[2] += 1
	test_label_count2[3] += 1
	test_label_count3[1] += 1
	test_label_count4[2] += 1
	test_label_count5[5] += 1
        y_test1.append([0,0,1])
        y_test2.append([0,0,0,1])
        y_test3.append([0,1])
        y_test4.append([0,0,1,0])
        y_test5.append([0,0,0,0,0,1,0])
    if int(line)==1:
	test_label_count1[2] += 1
	test_label_count2[3] += 1
	test_label_count3[1] += 1
	test_label_count4[1] += 1
	test_label_count5[4] += 1
        y_test1.append([0,0,1])
        y_test2.append([0,0,0,1])
        y_test3.append([0,1])
        y_test4.append([0,1,0,0])
        y_test5.append([0,0,0,0,1,0,0])
    if int(line)==0:
	test_label_count1[1] += 1
	test_label_count2[3] += 1
	test_label_count3[0] += 1
	test_label_count4[0] += 1
	test_label_count5[3] += 1
        y_test1.append([0,1,0])
        y_test2.append([0,0,0,1])
        y_test3.append([1,0])
        y_test4.append([1,0,0,0])
        y_test5.append([0,0,0,1,0,0,0])
    if int(line)==-1:
	test_label_count1[0] += 1
	test_label_count2[2] += 1
	test_label_count3[1] += 1
	test_label_count4[0] += 1
	test_label_count5[2] += 1
        y_test1.append([1,0,0])
        y_test2.append([0,0,1,0])
        y_test3.append([0,1])
        y_test4.append([1,0,0,0])
        y_test5.append([0,0,1,0,0,0,0])
    if int(line)==-2:
	test_label_count1[0] += 1
	test_label_count2[1] += 1
	test_label_count3[1] += 1
	test_label_count4[0] += 1
	test_label_count5[1] += 1
        y_test1.append([1,0,0])
        y_test2.append([0,1,0,0])
        y_test3.append([0,1])
        y_test4.append([1,0,0,0])
        y_test5.append([0,1,0,0,0,0,0])
    if int(line)==-3:
	test_label_count1[0] += 1
	test_label_count2[0] += 1
	test_label_count3[1] += 1
	test_label_count4[0] += 1
	test_label_count5[0] += 1
        y_test1.append([1,0,0])
        y_test2.append([1,0,0,0])
        y_test3.append([0,1])
        y_test4.append([1,0,0,0])
        y_test5.append([1,0,0,0,0,0,0])
f.close()

y_test1 = np.asarray(y_test1)
y_test2 = np.asarray(y_test2)
y_test3 = np.asarray(y_test3)
y_test4 = np.asarray(y_test4)
y_test5 = np.asarray(y_test5)
target = np.asarray(target)

print('train label : ', train_label_count)
print('train label1 : ', train_label_count1)
print('train label2 : ', train_label_count2)
print('train label3 : ', train_label_count3)
print('train label4 : ', train_label_count4)
print('train label5 : ', train_label_count5)
'''
print('dev label1 : ', dev_label_count1)
print('dev label2 : ', dev_label_count2)
print('dev label3 : ', dev_label_count3)
print('dev label4 : ', dev_label_count4)
print('dev label5 : ', dev_label_count5)
'''
print('test label1 : ', test_label_count1)
print('test label2 : ', test_label_count2)
print('test label3 : ', test_label_count3)
print('test label4 : ', test_label_count4)
print('test label5 : ', test_label_count5)
print('target : ', target.shape)

#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading word vectors...')

embeddings_index = {}
#f = open('./vector/glove.twitter.27B.200d.txt')
#f = open('./vector/word2vec-twitter-2018.txt') #old preprocessing
#f = open('./vector/word2vec-2018.txt') #new preprocessing
#f = open('./vector/word2vec-2017-2018.txt') #new preprocessing
#f = open('./vector/word2vec-twitter-2017-2018.txt')

#f = open('./vector/word2vec-2017-2018.txt') #ACL W-NUT 2015 400d
#f = open('./vector/glove.twitter.27B.200d.2017.2018.txt') #glove twitter 200d
#f = open('./vector/glove.840B.300d-2017-2018.txt') #glove common crawl 300d
#f = open('./vector/GoogleNews-vectors-negative300-2017-2018.txt') #GoogleNews 300
f = open('./vector/selfWordVector.txt') #Self train word2vec 400d
for line in f:
    values = line.split()
    if len(values) < 50 :
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
'''
for i in range(len(embedding_matrix)) :
    if i == 0 :
        continue
    else :
        embedding_matrix[i] == np.random.rand(400)
'''
for word, i in word_list.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[int(i)] = embedding_vector

count = -1 #because embedding_matrix[0] is zero vector
for i in range(len(word_list)+1) :
    for j in range(EMBEDDING_DIM) :
        if embedding_matrix[i][j] != 0 :
            break
        if j == EMBEDDING_DIM-1 :
            count += 1

print('embedding_matrix :', embedding_matrix.shape)
print('Number of zero embedding :', count)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('input to embedding...')

def word_embedding(data, score, dim):
    a=[]
    for idx1, i in enumerate(data):
        b=[]
        for idx2, j in enumerate(i):
            c = list( np.zeros(dim) )
            c = embedding_matrix[j]
            c = np.concatenate((c, score[idx1][idx2]), axis=0)
            b.append(c)
        a.append(b)
    a=np.asarray(a)
    
    return a

X_train = word_embedding(X_train, ScoreTrain, EMBEDDING_DIM)
X_train2 = word_embedding(X_train2, ScoreTrain2, EMBEDDING_DIM)
#X_dev = word_embedding(X_dev, EMBEDDING_DIM)
X_test = word_embedding(X_test, ScoreTest, EMBEDDING_DIM)

#X_train = np.concatenate((X_train, ScoreTrain), axis=2)
#X_train2 = np.concatenate((X_train2, ScoreTrain2), axis=2)
#X_test = np.concatenate((X_test, ScoreTest), axis=2)

print('X_train.shape :', X_train.shape)
print('X_train2.shape :', X_train2.shape)
#print('X_dev.shape :', X_dev.shape)
print('X_test.shape :', X_test.shape)
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Training...')

def getMatrix(x) :
    return x[:,-1,:]

def normalization(x) :
    x = (x-0.5)*6
    return x

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
_input = Input(shape=[X_train.shape[1], X_train.shape[2]], dtype='float32')

monitors = ['val_loss', 'val_loss', 'val_loss', 'val_loss', 'val_loss']
patiences = [10, 15, 15, 15, 15]
modelName = ['Three', 'Negative', 'Neural', 'Positive', 'Seven']
earlyStopping1 = EarlyStopping(monitor=monitors[0], min_delta=0, patience=patiences[0], verbose=0, mode='auto')
checkpoint1 = ModelCheckpoint('./save_model/ensemble/'+modelName[0]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb1 = [earlyStopping1, checkpoint1]
earlyStopping2 = EarlyStopping(monitor=monitors[1], min_delta=0, patience=patiences[1], verbose=0, mode='auto')
checkpoint2 = ModelCheckpoint('./save_model/ensemble/'+modelName[1]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb2 = [earlyStopping2, checkpoint2]
earlyStopping3 = EarlyStopping(monitor=monitors[2], min_delta=0, patience=patiences[2], verbose=0, mode='auto')
checkpoint3 = ModelCheckpoint('./save_model/ensemble/'+modelName[2]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb3 = [earlyStopping3, checkpoint3]
earlyStopping4 = EarlyStopping(monitor=monitors[3], min_delta=0, patience=patiences[3], verbose=0, mode='auto')
checkpoint4 = ModelCheckpoint('./save_model/ensemble/'+modelName[3]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb4 = [earlyStopping4, checkpoint4]
earlyStopping5 = EarlyStopping(monitor=monitors[4], min_delta=0, patience=patiences[4], verbose=0, mode='auto')
checkpoint5 = ModelCheckpoint('./save_model/ensemble/'+modelName[4]+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb5 = [earlyStopping5, checkpoint5]
'''
kw1 = 14310.0
three_class_weight = {0:kw1/8956.0, 1:kw1/18461.0, 2:kw1/15514.0}
kw2 = 150.0
negative_class_weight = {0:kw2/111.0, 1:kw2/204.0, 2:kw2/60.0, 3:kw2/570.0}
kw3 = 400.0
neutral_class_weight = {0:kw3/275.0, 1:kw3/670.0}
kw4 = 150.0
positive_class_weight = {0:kw4/650.0, 1:kw4/128.0, 2:kw4/74.0, 3:kw4/93.0}
kw5 = 120.0
seven_class_weight = {0:kw5/111.0, 1:kw5/204.0, 2:kw5/60.0, 3:kw5/275.0, 4:kw5/128.0, 5:kw5/74.0, 6:kw5/93.0}
print('three_class_weight = ', three_class_weight)
print('negative_class_weight = ', negative_class_weight)
print('neutral_class_weight = ', neutral_class_weight)
print('positive_class_weight = ', positive_class_weight)
print('seven_class_weight = ', seven_class_weight)
'''
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

# {-1, 0, 1}------------------------------------------------------------------------------------------
'''
cnn1 = Conv1D(200, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN1')(_input)
cnn1 = Dropout(0.5)(cnn1)
cnn2 = Conv1D(200, 5, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN2')(_input)
cnn2 = Dropout(0.5)(cnn2)
cnn3 = Conv1D(200, 7, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN3')(_input)
cnn3 = Dropout(0.5)(cnn3)
mp = merge([cnn1, cnn2, cnn3], mode='sum')
mp = AveragePooling1D(pool_size=3, strides=3, padding='valid')(mp)
'''
ThreeLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='ThreeLstm')(_input)
ThreeLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(ThreeLstm)
ThreeDense = Dense(200, activation='tanh', name='ThreeRepre')(ThreeLstmSum)
ThreeOutput = Dense(3, activation='softmax')(ThreeDense)

model1 = Model(input=_input, output=ThreeOutput)
model1.summary()
model1.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
#model1.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, validation_data=(X_test, y_test1), callbacks=cb1, verbose=1, class_weight=three_class_weight) #verbose=2
#model1.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev1), callbacks=cb1, verbose=1) #verbose=2
#model1.fit(X_train, y_train1, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev1), callbacks=cb1, verbose=2) #verbose=2
model1.load_weights('./save_model/ensemble/' + modelName[0] + '.hdf5')
'''
w01 = model1.get_layer(name='CNN1').get_weights()
w02 = model1.get_layer(name='CNN2').get_weights()
w03 = model1.get_layer(name='CNN3').get_weights()
'''
w1 = model1.get_layer(name='ThreeLstm').get_weights()

# {-3, -2, -1, other}----------------------------------------------------------------------------------
'''
cnn1 = Conv1D(200, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN1')(_input)
cnn1 = Dropout(0.5)(cnn1)
cnn2 = Conv1D(200, 5, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN2')(_input)
cnn2 = Dropout(0.5)(cnn2)
cnn3 = Conv1D(200, 7, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN3')(_input)
cnn3 = Dropout(0.5)(cnn3)
mp = merge([cnn1, cnn2, cnn3], mode='sum')
mp = AveragePooling1D(pool_size=3, strides=3, padding='valid')(mp)
'''
NegativeLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='NegativeLstm1')(_input)
NegativeLstm = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='NegativeLstm2')(NegativeLstm)
NegativeLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(NegativeLstm)
NegativeDense = Dense(200, activation='tanh', name='NegativeRepre')(NegativeLstmSum)
NegativeOutput = Dense(4, activation='softmax')(NegativeDense)

model2 = Model(input=_input, output=NegativeOutput)
model2.summary()
'''
model2.get_layer(name='CNN1').set_weights(w01)
model2.get_layer(name='CNN2').set_weights(w02)
model2.get_layer(name='CNN3').set_weights(w03)
'''
model2.get_layer(name='NegativeLstm1').set_weights(w1)
model2.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
#model2.fit(X_train2, y_train2, batch_size=batch_size, nb_epoch=200, validation_data=(X_test, y_test2), callbacks=cb2, verbose=1, class_weight=negative_class_weight) #verbose=2
#model2.fit(X_train2, y_train2, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev2), callbacks=cb2, verbose=1) #verbose=2
#model2.fit(X_train, y_train2, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev2), callbacks=cb2, verbose=2) #verbose=2
model2.load_weights('./save_model/ensemble/' + modelName[1] + '.hdf5')
w2 = model2.get_layer(name='NegativeLstm2').get_weights()

# {0, other}-----------------------------------------------------------------------------------------
'''
cnn1 = Conv1D(200, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN1')(_input)
cnn1 = Dropout(0.5)(cnn1)
cnn2 = Conv1D(200, 5, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN2')(_input)
cnn2 = Dropout(0.5)(cnn2)
cnn3 = Conv1D(200, 7, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN3')(_input)
cnn3 = Dropout(0.5)(cnn3)
mp = merge([cnn1, cnn2, cnn3], mode='sum')
mp = AveragePooling1D(pool_size=3, strides=3, padding='valid')(mp)
'''
NeuralLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='NeuralLstm1')(_input)
NeuralLstm = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='NeuralLstm2')(NeuralLstm)
NeuralLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(NeuralLstm)
NeuralDense = Dense(100, activation='tanh', name='NeuralRepre')(NeuralLstmSum)
NeuralOutput = Dense(2, activation='softmax')(NeuralDense)

model3 = Model(input=_input, output=NeuralOutput)
model3.summary()
'''
model3.get_layer(name='CNN1').set_weights(w01)
model3.get_layer(name='CNN2').set_weights(w02)
model3.get_layer(name='CNN3').set_weights(w03)
'''
model3.get_layer(name='NeuralLstm1').set_weights(w1)
model3.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
#model3.fit(X_train2, y_train3, batch_size=batch_size, nb_epoch=200, validation_data=(X_test, y_test3), callbacks=cb3, verbose=1, class_weight=neutral_class_weight) #verbose=2
#model3.fit(X_train2, y_train3, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev3), callbacks=cb3, verbose=1) #verbose=2
#model3.fit(X_train, y_train3, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev3), callbacks=cb3, verbose=2) #verbose=2
model3.load_weights('./save_model/ensemble/' + modelName[2] + '.hdf5')
w3 = model3.get_layer(name='NeuralLstm2').get_weights()

# {1, 2, 3, other}-----------------------------------------------------------------------------------
'''
cnn1 = Conv1D(200, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN1')(_input)
cnn1 = Dropout(0.5)(cnn1)
cnn2 = Conv1D(200, 5, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN2')(_input)
cnn2 = Dropout(0.5)(cnn2)
cnn3 = Conv1D(200, 7, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN3')(_input)
cnn3 = Dropout(0.5)(cnn3)
mp = merge([cnn1, cnn2, cnn3], mode='sum')
mp = AveragePooling1D(pool_size=3, strides=3, padding='valid')(mp)
'''
PositiveLstm = Bidirectional(LSTM(200, dropout=0.5, return_sequences=True), merge_mode='concat', name='PositiveLstm1')(_input)
PositiveLstm = Bidirectional(LSTM(150, dropout=0.3, return_sequences=True), merge_mode='concat', name='PositiveLstm2')(PositiveLstm)
PositiveLstmSum = Lambda(lambda xin: K.mean(xin, axis=1))(PositiveLstm)
PositiveDense = Dense(200, activation='tanh', name='PositiveRepre')(PositiveLstmSum)
PositiveOutput = Dense(4, activation='softmax')(PositiveDense)

model4 = Model(input=_input, output=PositiveOutput)
model4.summary()
'''
model4.get_layer(name='CNN1').set_weights(w01)
model4.get_layer(name='CNN2').set_weights(w02)
model4.get_layer(name='CNN3').set_weights(w03)
'''
model4.get_layer(name='PositiveLstm1').set_weights(w1)
model4.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
#model4.fit(X_train2, y_train4, batch_size=batch_size, nb_epoch=200, validation_data=(X_test, y_test4), callbacks=cb4, verbose=1, class_weight=positive_class_weight) #verbose=2
#model4.fit(X_train2, y_train4, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev4), callbacks=cb4, verbose=1) #verbose=2
#model4.fit(X_train, y_train4, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev4), callbacks=cb4, verbose=2) #verbose=2
model4.load_weights('./save_model/ensemble/' + modelName[3] + '.hdf5')
w4 = model4.get_layer(name='PositiveLstm2').get_weights()

# {-3, -2, -1, 0, 1, 2, 3}---------------------------------------------------------------------------
'''
cnn1 = Conv1D(200, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN1')(_input)
cnn1 = Dropout(0.5)(cnn1)
cnn2 = Conv1D(200, 5, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN2')(_input)
cnn2 = Dropout(0.5)(cnn2)
cnn3 = Conv1D(200, 7, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='CNN3')(_input)
cnn3 = Dropout(0.5)(cnn3)
mp = merge([cnn1, cnn2, cnn3], mode='sum')
mp = AveragePooling1D(pool_size=3, strides=3, padding='valid')(mp)
'''
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

'''
kernel = [1,2,3,4,5,6]
cnn1 = Conv1D(100, kernel[0], strides=1, padding='same', activation='tanh')(SevenLstm)
mp1 = GlobalMaxPooling1D()(cnn1)

cnn2 = Conv1D(100, kernel[1], strides=1, padding='same', activation='tanh')(SevenLstm)
mp2 = GlobalMaxPooling1D()(cnn2)

cnn3 = Conv1D(100, kernel[2], strides=1, padding='same', activation='tanh')(SevenLstm)
mp3 = GlobalMaxPooling1D()(cnn3)

cnn4 = Conv1D(100, kernel[3], strides=1, padding='same', activation='tanh')(SevenLstm)
mp4 = GlobalMaxPooling1D()(cnn4)

cnn5 = Conv1D(100, kernel[4], strides=1, padding='same', activation='tanh')(SevenLstm)
mp5 = GlobalMaxPooling1D()(cnn5)

cnn6 = Conv1D(100, kernel[5], strides=1, padding='same', activation='tanh')(SevenLstm)
mp6 = GlobalMaxPooling1D()(cnn6)

representation = merge([mp1, mp2, mp3, mp4, mp5, mp6], mode='concat')
'''

SevenDense = Dense(200, activation='tanh', name='SevenRepre')(representation) #200
SevenOutput = Dense(7, activation='softmax')(SevenDense)

#model5 = Model(input=_input, output=SevenOutput)
model5 = Model(_input, SevenOutput)
model5.summary()
'''
model5.get_layer(name='CNN1').set_weights(w01)
model5.get_layer(name='CNN2').set_weights(w02)
model5.get_layer(name='CNN3').set_weights(w03)
'''
model5.get_layer(name='SevenLstm1').set_weights(w1)
model5.get_layer(name='SevenLstm2').set_weights(w2)
model5.get_layer(name='SevenLstm3').set_weights(w3)
model5.get_layer(name='SevenLstm4').set_weights(w4)

model5.compile(loss='categorical_crossentropy', optimizer=adamC, metrics=['accuracy', PCC])
#model5.fit(X_train2, y_train5, batch_size=batch_size, nb_epoch=200, validation_data=(X_test, y_test5), callbacks=cb5, verbose=1, class_weight=seven_class_weight) #verbose=2
#model5.fit(X_train2, y_train5, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev5), callbacks=cb5, verbose=1) #verbose=2
#model5.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev5), callbacks=cb5, verbose=2) #verbose=2
#model5.fit(X_train, y_train5, batch_size=batch_size, nb_epoch=200, validation_data=(X_dev, y_dev5), callbacks=cb5, verbose=2) #verbose=2
#model5.load_weights('./save_model/ensemble/' + modelName[4] + '.hdf5')

#--------------predict--------------
predict = [[], [], [], [], []]
model1.load_weights('./save_model/ensemble/' + modelName[0] + '.hdf5')
predict[0] = model1.predict(X_test, batch_size=batch_size)

model2.load_weights('./save_model/ensemble/' + modelName[1] + '.hdf5')
predict[1] = model2.predict(X_test, batch_size=batch_size)

model3.load_weights('./save_model/ensemble/' + modelName[2] + '.hdf5')
predict[2] = model3.predict(X_test, batch_size=batch_size)

model4.load_weights('./save_model/ensemble/' + modelName[3] + '.hdf5')
predict[3] = model4.predict(X_test, batch_size=batch_size)

model5.load_weights('./save_model/ensemble/' + modelName[4] + '.hdf5')
predict[4] = model5.predict(X_test, batch_size=batch_size)
#np.savetxt('./word2vec_ensembel_post_predict/ACL2015_400d_predict.txt', predict[4])

#--------------Metric--------------
matrix = [np.zeros((3, 3)), np.zeros((4, 4)), np.zeros((2, 2)), np.zeros((4, 4)), np.zeros((7, 7))]
pred_str = []
pred = []
#Calculate confusion matrix
for l in range(len(predict)) :
    if l == 0 :
        Target = y_test1
    elif l == 1 :
        Target = y_test2
    elif l == 2 :
        Target = y_test3
    elif l == 3 :
        Target = y_test4
    elif l == 4 :
        Target = y_test5
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
'''
pred_str = []
for i, Label in enumerate(predict[4]) :
    m = np.max(Label)
    for j, value in enumerate(Label) :
        if value == m :
            pred_str.append(str(j - 3))
            break
'''
f = open('./pred/ensemble/'+ modelName[4] + '.txt', 'w')
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
        acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2]) / len(y_test1)
        print('--------------------------------'+ modelName[i] +'---------------------------------------')
        print('Average Recall : ', ar[i])
        print('-1 : ' + str(matrix[i][0][0]/test_label_count1[0]) + '\n0 : ' + str(matrix[i][1][1]/test_label_count1[1]) + '\n1 : ' + str(matrix[i][2][2]/test_label_count1[2]))
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
            acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2] + matrix[i][3][3]) / len(y_test2)
            print('-3 : ' + str(matrix[i][0][0]/test_label_count2[0]) + '\n-2 : ' + str(matrix[i][1][1]/test_label_count2[1]) + '\n-1 : ' + str(matrix[i][2][2]/test_label_count2[2]) + '\nother : ' + str(matrix[i][3][3]/test_label_count2[3]))
        if i == 3 :
            acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2] + matrix[i][3][3]) / len(y_test4)
            print('other : ' + str(matrix[i][0][0]/test_label_count4[0]) + '\n1 : ' + str(matrix[i][1][1]/test_label_count4[1]) + '\n2 : ' + str(matrix[i][2][2]/test_label_count4[2]) + '\n3 : ' + str(matrix[i][3][3]/test_label_count4[3]))
        print('Acc. : ', acc[i])
    elif i == 2 :
        average_recall[i][0] = matrix[i][0][0] / (matrix[i][0][0] + matrix[i][0][1])
        average_recall[i][1] = matrix[i][1][1] / (matrix[i][1][0] + matrix[i][1][1])
        ar[i] = (average_recall[i][0]+average_recall[i][1]) / 2
        acc[i] = (matrix[i][0][0] + matrix[i][1][1]) / len(y_test3)
        print('--------------------------------'+ modelName[i] +'---------------------------------------')
        print('Average Recall : ', ar[i])
        print('0 : ' + str(matrix[i][0][0]/test_label_count3[0]) + '\nother : ' + str(matrix[i][1][1]/test_label_count3[1]))
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
        acc[i] = (matrix[i][0][0] + matrix[i][1][1] + matrix[i][2][2] + matrix[i][3][3] + matrix[i][4][4] + matrix[i][5][5] + matrix[i][6][6]) / len(y_test5)
        print('--------------------------------'+ modelName[i] +'---------------------------------------')
        print('Average Recall : ', ar[i])
        print('-3 : ' + str(matrix[i][0][0]/test_label_count5[0]) + '\n-2 : ' + str(matrix[i][1][1]/test_label_count5[1]) + '\n-1 : ' + str(matrix[i][2][2]/test_label_count5[2]) + '\n 0 : ' + str(matrix[i][3][3]/test_label_count5[3]) + '\n 1 : ' + str(matrix[i][4][4]/test_label_count5[4]) + '\n 2 : ' + str(matrix[i][5][5]/test_label_count5[5]) + '\n 3 : ' + str(matrix[i][6][6]/test_label_count5[6]))
        print('Acc. : ', acc[i])

#------------------------------------------------------------------------------------------------------------ 
print('--------------------------------------------------------------------------')
print('pearson correlation coefficient')

pred_mean = np.mean(pred, axis=0)
label_mean = np.mean(target, axis=0)
print('pred_mean = ', pred_mean)
print('label_mean = ', label_mean)

covariance = np.sum(np.dot(pred-pred_mean, target-label_mean))
print('covariance = ', covariance)

standard_deviation_pred = np.sqrt(np.sum(np.power(pred-pred_mean, 2)))
standard_deviation_label = np.sqrt(np.sum(np.power(target-label_mean, 2)))
print('standard_deviation_pred = ', standard_deviation_pred)
print('standard_deviation_label = ', standard_deviation_label)

pearson = covariance / (standard_deviation_pred * standard_deviation_label)
print('pearson = ', pearson)
'''
f = open('./pearson/pearson.txt', 'a')
f.write(modelName[i] + ' ---> Avg.Recall='+str(ar[4])+', Acc.='+str(acc[4])+', pearson='+str(pearson) + '\n')
f.close()
'''
sys.exit(1)
#representation of DeepMoji
trainRepresentation = np.loadtxt('../../../DeepMoji-master/examples/Representation/Representation_semeval2018_last_0.768_0.448_train.txt')
devRepresentation = np.loadtxt('../../../DeepMoji-master/examples/Representation/Representation_semeval2018_last_0.768_0.448_dev.txt')


m1 = Model(input=_input, output=model1.get_layer(name='ThreeRepre').output)
m2 = Model(input=_input, output=model2.get_layer(name='NegativeRepre').output)
m3 = Model(input=_input, output=model3.get_layer(name='NeuralRepre').output)
m4 = Model(input=_input, output=model4.get_layer(name='PositiveRepre').output)
m5 = Model(input=_input, output=model5.get_layer(name='SevenRepre').output)

r1 = m1.predict(X_train2, batch_size=batch_size)
r2 = m2.predict(X_train2, batch_size=batch_size)
r3 = m3.predict(X_train2, batch_size=batch_size)
r4 = m4.predict(X_train2, batch_size=batch_size)
r5 = m5.predict(X_train2, batch_size=batch_size)

rr1 = m1.predict(X_test, batch_size=batch_size)
rr2 = m2.predict(X_test, batch_size=batch_size)
rr3 = m3.predict(X_test, batch_size=batch_size)
rr4 = m4.predict(X_test, batch_size=batch_size)
rr5 = m5.predict(X_test, batch_size=batch_size)

ip1_ = Input(shape=[r1.shape[1]], dtype='float32')
ip2_ = Input(shape=[r2.shape[1]], dtype='float32')
ip3_ = Input(shape=[r3.shape[1]], dtype='float32')
ip4_ = Input(shape=[r4.shape[1]], dtype='float32')
ip5_ = Input(shape=[r5.shape[1]], dtype='float32')
ip6_ = Input(shape=[trainRepresentation.shape[1]], dtype='float32')

i = Concatenate()([ip1_,ip2_,ip3_,ip4_,ip5_,ip6_])
o = Dense(7, activation='softmax')(i)

adamCC = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
checkpoint = ModelCheckpoint('./save_model/ensemble/RRRRRRRRRRRRRRR.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
cb = [earlyStopping, checkpoint]
m = Model(input=[ip1_,ip2_,ip3_,ip4_,ip5_,ip6_], output=o)
m.summary()
m.compile(loss='categorical_crossentropy', optimizer=adamCC, metrics=['accuracy', PCC])
m.fit([r1,r2,r3,r4,r5,trainRepresentation], y_train5, batch_size=batch_size, nb_epoch=1000, validation_data=([rr1,rr2,rr3,rr4,rr5,devRepresentation], y_test5), verbose=1, callbacks=cb, class_weight=seven_class_weight) #verbose=2, callbacks=cb5
p = m.predict([rr1,rr2,rr3,rr4,rr5,devRepresentation], batch_size=batch_size)


#--------------Metric--------------
matrix = np.zeros((7, 7))
pred_str = []
pred = []
#Calculate confusion matrix
Target = y_test5
for i, (tar, Label) in enumerate( zip(Target, p) ) :
    m = np.max(Label)
    for j, value in enumerate(Label) :
        if value == m :
            pred_str.append(str(j - 3))
            pred.append(int(j - 3))
            for k, num in enumerate(tar) :
                if num == 1 :
                    matrix[k][j] += 1
                    break
            break

#--------------Save Predict--------------
f = open('./pred/ensemble/RRRRRRRRRRRRR.txt', 'w')
f.write('\n'.join(pred_str))
f.close()

#------------------------------------------------------------------------------------------------------------
average_recall = np.zeros((7))
ar = 0
acc = 0
average_recall[0] = matrix[0][0] / (matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3] + matrix[0][4] + matrix[0][5] + matrix[0][6])
average_recall[1] = matrix[1][1] / (matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3] + matrix[1][4] + matrix[1][5] + matrix[1][6])
average_recall[2] = matrix[2][2] / (matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3] + matrix[2][4] + matrix[2][5] + matrix[2][6])
average_recall[3] = matrix[3][3] / (matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3] + matrix[3][4] + matrix[3][5] + matrix[3][6])
average_recall[4] = matrix[4][4] / (matrix[4][0] + matrix[4][1] + matrix[4][2] + matrix[4][3] + matrix[4][4] + matrix[4][5] + matrix[4][6])
average_recall[5] = matrix[5][5] / (matrix[5][0] + matrix[5][1] + matrix[5][2] + matrix[5][3] + matrix[5][4] + matrix[5][5] + matrix[5][6])
average_recall[6] = matrix[6][6] / (matrix[6][0] + matrix[6][1] + matrix[6][2] + matrix[6][3] + matrix[6][4] + matrix[6][5] + matrix[6][6])
ar = (average_recall[0]+average_recall[1]+average_recall[2]+average_recall[3]+average_recall[4]+average_recall[5]+average_recall[6]) / 7
acc = (matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3] + matrix[4][4] + matrix[5][5] + matrix[6][6]) / len(y_test5)
print('Average Recall : ', ar)
print('-3 : ' + str(matrix[0][0]/test_label_count5[0]) + '\n-2 : ' + str(matrix[1][1]/test_label_count5[1]) + '\n-1 : ' + str(matrix[2][2]/test_label_count5[2]) + '\n 0 : ' + str(matrix[3][3]/test_label_count5[3]) + '\n 1 : ' + str(matrix[4][4]/test_label_count5[4]) + '\n 2 : ' + str(matrix[5][5]/test_label_count5[5]) + '\n 3 : ' + str(matrix[6][6]/test_label_count5[6]))
print('Acc. : ', acc)

#------------------------------------------------------------------------------------------------------------ 
print('--------------------------------------------------------------------------')
print('pearson correlation coefficient')

pred_mean = np.mean(pred, axis=0)
label_mean = np.mean(target, axis=0)
print('pred_mean = ', pred_mean)
print('label_mean = ', label_mean)

covariance = np.sum(np.dot(pred-pred_mean, target-label_mean))
print('covariance = ', covariance)

standard_deviation_pred = np.sqrt(np.sum(np.power(pred-pred_mean, 2)))
standard_deviation_label = np.sqrt(np.sum(np.power(target-label_mean, 2)))
print('standard_deviation_pred = ', standard_deviation_pred)
print('standard_deviation_label = ', standard_deviation_label)

pearson = covariance / (standard_deviation_pred * standard_deviation_label)
print('pearson = ', pearson)
