from __future__ import print_function
import numpy as np

#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Reading predict probability of system...')

#  predict probability
predprobDevFileName = ['./pre-trainedResults/probability/sysemPredictProbability_dev_glove-t.txt',
			   './pre-trainedResults/probability/sysemPredictProbability_dev_glove-g.txt',
			   './pre-trainedResults/probability/sysemPredictProbability_dev_acl2015.txt',
			   './pre-trainedResults/probability/sysemPredictProbability_dev_word2vec.txt',
			   './pre-trainedResults/probability/sysemPredictProbability_dev_self.txt']

predprobTestFileName = ['./pre-trainedResults/probability/sysemPredictProbability_test_glove-t.txt',
			    './pre-trainedResults/probability/sysemPredictProbability_test_glove-g.txt',
			    './pre-trainedResults/probability/sysemPredictProbability_test_acl2015.txt',
			    './pre-trainedResults/probability/sysemPredictProbability_test_word2vec.txt',
			    './pre-trainedResults/probability/sysemPredictProbability_test_self.txt']

predprobDev = []
predprobTest = []
for i in range(len(predprobDevFileName)) :
	predprobDev.append(np.loadtxt(predprobDevFileName[i]))
	predprobTest.append(np.loadtxt(predprobTestFileName[i]))
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Loading label...')

DevLabel = np.loadtxt('./Data/processed/2018-Valence-oc-En-dev-label.txt')
TestLabel = np.loadtxt('./Data/processed/2018-Valence-oc-En-test-label.txt')
#------------------------------------------------------------------------------------------------------------
print('--------------------------------------------------')
print('Weighted Average...')
s = 0.1
p = 0
a = 0
w = np.zeros(len(predprobDevFileName))
for a1 in range(10) :
    for a2 in range(10) :
        for a3 in range(10) :
            for a4 in range(10) :
                for a5 in range(10) :
					if a1+a2+a3+a4+a5==10 :
						predMean = a1*s*predprobDev[0] + a2*s*predprobDev[1] + a3*s*predprobDev[2] + a4*s*predprobDev[3] + a5*s*predprobDev[4]
						print('--------------------------------------------------------------------------')
						print('a1={},a2={},a3={},a4={},a5={}'.format(s*a1,s*a2,s*a3,s*a4,s*a5))
					
						#--------------Metric--------------
						matrix = np.zeros((7, 7))
						pred = []
						#Calculate confusion matrix
						for i, (tar, Label) in enumerate( zip(DevLabel, predMean) ) :
							m = np.max(Label)
							for j, value in enumerate(Label) :
								if value == m :
									pred.append(int(j - 3))
									matrix[int(tar)+3][j] += 1
									break
						#------------------------------------------------------------------------------------------------------------------------------------------
						acc = 0
						acc = (matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3] + matrix[4][4] + matrix[5][5] + matrix[6][6]) / len(predMean)
						print('Acc. : ', acc)
						#------------------------------------------------------------------------------------------------------------ 
						#print('pearson correlation coefficient')
						pred_mean = np.mean(pred, axis=0)
						label_mean = np.mean(DevLabel, axis=0)
						covariance = np.sum(np.dot(pred-pred_mean, DevLabel-label_mean))
						standard_deviation_pred = np.sqrt(np.sum(np.power(pred-pred_mean, 2)))
						standard_deviation_label = np.sqrt(np.sum(np.power(DevLabel-label_mean, 2)))
						pearson = covariance / (standard_deviation_pred * standard_deviation_label)
						print('pearson = ', pearson)
						if pearson > p :
						#if acc > a :
							a = acc
							p = pearson
							w[0]=s*a1
							w[1]=s*a2
							w[2]=s*a3
							w[3]=s*a4
							w[4]=s*a5
print('--------------------------------------------------------------------------')
print('The best results of dev are pearson={}, accuracy={}'.format(p,a))
print('The weights are {}, {}, {}, {}, {}'.format(w[0],w[1],w[2],w[3],w[4]))
#------------------------------------------------------------------------------------------------------------

predMeanTest = w[0]*predprobTest[0] + w[1]*predprobTest[1] + w[2]*predprobTest[2] + w[3]*predprobTest[3] + w[4]*predprobTest[4]
#--------------Metric--------------
matrix = np.zeros((7, 7))
pred = []
#Calculate confusion matrix
for i, (tar, Label) in enumerate( zip(TestLabel, predMeanTest) ) :
	m = np.max(Label)
	for j, value in enumerate(Label) :
		if value == m :
			pred.append(int(j - 3))
			matrix[int(tar)+3][j] += 1
			break
#------------------------------------------------------------------------------------------------------------------------------------------
acc = 0
acc = (matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3] + matrix[4][4] + matrix[5][5] + matrix[6][6]) / len(predMeanTest)
#------------------------------------------------------------------------------------------------------------ 
#print('pearson correlation coefficient')
pred_mean = np.mean(pred, axis=0)
label_mean = np.mean(TestLabel, axis=0)
covariance = np.sum(np.dot(pred-pred_mean, TestLabel-label_mean))
standard_deviation_pred = np.sqrt(np.sum(np.power(pred-pred_mean, 2)))
standard_deviation_label = np.sqrt(np.sum(np.power(TestLabel-label_mean, 2)))
pearson = covariance / (standard_deviation_pred * standard_deviation_label)
print('The results of test are pearson={}, accuracy={}'.format(pearson,acc))
