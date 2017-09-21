#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'
import time,datetime
import numpy as np
import csv
from sklearn import svm
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot
import math

'''for HDFS data using SVM'''
#==================SVM on HDFS data==================#
#													 #
#													 #
#=====================================================

para={
'path':'../Data/HDFS_data/', # data path
'matrixFile':'rm_repeat_rawTFVector.txt', # event count matrix for HDFS data
'labelFile':'rm_repeat_mlabel.txt', 	# labels for log sequences
'trainingSetPercent': 0.8  #80% percent of data are selected as training set
}

def processData(para):
	print('for HDFS data using SVM')
	print('Loading data...')
	numEvents=29
	totalNumRows=575061
	filePath=para['path']+para['matrixFile']
	labelFilePath=para['path']+para['labelFile']
	f=open(filePath)
	flabel=open(labelFilePath)
	labellines=flabel.readlines()
	sucRowCount=0
	labels=np.zeros((totalNumRows,1))
	i=0
	for laline in labellines:
		lab=int(laline.split()[0])
		labels[i]=lab
		if lab==0:
			sucRowCount+=1
		i+=1
	print('success label count is %d'%sucRowCount)
	rawData=np.zeros((totalNumRows,numEvents),int)
	lines=f.readlines()
	numLines=0

	for line in lines:
		line=line.strip().split()
		for j in range(numEvents):
			rawData[numLines,j]=int(line[j])
		numLines += 1
	f.close()
	flabel.close()
	return rawData,totalNumRows,labels

def makePrediction(para,temLogNumPerTW,timeWindow,labelTWL):
	traingSetSize=int(math.floor(timeWindow*para['trainingSetPercent']))
	print('%d timewindows are treated as training dataset!'%traingSetSize)
	trainX=np.array(temLogNumPerTW[0:traingSetSize])
	trainY=np.array(labelTWL[0:traingSetSize])
	clf=svm.LinearSVC(penalty='l1', tol=0.0001, C=1, dual=False, fit_intercept=True,
	intercept_scaling=1, class_weight='balanced',  max_iter=1000)
	clf=clf.fit(trainX,trainY.ravel())

	testingX=temLogNumPerTW[traingSetSize:]
	testingY=labelTWL[traingSetSize:]
	prediction=list(clf.predict(testingX))
	print np.count_nonzero(prediction)
	if len(prediction)!=len(testingY):
		print('prediction and testingY have different length and SOMEWHERE WRONG!')
	sameLabelNum=0
	sameFailureNum=0
	for i in range(len(testingY)):
		if prediction[i]==testingY[i]:
			sameLabelNum+=1
			if prediction[i]==1:
				sameFailureNum+=1

	accuracy=float(sameLabelNum)/len(testingY)
	print('accuracy is %.5f:'%accuracy)

	predictSuccess=0
	predictFailure=0
	for item in prediction:
		if item==0:
			predictSuccess+=1
		elif item==1:
			predictFailure+=1

	testSuccess=0
	testFailure=0
	for tt in testingY:
		if tt==0:
			testSuccess+=1
		elif tt==1:
			testFailure+=1

	print predictSuccess,predictFailure,testSuccess,testFailure,sameFailureNum
	if sameFailureNum==0:
		print('precision is 0 and recall is 0')
	else:
		precision=float(sameFailureNum)/(predictFailure)
		print('precision is %.5f'%precision)
		recall=float(sameFailureNum)/(testFailure)
		print('recall is %.5f'%recall)
		F_measure=2*precision*recall/(precision+recall)
		print('F_measure is %.5f'%F_measure)
	print predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def mainProcess(para):
	rawEventCount,timeWindow,labels=processData(para)
	makePrediction(para,rawEventCount,timeWindow,labels)

if __name__ == '__main__':
	mainProcess(para)
