#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'
import time,datetime
import numpy as np
import csv
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydot
import math
'''for SOSP data using Decision Tree'''
#============decision Tree on SOSP data==============#
#													 #
#													 #
#=====================================================

para={
'path':'../Data/SOSP_data/',
'matrixFile':'rm_repeat_rawTFVector.txt',   
'labelFile':'rm_repeat_mlabel.txt',             
'trainingSetPercent': 0.8, #which means how many(percentage) time windows would be regarded as training set
}

def processData(para):
	print('Loading data for SOSP using decision Tree...')
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

def makePrediction(para,rawData,totalNumRows,labels):
	traingSetSize=int(math.floor(totalNumRows*para['trainingSetPercent']))
	print('%d instances are selected as training dataset!'%traingSetSize)
	trainX=np.array(rawData[0:traingSetSize])
	trainY=np.array(labels[0:traingSetSize])
	clf=tree.DecisionTreeClassifier()
	clf=clf.fit(trainX,trainY)

	feaNames=['event'+str(i) for i in range(1,386)]
	classNames=trainY
	dot_data = StringIO()  #class_names=classNames,
	tree.export_graphviz(clf, out_file=dot_data, feature_names=feaNames,  
                         filled=True, rounded=True,  
                         special_characters=True)  
	graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
	graph.write_png('sample_SOSP.png')

	testingX=rawData[traingSetSize:]
	testingY=labels[traingSetSize:]
	# testingX = trainX
	# testingY = trainY
	prediction=list(clf.predict(testingX))
	if len(prediction)!=len(testingY):
		print ('prediction and testingY have different length and SOMEWHERE WRONG!')
	sameLabelNum=0
	sameFailureNum=0
	for i in range(len(testingY)):
		if prediction[i]==testingY[i]:
			sameLabelNum+=1
			if prediction[i]==1:
				sameFailureNum+=1

	accuracy=float(sameLabelNum)/len(testingY)
	print ('accuracy is %.5f:'%accuracy)

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
		print ('precision is 0 and recall is 0')
	else:
		precision=float(sameFailureNum)/(predictFailure)
		print('precision is %.5f'%precision)
		recall=float(sameFailureNum)/(testFailure)
		print('recall is %.5f'%recall)
		F_measure=2*precision*recall/(precision+recall)
		print('F_measure is %.5f'%F_measure)
	#print predictFailure,testFailure,sameFailureNum,precision,recall,F_measure
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def mainProcess(para):
	rawData,totalNumRows,labels = processData(para)
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure = makePrediction(para,rawData,totalNumRows,labels)
	
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

#mainProcess(para)
def repeat(para):
	repeatNum = 6
	result=np.zeros((repeatNum,6))
	i=0
	for ti in range(repeatNum):
		result[i,:]=mainProcess(para)
		i+=1
	print result
	np.savetxt('result_decisionTree_SOSP_6times.csv',result,delimiter=',')

repeat(para)