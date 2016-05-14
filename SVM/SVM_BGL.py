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
import os
import random
from sklearn import svm, grid_search, datasets

'''for BGL data with time window using SVM'''
#==================SVM on BGL data===================#
#In this case, failed time windows are deleted		 #
#													 #
#=====================================================

para={
'path':'../Data/BGL_data/',
'logfileName':'BGL_MERGED.log',
'mapLogTemplate':'logTemplateMap.csv',
'selectColumn':[0,4], # select the corresponding columns in the raw data
'timeIndex':1,  # the index of time in the selected columns, start from 0
'timeInterval':1,  #the size of time window with unit of hours
'trainingSetPercent': 0.8, #which means how many(percentage) time windows would be regarded as training set
'tf-idf':False,
'balance':False
}

def processData(para):
	print('for BGL data using SVM')
	print('Loading data...')
	filePath=para['path']+para['logfileName']
	dataLL=[]
	with open(filePath) as f:
		for line in f:
			wordSeq=line.strip().split()
			row=[]
			for i in para['selectColumn']:
				row.append(wordSeq[i])
			dataLL.append(row)
	logNum=len(dataLL)
	print('we have %d logs in this file'%logNum)

	timeWindowIndex=[]
	if not os.path.exists('../timeWindowCSV/timeWindowIndex_'+str(para['timeInterval'])+'h.csv'):
		#=================divide into time windows=============#
		startTime=time.mktime(datetime.datetime.strptime(dataLL[0][para['timeIndex']], "%Y-%m-%d-%H.%M.%S.%f").timetuple())
		timeWindow=1        
		t=startTime
		for row in dataLL:
			timeStamp=time.mktime(datetime.datetime.strptime(row[para['timeIndex']], "%Y-%m-%d-%H.%M.%S.%f").timetuple())
			if timeStamp>=t and timeStamp<t+para['timeInterval']*3600:
				timeWindowIndex.append(timeWindow)
			else:
				timeWindow+=1
				t+=para['timeInterval']*3600
				timeWindowIndex.append(timeWindow)
		print('there are %d time windows in this dataset!\n'%timeWindow)
		print('%d instances in this dataset'%len(timeWindowIndex))
		np.savetxt('../timeWindowCSV/timeWindowIndex_'+str(para['timeInterval'])+'h.csv',timeWindowIndex,delimiter=',',fmt='%d')
	else:
		print('Loading time window')
		with open('../timeWindowCSV/timeWindowIndex_'+str(para['timeInterval'])+'h.csv') as timeWidowfi:
			twreader = csv.reader(timeWidowfi)
			for logrow in twreader:
				timeWindowIndex.append(int(logrow[0]))
		timeWindow=len(set(timeWindowIndex))
		print('there are %d time windows in this dataset!\n'%timeWindow)

	#=================loading data of log-event relation=============#
	logIndexPerTWLL=[]
	for t in range(timeWindow):
		newTW=[]
		newfilteredLogTWL=[]
		logIndexPerTWLL.append(newTW)
	for i in range(len(timeWindowIndex)):
		tw=timeWindowIndex[i]
		logIndexPerTWLL[tw-1].append(i) #time window start from 1, in which log index start from 0

	logTemplateL=[]
	with open(para['path']+para['mapLogTemplate']) as fi:
	  reader = csv.reader(fi)
	  for row in reader:
		  logTemplateL.append(int(row[0]))
	print('calculating the number of templates altogether\n')
	temNum=len(set(logTemplateL))
	print('there are %d templates in the whole dataset\n'%temNum)

	#=================get labels and event count of each time window =============#
	labels=[]
	rawEventCount=np.zeros((timeWindow,temNum))
	for j in range(timeWindow):
		label=0   #0 represent success, 1 represent failure
		for k in logIndexPerTWLL[j]:
			tempIndex=logTemplateL[k]
			rawEventCount[j,tempIndex]+=1
			if dataLL[k][0]!='-':
				label=1
				continue
		labels.append(label)
	print np.count_nonzero(labels)

	#TF-IDF
	if para['tf-idf']:
		print('tf-idf preprocessing')
		numLines = timeWindow
		numEvents = temNum
		idfData=np.zeros((numLines,numEvents))
		for j in range(numEvents):
			cnt=0
			for i in range(numLines):
				if rawEventCount[i,j]!=0:     
					cnt+=1  #rawData[i,j]
			if cnt == 0:
				for i in range(numLines):
					idfData[i,j] = 0
				continue
			for i in range(numLines):
				idfData[i,j]=rawEventCount[i,j]*math.log(numLines/float(cnt))
		print('idf data not rawdata')
		return idfData, timeWindow, labels
	else:
		print('rawData not idf data')
		return rawEventCount,timeWindow,labels

def makePrediction(para,temLogNumPerTW,timeWindow,labelTWL):

	traingSetSize=int(math.floor(timeWindow*para['trainingSetPercent']))
	print('%d timewindows are treated as training dataset!'%traingSetSize)
	trainX=np.array(temLogNumPerTW[0:traingSetSize])
	trainY=np.array(labelTWL[0:traingSetSize])

	if para['balance']:
		#fix imbalance problem
		print('imbalance processing..')
		failNum = np.count_nonzero(trainY)
		failIndex = np.nonzero(trainY)[0]
		failIndex=failIndex.tolist()
		failData = []
		for i in failIndex:
			li=trainX[i,:]
			failData.append(list(li))
		failDataArray=np.array(failData)

		succNum = len(trainY)-failNum
		print ('In training data, %d success tw, and %d failure time window'%(succNum, failNum))
		diffNum =int(np.floor(float(succNum )/ failNum)) 
		print diffNum
		if diffNum > 1:
			newGenArrayY=[1]*(failNum)
			for i in range(diffNum-1):
				trainX = np.vstack((trainX,failDataArray))
				trainY = np.hstack((trainY,newGenArrayY))	
			print trainX.shape,trainY.shape  

	# 0.4 0.0001 0.01  0.51370 0.51370 0.51370  
	# clf=svm.SVC(C=0.4, cache_size=2000, decision_function_shape='ovr', gamma=0.0001, kernel='rbf', class_weight='balanced' ,tol=0.01)   
	#clf=svm.SVC(C=1, cache_size=2000, decision_function_shape='ovr', gamma=0.0001, kernel='linear', class_weight='balanced' ,tol=0.01)   
	
	# #0.95402  0.56849 l1 0.0001 False
	clf=svm.LinearSVC(penalty='l1', tol=0.0001, C=1, dual=False, fit_intercept=True,
	 intercept_scaling=1, class_weight='balanced',  max_iter=1000)
	clf=clf.fit(trainX,trainY)
	

	testingX=temLogNumPerTW[traingSetSize:]
	testingY=labelTWL[traingSetSize:]
	# testingX = trainX
	# testingY = trainY 

	prediction=list(clf.predict(testingX))
	print np.count_nonzero(prediction)
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
		precision = 0
		recall = 0
		F_measure = 0 
	else:
		precision=float(sameFailureNum)/(predictFailure)
		print('precision is %.5f'%precision)
		recall=float(sameFailureNum)/(testFailure)
		print('recall is %.5f'%recall)
		F_measure=2*precision*recall/(precision+recall)
		print('F_measure is %.5f'%F_measure)
	print predictFailure,testFailure,sameFailureNum,precision,recall,F_measure
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def mainProcess(para):
	t1= time.time()
	rawEventCount,timeWindow,labels=processData(para)
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure=makePrediction(para,rawEventCount,timeWindow,labels)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure
	print('time is ',time.time()-t1)

# mainProcess(para)

def diffTime(para):
	timeList=[1,1,1,1,1]
	tiLen=len(timeList)
	result=np.zeros((tiLen,6))
	i=0
	for ti in timeList:
		para['timeInterval']=ti
		result[i,:]=mainProcess(para)
		i+=1
	print result
	np.savetxt('result_SVM_BGL_1h_6times.csv',result,delimiter=',')

diffTime(para)