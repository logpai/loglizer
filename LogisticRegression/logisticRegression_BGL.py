#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'
import time,datetime
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression   
import pydot
import math
import os

'''for BGL data with time window using Decision Tree'''
#============decision Tree on BGL data===============#
#In this case, failed time windows are deleted		 #
#													 #
#=====================================================

para={
'path':'../Data/BGL_data/',
'logfileName':'BGL_MERGED.log',
'mapLogTemplate':'logTemplateMap.csv',
'selectColumn':[0,4],	 	# select the corresponding columns in the raw data, no need to change
'timeIndex':1,			 	# the index of time in the selected columns, start from 0
'timeInterval':1,			#the size of time window with unit of hours
'trainingSetPercent': 0.8, 	#which means how many(percentage) time windows would be regarded as training set
'tf-idf': False
}


def processData(para):
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

	#=================divide into time windows=============#
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

	#=================loading data of log-event relation=============#
	logIndexPerTWLL=[]
	for t in range(timeWindow):
		newTW=[]
		newfilteredLogTWL=[]
		logIndexPerTWLL.append(newTW)
	for i in range(len(timeWindowIndex)):
		tw=timeWindowIndex[i]
		logIndexPerTWLL[tw-1].append(i) #time window start from 0, in which log index start from 0

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
		label='success'   #0 represent success, 1 represent failure
		for k in logIndexPerTWLL[j]:
			tempIndex=logTemplateL[k]
			rawEventCount[j,tempIndex]+=1
			if dataLL[k][0]!='-':
				label='failure'
				continue
		labels.append(label)

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
	print('rawData not idf data')
	return rawEventCount,timeWindow,labels

def makePrediction(para,temLogNumPerTW,timeWindow,labelTWL):
	traingSetSize=int(math.floor(timeWindow*para['trainingSetPercent']))
	print('%d timewindows are treated as training dataset!'%traingSetSize)
	trainX=temLogNumPerTW[0:traingSetSize]
	trainY=labelTWL[0:traingSetSize]
	#100 l1 0.001  0.92000 0.65714
	#1000
	#1 0.001
	clf=LogisticRegression(C=1, penalty='l1', tol=0.0001,class_weight='balanced',multi_class='ovr')
	clf=clf.fit(trainX,trainY)

	testingX=temLogNumPerTW[traingSetSize:]
	testingY=labelTWL[traingSetSize:]
	# testingX =trainX
	# testingY = trainY
	prediction=list(clf.predict(testingX))
	
	if len(prediction)!=len(testingY):
		print ('prediction and testingY have different length and SOMEWHERE WRONG!')
	sameLabelNum=0
	sameFailureNum=0
	for i in range(len(testingY)):
		if prediction[i]==testingY[i]:
			sameLabelNum+=1
			if prediction[i]=='failure':
				sameFailureNum+=1

	accuracy=float(sameLabelNum)/len(testingY)
	print ('accuracy is %.5f:'%accuracy)

	predictSuccess=0
	predictFailure=0
	for item in prediction:
		if item=='success':
			predictSuccess+=1
		elif item=='failure':
			predictFailure+=1

	testSuccess=0
	testFailure=0
	for tt in testingY:
		if tt=='success':
			testSuccess+=1
		elif tt=='failure':
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
	print predictFailure,testFailure,sameFailureNum,precision,recall,F_measure
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def mainProcess(para):
	rawEventCount,timeWindow,labels = processData(para)
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure=makePrediction(para,rawEventCount,timeWindow,labels)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def diffTime(para):
	timeList=[1,1,1,1,1,1]
	tiLen=len(timeList)
	result=np.zeros((tiLen,6))
	i=0
	for ti in timeList:
		para['timeInterval']=ti
		result[i,:]=mainProcess(para)
		i+=1
	print result
	np.savetxt('result_LogisticRegress_BGL_1h_6times.csv',result,delimiter=',')

diffTime(para)
# mainProcess(para)