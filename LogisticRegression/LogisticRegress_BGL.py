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

para={
'path':'../Data/BGL_data/', #data path
'logfileName':'BGL_MERGED.log', #raw log data filename
'mapLogTemplate':'logTemplateMap.csv', # log event mapping relation list, obtained from log parsing
'selectColumn':[0,4], # select the corresponding columns in the raw data
'timeIndex':1,  # the index of time in the selected columns, start from 0
'timeInterval':6,  #the size of time window with unit of hour
'slidingWindow':1,  #the size of sliding window interval with unit of hour
'trainingSetPercent': 0.8, # 80% of the time windows are used for training
'tf-idf':False,	# whether turn on the tf-idf
'balance':False	# use balance mechanism during model building to solve imbalance problem
}


def processData(para):
	print('sliding window for BGL LogisticRegression')
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
	twStartEndTuple=[]   #list of tuples, tuple contains two number, which represent the start and end of sliding time window, i.e. (1,5),(2,6),(3,7)
	if not os.path.exists('../timeWindowCSV/slidingWindowIndex_'+str(para['timeInterval'])+'h_'+str(para['slidingWindow'])+'h.csv'):

		startTime=time.mktime(datetime.datetime.strptime(dataLL[0][para['timeIndex']], "%Y-%m-%d-%H.%M.%S.%f").timetuple())
		timeWindow=1
		t=startTime
		firstStartEndLog=tuple()
		startLog=0
		endLog=0
		for row in dataLL:
			timeStamp=time.mktime(datetime.datetime.strptime(row[para['timeIndex']], "%Y-%m-%d-%H.%M.%S.%f").timetuple())
			if timeStamp>=t and timeStamp<t+para['timeInterval']*3600:
				endLog+=1
				endTime=timeStamp
			else:
				startEndLog=tuple((startLog,endLog))
				twStartEndTuple.append(startEndLog)
				break
		start=0
		end=endLog
		while end<logNum:
			startTime=startTime+para['slidingWindow']*3600
			endTime=endTime+para['slidingWindow']*3600
			for i in range(start,end):
				timeStamp=time.mktime(datetime.datetime.strptime(dataLL[i][para['timeIndex']], "%Y-%m-%d-%H.%M.%S.%f").timetuple())
				if timeStamp<startTime:
					i+=1
				else:
					break
			for j in range(end,logNum):
				timeStamp=time.mktime(datetime.datetime.strptime(dataLL[j][para['timeIndex']], "%Y-%m-%d-%H.%M.%S.%f").timetuple())
				if timeStamp<endTime:
					j+=1
				else:
					break
			start=i
			end=j
			startEndLog=list((start,end))
			twStartEndTuple.append(startEndLog)
		print(twStartEndTuple[-1])
		timeWindow=len(twStartEndTuple)
		print('there are %d time windows in this dataset!\n'%timeWindow)
		print('%d instances in this dataset'%len(twStartEndTuple))
		np.savetxt('../timeWindowCSV/slidingWindowIndex_'+str(para['timeInterval'])+'h_'+str(para['slidingWindow'])+'h.csv',twStartEndTuple,delimiter=',',fmt='%d')
	else:
		print('Loading time window')
		with open('../timeWindowCSV/slidingWindowIndex_'+str(para['timeInterval'])+'h_'+str(para['slidingWindow'])+'h.csv') as timeWidowfi:
			twreader = csv.reader(timeWidowfi)
			for logrow in twreader:
				row=[int(logrow[0]),int(logrow[1])]
				twStartEndTuple.append(row)
		timeWindow=len(twStartEndTuple)
		print('there are %d time windows in this dataset!\n'%timeWindow)

	logIndexPerTWLL=[]
	for t in range(timeWindow):
		newTW=[]
		logIndexPerTWLL.append(newTW)
	for i in range(timeWindow):
		startLog=twStartEndTuple[i][0]
		endLog=twStartEndTuple[i][1]
		for l in range(startLog,endLog):
			logIndexPerTWLL[i].append(l) #logIndexPerTWLL, time window start from 0, in which log index start from 0

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
		print('In training data, %d success tw, and %d failure time window'%(succNum, failNum))
		diffNum =int(np.floor(float(succNum )/ failNum))
		print diffNum
		if diffNum > 1:
			newGenArrayY=[1]*(failNum)
			for i in range(diffNum-1):
				trainX = np.vstack((trainX,failDataArray))
				trainY = np.hstack((trainY,newGenArrayY))
			print trainX.shape,trainY.shape

	clf=LogisticRegression(C=0.01, penalty='l1', tol=0.01,class_weight='balanced',multi_class='ovr')
	clf=clf.fit(trainX,trainY)
	testingX=temLogNumPerTW[traingSetSize:]
	testingY=labelTWL[traingSetSize:]
	prediction=list(clf.predict(testingX))

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

	print(predictSuccess,predictFailure,testSuccess,testFailure,sameFailureNum)
	if sameFailureNum==0:
		print('precision is 0 and recall is 0')
	else:
		precision=float(sameFailureNum)/(predictFailure)
		print('precision is %.5f'%precision)
		recall=float(sameFailureNum)/(testFailure)
		print('recall is %.5f'%recall)
		F_measure=2*precision*recall/(precision+recall)
		print('F_measure is %.5f'%F_measure)
	print(predictFailure,testFailure,sameFailureNum,precision,recall,F_measure)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def mainProcess(para):
	rawEventCount,timeWindow,labels = processData(para)
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure=makePrediction(para,rawEventCount,timeWindow,labels)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

if __name__ == '__main__':
	mainProcess(para)
