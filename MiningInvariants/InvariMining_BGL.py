#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'Shilin He'
import time,datetime
import numpy as np
import csv
import math
import random
import fractions as fr
from itertools import combinations
import os


'''for BGL data with time window'''
#============Mining Invariants on BGL data===========#
#In this case, failed time windows are deleted		 #
#													 #
#=====================================================

para={
'path':'../Data/BGL_data/', 	#data path
'logfileName':'BGL_MERGED.log', #raw log data filename
'mapLogTemplate':'logTemplateMap.csv', # log event mapping relation list, obtained from log parsing
'selectColumn':[0,4],                       # select the corresponding columns in the raw data
'timeIndex':1,                              # the index of time in the selected columns, start from 0
'timeInterval':6,                        # the size of time window with unit of hours
'slidingWindow':1,						#sliding window size
'epsilon':3,                                # threshold for the step of estimating invariant space
'threshold': 0.98,                     		# how many percentage of vector Xj in matrix fit the condition that |Xj*Vi|<epsilon
'Llist':[1,2,3],							# list used to sacle the theta of float into integer
'filterFailure':True,						# filter the failure time window of data, default True
'stopOrNot':False,						# whether to set a constriant
'stopInvarNum':4                        # if try invariants with size more than 4, it will stop
}

#preprocess the data to count the event number occurred in each time window and filter some failed time windows
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
	csvFilePath = '../timeWindowCSV/slidingWindowIndex_'+str(para['timeInterval'])+'h_'+str(para['slidingWindow'])+'h.csv'
	twStartEndTuple=[]   #list of tuples, tuple contains two number, which represent the start and end of sliding time window, i.e. (1,5),(2,6),(3,7)
	if not os.path.exists(csvFilePath):
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
		np.savetxt(csvFilePath,twStartEndTuple,delimiter=',',fmt='%d')
	else:
		print('Loading time window')
		with open(csvFilePath) as timeWidowfi:
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

	#=================whether to filter the failed time window=============#
	if para['filterFailure']:
		failTWNum=np.count_nonzero(labels)
		print('failure time window number is %d'%(failTWNum))
		noFailEveCountMat=np.zeros((timeWindow-failTWNum,temNum))
		nofailCount=0
		for ti in range(timeWindow):
			if labels[ti]==0:
				for k in logIndexPerTWLL[ti]:
					tempIndex=logTemplateL[k]
					noFailEveCountMat[nofailCount,tempIndex]+=1
				nofailCount+=1
		print('In this step, we generate a matrix with size of %d*%d '%(noFailEveCountMat.shape[0],noFailEveCountMat.shape[1]))
		print('=====for validation usage only=====')
		print('the number of raw logs is %d, and the number of filtered logs is %d.'%(sum((sum(rawEventCount))),sum((sum(noFailEveCountMat)))))
		return noFailEveCountMat,rawEventCount,labels
	else:
		return rawEventCount,rawEventCount,labels


#Estimate the Invariant Space using SVD decomposition, return the space size r(integer)
#M: represent the matrix of message count vectors with size N*m, N is the instance(timeWindow) number and m is the feature(event type) number
def estiInvaSpace(M,para):
	t1=time.time()
	M_trans_M=np.dot(M.T,M)/float(M.shape[0])
	# SVD decomposition using the numpy.linalg.svd
	u,s,v=np.linalg.svd(M_trans_M)
	print("SVD decomposition results(U,S,V) size: ", u.shape,s.shape,v.shape)
	#the sigular value are in descending order, so we start from the right most column in matrix v
	#check whether more than threshold rows in M multiplies the above column smaller than the epsilon
	feaNum=v.shape[1]
	instaNum=M.shape[0]
	for i in range(feaNum-1,-1,-1):
		colVec=v[i,:].T
		count=0
		for j in range(instaNum):
			abso=abs(np.dot(M[j,:],colVec))
			if abso<para['epsilon']:
				count+=1
		if count<instaNum*para['threshold']:
			print('from the %d(not included) column to the rightmost column, they are validated invariant!'%i)
			break
	r=feaNum-i-1
	print('so the dimension of invariant space is %d'%r)
	return r

#given a selected-column matrix, use svd to compute its least eigenValue and corrsponding minimum eigenVector
#return the minmum eigenVector and whether this vector contains zero
def calMinTheta(M):
	vecContainZero=False
	zeroCount=0
	Mtrans_T=np.dot(M.T,M)
	u,s,v=np.linalg.svd(Mtrans_T)
	eigenValue=s[-1]
	minVec=v[-1,:]
	for i in minVec:
		if np.fabs(i) < (1e-3):
			zeroCount+=1
	if zeroCount!=0:
		print("0 exits and discard the following vector: ")
		vecContainZero=True
	print(minVec)
	return minVec.T,vecContainZero

#scale the returned eigenVector of float into integer and check whether the scaled theta is valid(satisfy the threshold)
#if valid, return True and the valid eigenVector, if not valid, return False and empty eigenVector
def checkInvariValid(para,M,pList):
	newMatrix=M[:,pList]
	instaNum=M.shape[0]
	valid=False
	print('selected matrix columns are',pList)
	minTheta,vecContainZero=calMinTheta(newMatrix)
	absminTheta=[np.fabs(it) for it in minTheta]
	if vecContainZero==True:
		return valid,[]
	else:
		for i in para['Llist']:
			minIndex=np.argmin(absminTheta)
			minValue=minTheta[minIndex]
			scale=float(i)/minValue
			scaleTheta=np.array([round(item*scale) for item in minTheta])
			scaleTheta[minIndex]=i
			scaleTheta=scaleTheta.T

			newZeroCount=0
			for sca in scaleTheta:
				if np.fabs(sca) == 0:
					newZeroCount+=1
			if newZeroCount!=0:
				continue
			Xj_theta=np.dot(newMatrix,scaleTheta)

			zeroCount=0
			for i in Xj_theta:
				if np.fabs(i) < (1e-8):
					zeroCount+=1
			if zeroCount>=para['threshold']* instaNum:
				valid=True
				print('======A valid invariants and corresponding columns============',scaleTheta,pList)
				break
		return valid,scaleTheta

#find incariants
searchSpace=[]     # only invariant candidates in this list are valid.
def invariantSearch(dataMatrix, para, r):
	''' Apriori framework to generate invariants candidates'''
	# initial set: 0 ~ num_features
	(num_samples, num_features) = dataMatrix.shape
	invariantDict = dict()	#save the mined Invariants(value) and its corrsponding columns(key)
	itemList = sorted([[item] for item in xrange(num_features)])

	for item in itemList:
		searchSpace.append(item)
	newItemList=itemList[:]
	for item in itemList:
		#if (np.sum(dataMatrix[:, item] == 0)/float(num_samples))>= para['threshold']:
		if np.sum(dataMatrix[:, item] == 0)== num_samples:
			invariantDict[frozenset(item)] = [1]
			searchSpace.remove(item)
			newItemList.remove(item)
	print('the remaining features are: ',newItemList)

	itemList=newItemList
	iniLen=len(invariantDict)
	length = 2
	breakLoop=False

	while(len(itemList) != 0):
		if para['stopOrNot']:
			if len(itemList[0]) >= para['stopInvarNum']:
				break
		_itemList = joinSet(itemList, length)    # generate new invariant candidates
		for items in _itemList:
			if validCandidate(items, length, searchSpace):
				searchSpace.append(items)
		#print searchSpace
		itemList=[]
		for item in _itemList:
			if frozenset(item) in invariantDict.keys():
				continue
			if item not in searchSpace:
				continue
			if  not validCandidate(frozenset(item), length, searchSpace) and length > 2:
				searchSpace.remove(item)
				continue 	#an item must be superset of all other subitems in searchSpace, else skip

			valid,scaleTheta=checkInvariValid(para,dataMatrix,item)
			if valid==True:
				prune(invariantDict.keys(),set(item),searchSpace)
				invariantDict[frozenset(item)] = scaleTheta
				searchSpace.remove(item)
			else:
				itemList.append(item)
			if len(invariantDict)>=r:
				breakLoop=True
				break
		if breakLoop==True:
			break
		length += 1
	return invariantDict

#add a newComingSet, and generate its possible combination of it and existed colList
def prune(colSetList,newComingSet,searchSpace):
	if len(colSetList)==0:
		return
	for se in colSetList:
		intersection=set(se) & newComingSet
		if len(intersection)==0:
			continue
		union = set(se) | newComingSet
		for item in list(intersection):
			diff=sorted(list(union - set([item])))
			if diff in searchSpace:
				searchSpace.remove(diff)

#generate new items with size of length
def joinSet(itemList, length):
	"""Join a set with itself and returns the n-element itemsets"""
	setLen=len(itemList)
	returnList=[]
	for i in range(setLen):
		for j in range(i+1,setLen):
			iSet=set(itemList[i])
			jSet=set(itemList[j])
			if len(iSet.union(jSet)) == length:
				joined=sorted(list(iSet.union(jSet)))
				if joined not in returnList:
					returnList.append(joined)
	return sorted(returnList)

#check whether an item's subitems are in searchspace
def validCandidate(item, length, searchSpace):
	for subItem in combinations(item, length - 1):
		if sorted(list(subItem)) not in searchSpace:
			return False
	return True

#calculate the precision and recall
def testing(selecRawData,validInvariList,validColList,realLabels):
	predictLabel=[]
	count=0
	for row in selecRawData:
		label=0
		for i,cols in enumerate(validColList):
			sumOfInva=0
			for j, c in enumerate(cols):
				sumOfInva += validInvariList[i][j]*row[c]
			if sumOfInva != 0:
				label=1         # same as raw data that 1 represent failure
				break
		predictLabel.append(label)
	sameFailureNum=0
	for i in range(len(predictLabel)):
		if(realLabels[i]==1):
			if(predictLabel[i]==1):
				sameFailureNum+=1
	predictSuccess=0
	predictFailure=0
	for item in predictLabel:
		if item==0:
			predictSuccess+=1
		elif item==1:
			predictFailure+=1
	testSuccess=0
	testFailure=0
	for tt in realLabels:
		if tt==0:
			testSuccess+=1
		elif tt==1:
			testFailure+=1
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
	noFailData,rawData,labels=processData(para)
	t1=time.time()
	r=estiInvaSpace(noFailData,para)
	invariantDict=invariantSearch(noFailData, para, r)
	print(invariantDict,len(invariantDict))
	validColList=[]
	validInvariList=[]
	for key in invariantDict:
		validColList.append(list(key))
		validInvariList.append(list(invariantDict[key]))
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure = testing(rawData,validInvariList,validColList,labels)
	t= time.time() - t1
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure,t

if __name__ == '__main__':
	mainProcess(para)
