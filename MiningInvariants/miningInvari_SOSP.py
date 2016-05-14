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

'''for SOSP data'''
#============Mining Invariants on SOSP data==========#
#													 #
#													 #
#=====================================================

para={
'path':'../Data/SOSP_data/',
'matrixFile':'rm_repeat_rawTFVector.txt',   
'labelFile':'rm_repeat_mlabel.txt',             
'epsilon':2.0,                          # threshold for the step of estimating invariant space
'threshold':0.98,                       # how many percentage of vector Xj in matrix fit the condition that |Xj*Vi|<epsilon
'Llist':[1,2,3],						# list used to sacle the theta of float into integer
'filterFailure':True,					# filter the failure time window of data, default True
'stopOrNot':True,						# whether to set a constriant
'stopInvarNum':4                        # if try invariants with size more than 4, it will stop
}

def processData(para):
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
			# #newly added for pinjia
			# va = float(line[j])
			# if va < 1e-8:
			# 	va = 0
			# else:
			# 	va= int(va)
			# rawData[numLines,j]=va
			# #above are newly added
			rawData[numLines,j]=int(line[j])
		numLines += 1
	sucLabel=[i for i in range(totalNumRows) if labels[i]==0]
	noFailData=rawData[sucLabel,:]
	print ('selected lines is %d and total line number is %d'%(noFailData.shape[0],numLines))
	np.savetxt('noFailData.txt',noFailData,fmt='%d')
	f.close()
	flabel.close()
	return noFailData,rawData,labels


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
	print ('selected matrix columns are',pList)
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
	print ('the remaining features are: ',newItemList)

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
	print selecRawData.shape
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
	print len(realLabels),len(predictLabel)

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
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def mainProcess(para):
	noFailData,rawData,labels=processData(para)
	r=estiInvaSpace(noFailData,para)
	invariantDict=invariantSearch(noFailData, para, r)
	print invariantDict,len(invariantDict)
	validColList=[]
	validInvariList=[]
	for key in invariantDict:
		validColList.append(list(key))
		validInvariList.append(list(invariantDict[key]))
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure = testing(rawData,validInvariList,validColList,labels)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure	

# mainProcess(para)
def diffThreshold(para):
	tiLen=len(threshold)
	result=np.zeros((tiLen,6))
	i=0
	for ti in threshold:
		result[i,:]=mainProcess(para)
		i+=1
	print result
	np.savetxt('result_miningInvari_SOSP_5_times.csv',result,delimiter=',')

diffThreshold(para)

#0.98 0.99
# {frozenset([7]): [1], frozenset([0]): [1], frozenset([6]): [1], frozenset([28]): [1], frozenset([13]): [1], frozenset([18]): [1],
# frozenset([11]): [1], frozenset([14]): [1], frozenset([23]): [1], frozenset([16]): [1], frozenset([12]): [1], frozenset([9]): [1]
# frozenset([8, 4]): array([ 1., -1.]),  frozenset([4, 21]): array([ 1., -3.]), frozenset([17, 5]): array([ 1., -1.]), 
# frozenset([24, 5]): array([ 1., -1.]),  frozenset([20, 22]): array([ 1., -1.]), frozenset([25, 4]): array([-1.,  1.]),
# frozenset([10, 4]): array([ 1., -1.]), frozenset([5, 15]): array([ 1., -1.]), frozenset([19, 5]): array([   1., -467.]),} 
# 21

#0.999
#{frozenset([24, 5]): array([ 1., -1.]), frozenset([10, 4, 15]): array([ 1., -1., -1.]), frozenset([8, 4, 5]): array([ 1., -1., -1.]), frozenset([16]): [1], frozenset([9]): [1], frozenset([7]): [1], frozenset([0]): [1], frozenset([17, 10, 4]): array([ 1., -1., -1.]), frozenset([5, 15]): array([ 1., -1.]), frozenset([28]): [1], frozenset([17, 5]): array([ 1., -1.]), frozenset([24, 10, 4]): array([ 1., -1., -1.]), frozenset([13]): [1], frozenset([11]): [1], frozenset([25, 10, 4]): array([-2.,  1.,  1.]), frozenset([14]): [1], frozenset([23]): [1], frozenset([8, 10]): array([ 1., -1.]), frozenset([8, 21]): array([ 1., -3.]), frozenset([6]): [1], frozenset([18]): [1], frozenset([25, 4, 5]): array([ 1.,  1., -1.]), frozenset([12]): [1]} 
#23