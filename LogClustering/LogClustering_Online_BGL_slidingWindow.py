#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy.special import expit
from numpy import linalg as LA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster, fclusterdata
from scipy.spatial.distance import pdist
import time
import os
import csv
'''for SOSP data using Log Clustering'''
#============Log Clustering on SOSP data=============#
#													 #
#													 #
#=====================================================
para={
'path':'../Data/BGL_data/',
'logfileName':'BGL_MERGED.log',  			# newtestData_25w   BGL_MERGED
'mapLogTemplate':'logTemplateMap.csv',  	# data that restore the template index of each log 
'selectColumn':[0,4],                       # select the corresponding columns in the raw data
'timeIndex':1,                              # the index of time in the selected columns, start from 0
'timeInterval':6,                        # the size of time window with unit of hours
'slidingWindow':1,  #the size of sliding window interval with unit of hour
'trainDataPercent':0.8,
'max_d': 1,  # the threshold for cutoff the cluster process
'repre_threshold':0.6,
'fail_threshold':0.6,   
'succ_threshold':0.6,
'initialTrainPercent': 0.8#percentage of training data that be used as initial training
}
#0.9 0.3 0.599 1            0.36449 0.79592 0.500  
#0.9 0.3 0.8 1 				0.35849 0.90476 0.51351
#0.9 0.3 0.7 1              0.36510 0.84694 0.51025
#0.9 0.3 0.1 1  			0.13521 0.16326 0.14791
#0.9 0.3 0.5 1				0.3557  0.738095 0.480088
#0.9 0.6 0.5 1				0.27134 0.49320
#0.9 0.3 0.5 0.7 			0.38897 0.86395 0.53643
#0.9 0.3 0.5 0.4 			0.34090 0.96938 0.50442
#0.9 0.3 0.5 0.1			0.288102261554 0.996598639456 0.446987032799
#0.9 0.3 0.8 0.7 			0.371391076115 0.962585034014 0.535984848485

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
	print('calculating the number of templates altogether')
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
	return rawEventCount,labels,timeWindow

def computeWeight(rawData):
	# to avoid the case that a term never occurs in a document, we add 1 to the cnt 
	numLines,numEvents=rawData.shape
	weightedData=np.zeros((numLines,numEvents),float)
	for j in range(numEvents):
		cnt = np.count_nonzero(rawData[:,j])
		for i in range(numLines):
			weight = 0.5 * expit(math.log(numLines/float(cnt)))
			weightedData[i,j] = rawData[i,j] * weight
	print('weighted data size is',weightedData.shape)
	return weightedData

def seperateData(para,weightedData,totalNumLines,labels):
	'''data can be divided into three part, initial training data(from start to initialTrainSize), total learning 
	   data(from initialTrainSize to trainSize), and testing data(from trainSize to end) three part are seperated by two number 
	   initialTrainSize and trainSize'''

	print('seperating the initial training Data...')
	trainSize = int(math.floor(totalNumLines*para['trainDataPercent']))
	initialTrainSize = int(math.floor(trainSize*para['initialTrainPercent']))

	initTrainData = weightedData[:initialTrainSize,:]
	
	totalLearnData = weightedData[:trainSize,:]
	testingData = weightedData[trainSize:,:]
	print ('initial training data size is',initTrainData.shape)
	print ('total learning datasize is',totalLearnData.shape)
	print ('testing datasize is',testingData.shape)

	initTrainLabel = labels[:initialTrainSize]
	totalLearnLabels = list(labels[:trainSize])
	testingLabels = list(labels[trainSize:])
	print len(initTrainLabel),len(totalLearnLabels),len(testingLabels)
	
	failSeqList = []
	succSeqList = []
	for i in range(len(initTrainLabel)):
		if initTrainLabel[i] == 0:
			succSeqList.append(i)
		else:
			failSeqList.append(i)

	return succSeqList,failSeqList,initTrainData,totalLearnData,totalLearnLabels,testingData,testingLabels,initialTrainSize,trainSize

def clustering(partSeqList, partData):
	print('clustering for the seperated dataset')
	#simiMatrix = simiMatrixCal(partData)
	'''Invoke the clustering method in library'''
	#print simiMatrix.shape

	data_dist = pdist(partData,metric=distCalculate)

	Z = linkage(data_dist, 'complete')
	clusterLabels = fcluster(Z, para['max_d'], criterion='distance')
	print ('there are altogether %d clusters in this initial clustering'%(len(np.unique(clusterLabels))))
	# print Z[:20]
	# plt.figure(figsize=(25, 10))
	# plt.title('Hierarchical Clustering Dendrogram')
	# plt.xlabel('sample index')
	# plt.ylabel('distance')
	# dendrogram(Z,                                                               
	#     leaf_rotation=90.,  # rotates the x axis labels
	#     leaf_font_size=8.,  # font size for the x axis labels
	#     )
	# plt.show()
	
	clusNum = len(set(clusterLabels))
	instIndexPerClus=[[] for i in range(clusNum)]  #initialization
	for i in range(len(clusterLabels)):
		lab = clusterLabels[i]-1
		instIndexPerClus[lab].append(partSeqList[i])
	return clusterLabels,instIndexPerClus

def findInitRepre(initTrainData,instIndexPerClus,scoreList):
	print('find initial representive')
	clusNum = len(instIndexPerClus)
	represents = []
	for eachClusterIndex in instIndexPerClus:
		eachCluster = initTrainData[eachClusterIndex,:]
		eachClusLen = len(eachCluster)
		distMatrix = distMatrixCal(np.array(eachCluster))
		minScore = float('inf')  #infinity
		minIndex = -1
		for cl in range(eachClusLen):
			index = eachClusterIndex[cl]   #which row in the rawdata
			score = np.sum(distMatrix[cl]) - distMatrix[cl,cl] 
			scoreList[index] = score
			#print score, index
			if score < minScore:
				minIndex = index
				minScore = score
		represents.append(initTrainData[minIndex])
		#print ('min Index is %d and min value is %f'%(minIndex,minScore))
	return represents

def anomalyDetect(para, failSeqList,succSeqList,initTrainData,totalLearnData,totalLearnLabels,testingData,testingLabels,initialTrainSize,trainSize,totalNumLines):
	failData = initTrainData[failSeqList,:]
	succData = initTrainData[succSeqList,:]
	print failData.shape,succData.shape
	if failData.shape[0] + succData.shape[0] != len(initTrainData):
		print('failure occurs while splitting success data and failure data in traing data')
	failClusters, failIndexPerClus = clustering(failSeqList, failData)
	print('failure data clustering finished...')
	succClusters, succIndexPerClus = clustering(succSeqList, succData)
	print('success data clustering finished...')

	scoreList=np.zeros(totalNumLines)  # A one dimension list of all zero with size of totalLineNum
	failRepres = findInitRepre(initTrainData, failIndexPerClus,scoreList)
	succRepres = findInitRepre(initTrainData, succIndexPerClus,scoreList)
	
	onlineLearn(para, totalLearnData,totalLearnLabels,scoreList,failRepres,succRepres,failIndexPerClus,succIndexPerClus,initialTrainSize)
	print len(failRepres), len(succRepres)


	#!!!!!!!!!!+=============================!!!!!
	# succFailClusterDist = np.zeros((len(failRepres), len(succRepres)))
	# for i,fa in enumerate(failRepres):
	# 	for j,su in enumerate(succRepres):
	# 		succFailClusterDist[i,j] = distCalculate(fa,su)
	# np.savetxt('succFailClusterDist.csv',succFailClusterDist,delimiter=',')


	predictLabel = checkCluster(failRepres,succRepres,testingData)
	return testing(predictLabel,testingLabels)


def onlineLearn(para,totalLearnData,totalLearnLabels,scoreList,failRepresents,succRepresents,failIndexPerClus,succIndexPerClus,initialTrainSize):
	totalTrainSize = totalLearnData.shape[0]
	threshold = para['repre_threshold']
	for i in range(initialTrainSize, totalTrainSize):  
		newInst = totalLearnData[i]
		if totalLearnLabels[i] == 1:   #failure data cluster
			minValue,minIndex = calDis(newInst, failRepresents)   #represent is the represent vector of each cluster 
			if minValue <= threshold:
				clusLabelIndex = minIndex   #index of cluster label, means that if it is 0 then cluster should be the first cluster one
				print('instance %d(failure) ------> clustered into failure label %d'%(i,clusLabelIndex+1))
				updateRepre(i, newInst, failIndexPerClus,totalLearnData,failRepresents, scoreList, clusLabelIndex)
			else:
				print('instance %d(failure) ------> should create new failure cluster%d'%(i,len(failRepresents)+1))
				failIndexPerClus.append([i])  	#scoreList[logIndex] still should be zero and do not need to be updated as 
				failRepresents.append(newInst)  #currently there is no other logs in this cluster
		else:
			minValue,minIndex  = calDis(newInst, succRepresents)
			if minValue <= threshold:
				clusLabelIndex = minIndex
				print('instance %d(success) ------> clustered into success label %d'%(i,clusLabelIndex+1))
				updateRepre(i, newInst, succIndexPerClus, totalLearnData,succRepresents,scoreList, clusLabelIndex)
			else:
				print('instance %d(success) ------> should create new success cluster %d'%(i,len(succRepresents)+1)) 
				succIndexPerClus.append([i])
				succRepresents.append(newInst) 
	# print 'finally, the resulted failure clusters is',failIndexPerClus
	# print 'finally, the resulted success clusters is', succIndexPerClus

def updateRepre(logIndex, newInst, indexPerClus, totalLearnData,represent, scoreList, clusLabelIndex):
	#update score of existing logs in current cluster 
	i=0
	updateDis=[]
	logIndexInCLuster = indexPerClus[clusLabelIndex]
	for ind in logIndexInCLuster:
		newDis = distCalculate(newInst, totalLearnData[ind])
		scoreList[ind] += newDis
		updateDis.append(newDis)
		
	# add current log index into the current cluster
	indexPerClus[clusLabelIndex].append(logIndex)

	# update newInstance data itself
	if scoreList[logIndex] ==0 :
		scoreList[logIndex] += sum(updateDis)
	else:
		print('ERROR!!!!+========++++++++++++++++++++++====++++++++++++')

	#if this row is the same as the representive vector,then there is no need to find the new representive as they must be the same
	# #choose the minimum value as the representive vector
	if not np.allclose(newInst,represent[clusLabelIndex]):  
		partScoreList=scoreList[logIndexInCLuster]
		minIndex = logIndexInCLuster[np.argmin(partScoreList)]
		represent[clusLabelIndex] = totalLearnData[minIndex]
		

def calDis(newInst, represents):
	minIndex = -1
	minValue = float('inf')
	for i,re in enumerate(represents):
		if np.allclose(newInst,re): 
			minIndex = i
			minValue = 0
			break
		newDis = distCalculate(newInst, re)
		if newDis < minValue:
			minValue = newDis
			minIndex = i
	return minValue,minIndex

def distMatrixCal(partData):
	numRows,numFea=partData.shape
	distMatrix=-1*np.ones((numRows,numRows))
	for i in range(numRows):
		for j in range(i,numRows):
			distMatrix[i,j] = distMatrix[j,i] = distCalculate(partData[i,:],partData[j,:])	
	return distMatrix

def distCalculate(Sei,Sej):
	#in our case, distCalculate must be between 0 ~ 1, where 1 represents totally the same.
	weigh=float(LA.norm(Sei)*LA.norm(Sej))
	if weigh ==0:
		return 1
	result = 1 - np.dot(Sei,Sej)/weigh
	if abs(result) < 1e-8:
		result = 0
	return result

def testing(predictLabel,realLabels):
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
		predictFailure=testFailure=0
		sameFailureNum=precision=0
		recall=F_measure=0
		return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure
	else:
		precision=float(sameFailureNum)/(predictFailure)
		print('precision is %.5f'%precision)
		recall=float(sameFailureNum)/(testFailure)
		print('recall is %.5f'%recall)
		F_measure=2*precision*recall/(precision+recall)
		print('F_measure is %.5f'%F_measure)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

def checkCluster(failRepres,succRepres,testingData):
	predictLabel=[-1]*testingData.shape[0]
	# clusterLabel=[0]*testingData.shape[0]
	clusterNum=len(failRepres)
	for r, row in enumerate(testingData):
		newDisList = []
		for failRep in failRepres:
			newDisList.append(distCalculate(failRep,row))  
		#print('failure distance',newDisList)
		#print newDisList
		if min(newDisList) < para['fail_threshold']:
			predictLabel[r] = 1 
		else: 
			succDist=[]
			for succRep in succRepres:
				succDist.append(distCalculate(succRep,row))  
			#print ('success distance',succDist)
			if min(succDist) < para['succ_threshold']:
				predictLabel[r] = 0#(np.argmax(succDist)+1)
			else:
				predictLabel[r] = 1
	#print predictLabel
	return predictLabel


def mainProcess(para):
	t1 = time.time()
	rawData,labels,totalNumLines = processData(para)
	weightedData = computeWeight(rawData)
	succSeqList,failSeqList,initTrainData,totalLearnData,totalLearnLabels,testingData,testingLabels,initialTrainSize,trainSize = seperateData(para,weightedData,totalNumLines,labels)
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure= anomalyDetect(para, failSeqList,succSeqList,initTrainData,totalLearnData,totalLearnLabels,testingData,testingLabels,initialTrainSize,trainSize,totalNumLines)
	print predictFailure,testFailure,sameFailureNum,precision,recall,F_measure
	print(time.time() - t1)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

# if __name__ == '__main__':
# 	mainProcess(para)

# def chooseParameters(para):
# 	maxd=[1]    #0.8 0.9 highest precision and recall(always the same in this dataset 0.012)
# 	repre_threshold=[0.6]
# 	fail_threshold=[0.6]  #0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
# 	succ_threshold=[ 0.6]  # 0.1,0.2,0.3,0.4,0.5,0.6,0 .7,0.8,0.9,1.0

# 	cc=0
# 	result=np.zeros((500,10))  #10000
# 	for i in range(len(maxd)):
# 		para['max_d']=maxd[i]
# 		for j in range(len(repre_threshold)):
# 			para['repre_threshold']=repre_threshold[j]	
# 			for k in range(len(fail_threshold)):
# 				para['fail_threshold'] = fail_threshold[k]	
# 				for t in range(len(succ_threshold)):
# 					result[cc,0] =maxd[i]
# 					result[cc,1] =repre_threshold[j]
# 					result[cc,2] =fail_threshold[k]
# 					para['succ_threshold'] = succ_threshold[t]
# 					print maxd[i]
# 					print repre_threshold[j]
# 					print fail_threshold[k]
# 					print succ_threshold[t]
# 					result[cc,3] =succ_threshold[t]
# 					result[cc,4:10] = mainProcess(para)
# 					cc+=1
# 	print result
# 	np.savetxt('Newresults.csv',result,delimiter=',')
# chooseParameters(para)

# def diffTimes(para):
# 	times = [1,1,1,1,1]
# 	tiLen=len(times)
# 	result=np.zeros((tiLen,6))
# 	i=0
# 	for ti in times:
# 		result[i,:]=mainProcess(para)
# 		i+=1
# 	print result
# 	np.savetxt('result_logCustering_BGL_5_times.csv',result,delimiter=',')

# diffTimes(para)


def diffTimes(para):
	#timeList=[1,3,6,9,12]
	timeList=[ 0.0833333333]#, 0.083333 not suitable  0.0833333333] #
	tiLen=len(timeList)
	result=np.zeros((tiLen,6))
	i=0
	for ti in timeList:
		para['slidingWindow']=ti
		result[i,:]=mainProcess(para)
		i+=1
		print result
	
	np.savetxt('Tune_slidingWindow_LogClustering_BGL_keep6h_ChangeslidingSize.csv',result,delimiter=',')

diffTimes(para)
