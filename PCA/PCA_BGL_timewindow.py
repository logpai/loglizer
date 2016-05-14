#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import math
import time,datetime
import csv
import os
'''for BGL data with time window using BGL'''
#================PCA on BGL data=====================#
#In this case, failed time windows are deleted		 #
#													 #
#=====================================================

para={
'path':'../Data/BGL_data/',
'logfileName':'BGL_MERGED.log',  			# newtestData_25w   BGL_MERGED
'mapLogTemplate':'logTemplateMap.csv',  	# data that restore the template index of each log 
'selectColumn':[0,4],                       # select the corresponding columns in the raw data
'timeIndex':1,                              # the index of time in the selected columns, start from 0
'timeInterval':6,                        	# the size of time window with unit of hours
'fraction':0.95,
'c_alpha': 5.3267
}

def processData(para):	
	print('for BGL data using PCA')
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
		savetxt('../timeWindowCSV/timeWindowIndex_'+str(para['timeInterval'])+'h.csv',timeWindowIndex,delimiter=',',fmt='%d')
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
	rawEventCount=zeros((timeWindow,temNum))
	for j in range(timeWindow):
		label=0   #0 represent success, 1 represent failure
		for k in logIndexPerTWLL[j]:
			tempIndex=logTemplateL[k]
	  		rawEventCount[j,tempIndex]+=1
			if dataLL[k][0]!='-':
				label=1
				continue
		labels.append(label)
	return rawEventCount,timeWindow,temNum,labels

def computeData(rawData,numLines,numEvents):
	#'''TF-IDF
	idfData=zeros((numLines,numEvents))
	for j in range(numEvents):
		cnt=count_nonzero(rawData[:,j])
		for i in range(numLines):
			idfData[i,j]=rawData[i,j]*math.log(numLines/float(cnt))

	rawData=rawData.T
	idfData=idfData.T
	meanData=rawData.mean(axis=1)
	data=zeros((numEvents,numLines))
	for i in range(numLines):
		data[:,i]=rawData[:,i]-meanData
	U,sigma,VT=linalg.svd(data,full_matrices=False)
	return data,U,sigma

def getThreshold(para,sigma,numLines,U,numEvents):
	k=4       #number of principal components
	tot=dot(sigma,sigma)
	tmp=0
	for i in range(len(sigma)):
		tmp+=sigma[i]*sigma[i]
		if (tmp/tot)>=para['fraction']:
			break
	k=i+1
	print ('principal components=%d' % (k))

	for i in range(numEvents):
		sigma[i]=sigma[i]*sigma[i]/float(numLines)

	phi=zeros(3)
	for i in range(3):
		for j in range(k,numEvents):
			phi[i]+=math.pow(sigma[j],i+1)

	h0=1-2*phi[0]*phi[2]/(3*phi[1]*phi[1])

	c_alpha= para['c_alpha']
		# double c_alpha = 1.7507; // alpha = 0.08
	# c_alpha = 1.9600;  // alpha = 0.05
	# c_alpha = 2.5758;  // alpha = 0.01
	# c_alpha = 2.807;   // alpha = 0.005
	# c_alpha = 2.9677;  // alpha = 0.003
	# c_alpha = 3.2905;  // alpha = 0.001
	# c_alpha = 3.4808;  // alpha = 0.0005
	# c_alpha = 3.8906;  // alpha = 0.0001
	# c_alpha = 4.4172;  // alpha = 0.00001
	# c_alpha = 4.8916;  //alpha==0.00001
	# c_alpha = 5.3267;  //alpha==0.000001

	threshold=phi[0]*math.pow((c_alpha*math.sqrt(2*phi[1]*h0*h0)/phi[0] + 1.0 + phi[1]*h0*(h0-1)/phi[0]/phi[0]), (1.0/h0))
	P=U[:,:k]
	I=identity(numEvents,int)
	C=I-dot(P,P.transpose())
	return threshold,C

def anomalyDetect(para,data,C,threshold,numLines,labels):
	i=0
	trueAnomaly=labels[:]
	print ('trueAnomaly=%d' % (sum(trueAnomaly)))

	detectAnomaly=zeros((numLines),int)
	print ('threshold=%f' % (threshold))
	for i in range(numLines):
		ya=dot(C,data[:,i])
		SPE=dot(ya,ya)
		if SPE>threshold:
			detectAnomaly[i]=1	#1 represent failure
	print ('detectAnomaly=%d' % (sum(detectAnomaly)) )

	tot=0
	for i in range(numLines):
		if trueAnomaly[i]&detectAnomaly[i]:
			tot+=1
	print ('really true=%d' % (tot))
	precision = tot/float(sum(detectAnomaly))
	recall = tot/float(sum(trueAnomaly))
	print('the precision is %.5f'%(precision))
	print('the recall is %.5f'%(recall))
	F_measure=2*precision*recall/(precision+recall)
	print('F_measure is %.5f'%F_measure)
	return sum(detectAnomaly),sum(trueAnomaly),tot,precision,recall,F_measure
	
def mainProcess(para):
	rawData,numLines,numEvents,labels=processData(para)
	data,U,sigma=computeData(rawData,numLines,numEvents)
	threshold,C=getThreshold(para,sigma,numLines, U,numEvents)
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure = anomalyDetect(para,data,C,threshold,numLines,labels)
	print predictFailure,testFailure,sameFailureNum,precision,recall,F_measure  
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

# def diffTime(para):
# 	# c_alpha = 1.9600;  // alpha = 0.05
# 	# c_alpha = 2.5758;  // alpha = 0.01
# 	# c_alpha = 2.807;   // alpha = 0.005
# 	# c_alpha = 2.9677;  // alpha = 0.003
# 	# c_alpha = 3.2905;  // alpha = 0.001
# 	# c_alpha = 3.4808;  // alpha = 0.0005
# 	# c_alpha = 3.8906;  // alpha = 0.0001
# 	# c_alpha = 4.4172;  // alpha = 0.00001
# 	# c_alpha = 4.8916;  //alpha==0.00001
# 	# c_alpha = 5.3267;  //alpha==0.000001
# 	timeList=[0.0833333333, 0.5, 1.0, 6, 12]
# 	#timeList=[1.9600,2.5758, 2.807, 2.9677,3.2905,3.4808,3.8906,4.4172,4.8916,5.3267]
# 	tiLen=len(timeList)
# 	result=zeros((tiLen,6))
# 	i=0
# 	for ti in timeList:
# 		para['timeInterval']=ti
# 		result[i,:]=mainProcess(para)
# 		i+=1
# 	print result
# 	savetxt('result_PCA_BGL_timeInterval.csv',result,delimiter=',')
# diffTime(para)
mainProcess(para)