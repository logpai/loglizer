#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import math
'''for HDFS data using PCA'''
#================PCA on HDFS data====================#
#													 #
#													 #
#=====================================================
para={
'path':'../Data/HDFS_data/',
'tfVectorName':'rm_repeat_rawTFVector.txt',
'tfLabelName':'rm_repeat_mlabel.txt',
'fraction':0.95
}
numEvents=29
def processData(para):
	numRows=575061
	rawData=zeros((numRows,numEvents))
	print('for HDFS data using PCA')
	print('Loading data...')
	filePath=para['path']+para['tfVectorName']
	numLines=0
	with open(filePath) as f:
		for line in f:
			line=line.strip().split()
			for j in range(numEvents):
				rawData[numLines,j]=int(line[j])
			numLines += 1
	print('total line number is %d'%(numLines))
	return rawData,numLines

def computeData(rawData,numLines):
	#'''TF-IDF
	idfData=zeros((numLines,numEvents))
	for j in range(numEvents):
		cnt=count_nonzero(rawData[:,j])
		for i in range(numLines):
			idfData[i,j]=rawData[i,j]*math.log(numLines/float(cnt))

	rawData=rawData.T
	idfData=idfData.T
	meanData=idfData.mean(axis=1)
	data=zeros((numEvents,numLines))
	for i in range(numLines):
		data[:,i]=idfData[:,i]-meanData
	U,sigma,VT=linalg.svd(data,full_matrices=False)
	return data,U,sigma

def getThreshold(sigma,numLines,U):
	k=4       #number of principal components
	tot=dot(sigma,sigma)
	tmp=0
	for i in range(len(sigma)):
		tmp+=sigma[i]*sigma[i]
		if (tmp/tot)>=para['fraction']:
			break
	k=i+1
	print('principal components=%d' % (k))

	for i in range(numEvents):
		sigma[i]=sigma[i]*sigma[i]/float(numLines)

	phi=zeros(3)
	for i in range(3):
		for j in range(k,numEvents):
			phi[i]+=math.pow(sigma[j],i+1)

	h0=1-2*phi[0]*phi[2]/(3*phi[1]*phi[1])

	c_alpha=3.2905

	threshold=phi[0]*math.pow((c_alpha*math.sqrt(2*phi[1]*h0*h0)/phi[0] + 1.0 + phi[1]*h0*(h0-1)/phi[0]/phi[0]), (1.0/h0))
	P=U[:,:k]
	I=identity(numEvents,int)
	C=I-dot(P,P.transpose())
	return threshold,C

def anomalyDetect(para,data,C,threshold,numLines):
	trueAnomaly=zeros((numLines),int)
	i=0
	f1=open(para['path']+para['tfLabelName'])
	lines=f1.readlines()
	for line in lines:
		line=line.split()
		trueAnomaly[i]=int(line[0])
		i+=1
	print('trueAnomaly=%d' % (sum(trueAnomaly)))

	detectAnomaly=zeros((numLines),int)
	print('threshold=%f' % (threshold))
	for i in range(numLines):
		ya=dot(C,data[:,i])
		SPE=dot(ya,ya)
		if SPE>threshold:
			detectAnomaly[i]=1	#1 represent failure
	print('detectAnomaly=%d' % (sum(detectAnomaly)) )

	tot=0
	for i in range(numLines):
		if trueAnomaly[i]&detectAnomaly[i]:
			tot+=1
	print('really true=%d' % (tot))
	precision = tot/float(sum(detectAnomaly))
	recall = tot/float(sum(trueAnomaly))
	print('the precision is %.5f'%(precision))
	print('the recall is %.5f'%(recall))
	F_measure=2*precision*recall/(precision+recall)
	print('F_measure is %.5f'%F_measure)
	return sum(detectAnomaly),sum(trueAnomaly),tot,precision,recall,F_measure

def mainProcess(para):
	rawData,numLines=processData(para)
	data,U,sigma=computeData(rawData,numLines)
	threshold,C=getThreshold(sigma,numLines, U)
	predictFailure,testFailure,sameFailureNum,precision,recall,F_measure = anomalyDetect(para,data,C,threshold,numLines)
	print(predictFailure,testFailure,sameFailureNum,precision,recall,F_measure)
	return predictFailure,testFailure,sameFailureNum,precision,recall,F_measure

if __name__ == '__main__':
	mainProcess(para)
