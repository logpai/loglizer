# loglizer
A Python toolkit for anomaly detection via log analysis

We will give a brief introduction about anomaly detection first, next, the overview of framework is described. Finally, six anomay detection methods are given, together with their codes and original papers.

***
## Introduction
System logs can be utilized to detect system anomalies, which plays an important role in the maintainence of large-scale distributed systems. In this toolbox, we implenmented and released six state-of-the-art log-based anomaly detection methods, including three supervised methods and three unsupervised methods. The framework of our anomaly detection toolbox are given as following:

## Overview of framework
**1. Log collection:** Logs are generated and collected from systems, and saved as \*.log file.    
**2. Log Parsing:** Raw Logs are parsed to log events by log parsers.  
**3. Feature Creation:** Grouping Logs into log sequences with various windowing methods, and forme the event count vectors.  
**4. Anomaly Detection:** Building anomaly detection model, and detecting anomalies.  

In our toolbox, we mainly focus on Step 3 (Feature Creation) and Step 4 (Anomaly Detection), because raw logs are collected as our dataset, and then parsed by our log parsing tool: [Logparser](https://github.com/cuhk-cse/logparser). In our anomaly detection toolkit, the input is the parsed log events and the output is detected anomaly instances. 

## Anomaly detection Methods
* ***Supervised Anomaly Detection:***  
  **1. Logistic Regression:**  
  paper: [Fingerprinting the Datacenter:Automated Classification of Performance Crises](http://delivery.acm.org/10.1145/1760000/1755926/p111-bodik.pdf?ip=137.189.205.45&id=1755926&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2E63D3CA449C1BD759%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=618976206&CFTOKEN=89837482&__acm__=1463921661_0e7a2639d248dd919d3ec446bfa12586)  
  **2. Decision Tree:**  
  paper: [Failure Diagnosis Using Decision Trees](http://www.cs.berkeley.edu/~brewer/papers/icac2004_chen_diagnosis.pdf)  
  **3. SVM: Paper:**  
  paper: [Failure Prediction in IBM BlueGeneL Event Logs](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4536397)  
* ***Unsupervised Anomaly Detection:***  
  **1. Log Clustering:**  
  paper: [Log Clustering based Problem Identification for Online Service Systems](http://www.msr-waypoint.net/apps/pubs/default.aspx?id=260324)  
  **2. PCA:**  
  paper: [Large-Scale System Problems Detection by Mining Console Logs](https://www.usenix.org/legacy/event/sysml08/tech/full_papers/xu/xu.pdf)  
  **3. Invariants Mining:**  
  paper: [Mining Invariants from Console Logs for System Problem Detection](http://research.microsoft.com/pubs/121673/Mining%20Invariants%20from%20Console%20Logs.pdf)  



