# loglizer
**A Python toolkit for anomaly detection via log analysis**

Loglizer is an open-source python tool for automatic anomaly detection via log analysis. In this project, six popular anomaly detection methods are implemented and compared on two public datasets, and detailed information (e.g., experimental results, findings) can be found in our [paper](http://ieeexplore.ieee.org/document/7774521/).

Notes: Recently, many researchers and developers emailed us for the datasets, but unfortunately some datasets cannot be distributed as we do not have the copyright. The good news is that we ([logpai](https://github.com/logpai)) are trying to public some log datasets ([loghub](https://github.com/logpai/loghub)). Please follow us for the latest progress.

## Paper
Our paper is published on the 27th International Symposium on Software Reliability Engineering (**ISSRE 2016**), Ottawa, Canada. The information can be found here:  
**Title: Experience Report: System Log Analysis for Anomaly Detection**    
**Authors:** Shilin He, Jieming Zhu, Pinjia He, and Michael R. Lyu  
**Paper link:** [paper](http://ieeexplore.ieee.org/document/7774521/)   
Please feel free to contact us if you have any questions: slhe@cse.cuhk.edu.hk  

***
## Background
Anomaly detection plays an important role in management of modern large-scale distributed systems. Logs, which record system runtime information, are widely used for anomaly detection. Traditionally, developers (or operators) often inspect the logs manually with keyword search and rule matching. The increasing scale and complexity of modern systems, however, make the volume of logs explode, which renders the infeasibility of manual inspection. To reduce manual effort, many anomaly detection methods based on automated log analysis are proposed.  
In our paper, we provide a detailed review and evaluation of six state-of-the-art log-based anomaly detection methods, including three supervised methods and three unsupervised methods, and also release an open-source toolkit allowing ease of reuse. These methods have been evaluated on two publicly-available production log datasets.

The framework of our anomaly detection toolbox are given as following:

## Overview of framework
**1. Log collection:** Logs are generated and collected by system and sofwares during running, which includes distributed systems (e.g., Spark, Hadoop), standalone systems (e.g., Windows, Mac OS) and softwares (e.g., Zookeeper).     
**2. Log Parsing:** Raw Logs contain too much runtime information (e.g., IP address, file name). These variable information are often removed after log parsing as they are useless for debugging. After parsing, raw logs become log events, which are abstraction of raw logs. Details are given in our previous work: [Logparser](https://github.com/logpai/logparser)  
**3. Feature Extraction:** Logs are grouped into log sequences via Task ID or time, and these log sequences are vectorized and weighted.  
**4. Anomaly Detection:** Some machine learning models are trained and applied to detect anomalies.  

The framework is illustrated as follows:

![Framework of Anomaly Detection](/README/FrameWork.png)

In our toolbox, we mainly focus on Feature Extraction and Anomaly Detection, while Log Collection and Log Parsing are out of the scope of this project. To be more specific, the input is the parsed log events, and the output is whether it is anomaly for each log sequence.

## Anomaly detection methods
* ***Supervised Anomaly Detection:***  
  **1. Logistic Regression:**  
  paper: [Fingerprinting the Datacenter:Automated Classification of Performance Crises](http://dl.acm.org/citation.cfm?id=1755926)  
  **2. Decision Tree:**  
  paper: [Failure Diagnosis Using Decision Trees](http://www.cs.berkeley.edu/~brewer/papers/icac2004_chen_diagnosis.pdf)  
  **3. SVM:**  
  paper: [Failure Prediction in IBM BlueGeneL Event Logs](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4536397)  
* ***Unsupervised Anomaly Detection:***  
  **1. Log Clustering:**  
  paper: [Log Clustering based Problem Identification for Online Service Systems](http://www.msr-waypoint.net/apps/pubs/default.aspx?id=260324)  
  **2. PCA:**  
  paper: [Large-Scale System Problems Detection by Mining Console Logs](https://www.usenix.org/legacy/event/sysml08/tech/full_papers/xu/xu.pdf)  
  **3. Invariants Mining:**  
  paper: [Mining Invariants from Console Logs for System Problem Detection](http://research.microsoft.com/pubs/121673/Mining%20Invariants%20from%20Console%20Logs.pdf)  
