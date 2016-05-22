# loglizer
A Python toolkit for anomaly detection via log analysis

***
## Introduction
System logs can be utilized to detect system anomalies, which plays an important role in the maintainence of large-scale distributed systems. In this toolbox, we implenmented and released six state-of-the-art log-based anomaly detection methods, including three supervised methods and three unsupervised methods. The framework of our anomaly detection toolbox are given as following:

## Framework
**1. Log collection:** Logs are generated and collected from systems, and saved as \*.log file.    
**2. Log Parsing:** Raw Logs are parsed to log events by log parsers.  
**3. Feature Creation:** Grouping Logs into log sequences with various windowing methods, and forme the event count vectors.  
**4. Anomaly Detection:** Building anomaly detection model, and detecting anomalies.  

## Anomaly detection Methods
* Supervised Anomaly Detection
  1. Logistic Regression [Fingerprinting the Datacenter:Automated Classification of Performance Crises]()
  2. Decision Tree [Failure Diagnosis Using Decision Trees]()
  3. SVM [Failure Prediction in IBM BlueGeneL Event Logs]()
* Unsupervised Anomaly Detection
  1. Log Clustering [Log Clustering based Problem Identification for Online Service Systems]()
  2. PCA [Large-Scale System Problems Detection by Mining Console Logs]()
  3. Invariants Mining [Mining Invariants from Console Logs for System Problem Detection]()


