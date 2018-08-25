<p align="center"> <a href="https://github.com/logpai"> <img src="https://github.com/logpai/logpai.github.io/blob/master/img/logpai_logo.jpg" width="500" height="125"/> </a>
</p>


# loglizer
**A Python toolkit for anomaly detection via log analysis**

Loglizer is an open-source python tool for automatic log-based anomaly detection with machine learning techniques. In this project, six popular anomaly detection methods are implemented and evaluated on two public datasets. More details (e.g., experimental results, findings) can be found in our [paper](http://ieeexplore.ieee.org/document/7774521/).


## Log Data
We collected a number of log datasets in [loghub](https://github.com/logpai/loghub). Please send a request to acquire the data through the link.

### More about logTemplateMap.csv of BGL data
The file locates in the *demo_bgl* folder. The file is used to quickly index the event Id for each log message. For example, suppose there are 1000 log messages and each of them can be parsed into a corresponding log event (assume 10 log events in total). Then there are 1000 rows in the logTemplateMap.csv, and each entry contains the event id (1-10) of that log.

***
## Background
Anomaly detection plays an important role in management of modern large-scale distributed systems. Logs, which record system runtime information, are widely used for anomaly detection. Traditionally, developers (or operators) often inspect the logs manually with keyword search and rule matching. The increasing scale and complexity of modern systems, however, make the volume of logs explode, which renders the infeasibility of manual inspection. To reduce manual effort, many anomaly detection methods based on automated log analysis are proposed.  
In our paper, we provide a detailed review and evaluation of six state-of-the-art log-based anomaly detection methods, including three supervised methods and three unsupervised methods, and also release an open-source toolkit allowing ease of reuse. These methods have been evaluated on two publicly-available production log datasets.

## Overview of framework
**1. Log collection:** Logs are generated and collected by system and sofwares during running, which includes distributed systems (e.g., Spark, Hadoop), standalone systems (e.g., Windows, Mac OS) and softwares (e.g., Zookeeper).     
**2. Log Parsing:** Raw Logs contain too much runtime information (e.g., IP address, file name). These variable information are often removed after log parsing as they are useless for debugging. After parsing, raw logs become log events, which are abstraction of raw logs. Details are given in our previous work: [Logparser](https://github.com/logpai/logparser)  
**3. Feature Extraction:** Logs are grouped into log sequences via Task ID or time, and these log sequences are vectorized and weighted.  
**4. Anomaly Detection:** Some machine learning models are trained and applied to detect anomalies.  

The framework is illustrated as follows:

![Framework of Anomaly Detection](/img/FrameWork.png)

In our toolbox, we mainly focus on Feature Extraction and Anomaly Detection, while Log Collection and Log Parsing are out of the scope of this project. To be more specific, the input is the parsed log events, and the output is whether it is anomaly for each log sequence.

## Anomaly detection methods
* ***Supervised Anomaly Detection:***  
  **1. Logistic Regression:**  
  Paper: [Fingerprinting the Datacenter: Automated Classification of Performance Crises](http://dl.acm.org/citation.cfm?id=1755926)  
  Affiliations: UC Berkeley, Cornell, Microsoft  
  **2. Decision Tree:**  
  Paper: [Failure Diagnosis Using Decision Trees](http://www.cs.berkeley.edu/~brewer/papers/icac2004_chen_diagnosis.pdf)  
  Affiliations: UC Berkeley, eBay   
  **3. SVM:**  
  Paper: [Failure Prediction in IBM BlueGeneL Event Logs](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4536397)  
  Affiliations: Rutgers University, IBM     
* ***Unsupervised Anomaly Detection:***  
  **1. Log Clustering:**  
  Paper: [Log Clustering based Problem Identification for Online Service Systems](http://www.msr-waypoint.net/apps/pubs/default.aspx?id=260324)  
  Affiliations: Microsoft Research   
  **2. PCA:**  
  Paper: [Large-Scale System Problems Detection by Mining Console Logs](https://www.usenix.org/legacy/event/sysml08/tech/full_papers/xu/xu.pdf)  
  Affiliations: UC Berkeley  
  **3. Invariants Mining:**  
  Paper: [Mining Invariants from Console Logs for System Problem Detection](http://research.microsoft.com/pubs/121673/Mining%20Invariants%20from%20Console%20Logs.pdf)  
  Affiliations: Microsoft Research  

## Paper
Our paper is published on the 27th International Symposium on Software Reliability Engineering (**ISSRE 2016**), Ottawa, Canada. The information can be found here:  
**Title: Experience Report: System Log Analysis for Anomaly Detection**    
**Authors:** Shilin He, Jieming Zhu, Pinjia He, and Michael R. Lyu  
**Paper link:** [paper](http://ieeexplore.ieee.org/document/7774521/) 

**Bibtex:**<br />
*@Inproceedings{He16ISSRE,<br />
  title={Experience Report: System Log Analysis for Anomaly Detection},<br />
  author={He, S. and Zhu, J. and He, P. and Lyu, M. R.},<br />
  booktitle={ISSRE'16: Proc. of the 27th International Symposium on Software Reliability Engineering}<br />
}<br />*


## Question:
For easy maintenance, please raise an issue at [ISSUE](https://github.com/logpai/loglizer/issues/new)


## History:
* May 14, 2016: initial commit 
* Sep 21, 2017: update code and ReadME 
* March 21, 2018: rewrite most of the code, re-organize the project structure, and add detailed comments
