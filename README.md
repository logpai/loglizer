<p align="center"> <a href="https://github.com/logpai"> <img src="https://github.com/logpai/logpai.github.io/blob/master/img/logpai_logo.jpg" width="425"></a></p>


# loglizer

Loglizer is a machine learning-based log analysis toolkit for system anomaly detection. Logs are imperative in the development and maintenance process of many software systems. They record detailed
runtime information during system operation that allows developers and support engineers to monitor their systems and dissect anomalous behaviors and errors. Loglizer provides such a tool that implements a set of automated log analysis techniques for anomaly detection. 


:telescope: If you use loglizer in your research for publication, please kindly cite the following paper.
+ Shilin He, Jieming Zhu, Pinjia He, Michael R. Lyu. [Experience Report: System Log Analysis for Anomaly Detection](https://jiemingzhu.github.io/pub/slhe_issre2016.pdf), *IEEE International Symposium on Software Reliability Engineering (ISSRE)*, 2016. [[Bibtex](https://dblp.org/rec/bibtex/conf/issre/HeZHL16)]


## Framework

![Framework of Anomaly Detection](/docs/img/FrameWork.png)

The log analysis framework for anomaly detection usually comprises the following components:

1. **Log collection:** Logs are generated at runtime and aggregated into a centralized place with a data streaming pipeline, such as Flume and Kafka. 
2. **Log parsing:** Logs are naturally unstructured. The goal of log parsing is to convert unstructured log messages into a sequence of structured events, based on which sophisticated machine learning models can be applied. The details of log parsing can be found at [our logparser project](https://github.com/logpai/logparser).
3. **Feature extraction:** Structured logs can be sliced into separate log sequences through interval window, sliding window, or session window. Then, each log sequence is vectorized into feature representation, for example, using an event counting vector. 
4. **Anomaly detection:** Anomaly detection models are trained to check whether a given feature vector is an anomaly or not.


## Models

Anomaly detection models currently available:

| Model | Paper reference |
| :--- | :--- |
| **Supervised models** |
| LR | [**EuroSys'10**] Peter Bodík, Moises Goldszmidt, Armando Fox, Hans Andersen. [Fingerprinting the Datacenter: Automated Classification of Performance Crises](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/07/hiLighter.pdf). [Berkeley, Microsoft, Cornell] |
| Decision Tree | [**ICAC'04**] Mike Chen, Alice X. Zheng, Jim Lloyd, Michael I. Jordan, Eric Brewer. [Failure Diagnosis Using Decision Trees](http://www.cs.berkeley.edu/~brewer/papers/icac2004_chen_diagnosis.pdf). [Berkeley, eBay] |
| SVM | [**ICDM'07**] Yinglung Liang, Yanyong Zhang, Hui Xiong, Ramendra Sahoo. [Failure Prediction in IBM BlueGene/L Event Logs](https://www.researchgate.net/publication/4324148_Failure_Prediction_in_IBM_BlueGeneL_Event_Logs). [Rutgers University, IBM]|
| **Unsupervised models** |
| Clustering | [**ICSE'16**] Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen. [Log Clustering based Problem Identification for Online Service Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/ICSE-2016-2-Log-Clustering-based-Problem-Identification-for-Online-Service-Systems.pdf). [Microsoft]| 
| PCA | [**SOSP'09**] Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan. [Large-Scale System Problems Detection by Mining Console Logs](http://iiis.tsinghua.edu.cn/~weixu/files/sosp09.pdf) [Berkeley, Intel] |
| Invariants Mining | [**ATC'10**] Jian-Guang Lou, Qiang Fu, Shengqi Yang, Ye Xu, Jiang Li. [Mining Invariants from Console Logs for System Problem Detection](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Lou.pdf) [Microsoft, BUPT, NJU]|


## Log data
We have released a variety of log datasets in [loghub](https://github.com/logpai/loghub) for research purposes. If you are interested in these datasets, please request the logs through the link.


## Usage
Please follow [the demo](./docs/demo.md) in the docs to get started.


## Contributors
+ [Shilin He](https://shilinhe.github.io), The Chinese University of Hong Kong
+ [Jieming Zhu](https://jiemingzhu.github.io), The Chinese University of Hong Kong, currently at Huawei Noah's Ark Lab
+ [Pinjia He](https://pinjiahe.github.io/), The Chinese University of Hong Kong, currently at ETH Zurich


## Feedback
For any questions or feedback, please post to [the issue page](https://github.com/logpai/loglizer/issues/new). 


## History
* May 14, 2016: initial commit 
* Sep 21, 2017: update code and readme 
* March 21, 2018: rewrite most of the code and add detailed comments
* Dec 15, 2018: restructure the repository with hands-on demo
