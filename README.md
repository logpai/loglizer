<p align="center"> <a href="https://github.com/logpai"> <img src="https://github.com/logpai/logpai.github.io/blob/master/img/logpai_logo.jpg" width="425"></a></p>


# loglizer


**Loglizer is a machine learning-based log analysis toolkit for automated anomaly detection**. 
> Loglizer是一款基于AI的日志大数据分析工具, 能用于自动异常检测、智能故障诊断等场景
  

Logs are imperative in the development and maintenance process of many software systems. They record detailed
runtime information during system operation that allows developers and support engineers to monitor their systems and track abnormal behaviors and errors. Loglizer provides a toolkit that implements a number of machine-learning based log analysis techniques for automated anomaly detection. 

:telescope: If you use loglizer in your research for publication, please kindly cite the following paper.
+ Shilin He, Jieming Zhu, Pinjia He, Michael R. Lyu. [Experience Report: System Log Analysis for Anomaly Detection](https://jiemingzhu.github.io/pub/slhe_issre2016.pdf), *IEEE International Symposium on Software Reliability Engineering (ISSRE)*, 2016. [[Bibtex](https://dblp.org/rec/bibtex/conf/issre/HeZHL16)][[中文版本](https://github.com/AmateurEvents/article/issues/2)]
**(ISSRE Most Influential Paper)**

## Framework

![Framework of Anomaly Detection](/docs/img/framework.png)

The log analysis framework for anomaly detection usually comprises the following components:

1. **Log collection:** Logs are generated at runtime and aggregated into a centralized place with a data streaming pipeline, such as Flume and Kafka. 
2. **Log parsing:** The goal of log parsing is to convert unstructured log messages into a map of structured events, based on which sophisticated machine learning models can be applied. The details of log parsing can be found at [our logparser project](https://github.com/logpai/logparser).
3. **Feature extraction:** Structured logs can be sliced into short log sequences through interval window, sliding window, or session window. Then, feature extraction is performed to vectorize each log sequence, for example, using an event counting vector. 
4. **Anomaly detection:** Anomaly detection models are trained to check whether a given feature vector is an anomaly or not.


## Models

Anomaly detection models currently available:

| Model | Paper reference |
| :--- | :--- |
| **Supervised models** |
| LR | [**EuroSys'10**] [Fingerprinting the Datacenter: Automated Classification of Performance Crises](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/07/hiLighter.pdf), by Peter Bodík, Moises Goldszmidt, Armando Fox, Hans Andersen. [**Microsoft**] |
| Decision Tree | [**ICAC'04**] [Failure Diagnosis Using Decision Trees](http://www.cs.berkeley.edu/~brewer/papers/icac2004_chen_diagnosis.pdf), by Mike Chen, Alice X. Zheng, Jim Lloyd, Michael I. Jordan, Eric Brewer. [**eBay**] |
| SVM | [**ICDM'07**] [Failure Prediction in IBM BlueGene/L Event Logs](https://www.researchgate.net/publication/4324148_Failure_Prediction_in_IBM_BlueGeneL_Event_Logs), by Yinglung Liang, Yanyong Zhang, Hui Xiong, Ramendra Sahoo. [**IBM**]|
| **Unsupervised models** |
| LOF | [**SIGMOD'00**] [LOF: Identifying Density-Based Local Outliers](), by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, Jörg Sander. |
| One-Class SVM | [**Neural Computation'01**] [Estimating the Support of a High-Dimensional Distribution](), by John Platt, Bernhard Schölkopf, John Shawe-Taylor, Alex J. Smola, Robert C. Williamson. |
| Isolation Forest | [**ICDM'08**] [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf), by Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. |
| PCA | [**SOSP'09**] [Large-Scale System Problems Detection by Mining Console Logs](http://iiis.tsinghua.edu.cn/~weixu/files/sosp09.pdf), by Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan. [**Intel**] |
| Invariants Mining | [**ATC'10**] [Mining Invariants from Console Logs for System Problem Detection](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Lou.pdf), by Jian-Guang Lou, Qiang Fu, Shengqi Yang, Ye Xu, Jiang Li. [**Microsoft**]|
| Clustering | [**ICSE'16**] [Log Clustering based Problem Identification for Online Service Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/ICSE-2016-2-Log-Clustering-based-Problem-Identification-for-Online-Service-Systems.pdf), by Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen. [**Microsoft**]|
| DeepLog (coming)| [**CCS'17**] [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf), by Min Du, Feifei Li, Guineng Zheng, Vivek Srikumar. |
| AutoEncoder (coming)| [**Arxiv'18**] [Anomaly Detection using Autoencoders in High Performance Computing Systems](https://arxiv.org/abs/1811.05269), by Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca Benini. |


## Log data
We have collected a set of labeled log datasets in [loghub](https://github.com/logpai/loghub) for research purposes. If you are interested in the datasets, please follow the link to submit your access request.

## Install
```bash
git clone https://github.com/logpai/loglizer.git
cd loglizer
pip install -r requirements.txt
```

## API usage

```python
# Load HDFS dataset. If you would like to try your own log, you need to rewrite the load function.
(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(...)

# Feature extraction and transformation
feature_extractor = preprocessing.FeatureExtractor()
feature_extractor.fit_transform(...) 

# Model training
model = PCA()
model.fit(...)

# Feature transform after fitting
x_test = feature_extractor.transform(...)
# Model evaluation with labeled data
model.evaluate(...)

# Anomaly prediction
x_test = feature_extractor.transform(...)
model.predict(...) # predict anomalies on given data
```

For more details, please follow [the demo](./docs/demo.md) in the docs to get started. Please note that all ML models are not magic, you need to figure out how to tune the parameters in order to make them work on your own data. 

## Benchmarking results 

If you would like to reproduce the following results, please run [benchmarks/HDFS_bechmark.py](./benchmarks/HDFS_bechmark.py) on the full HDFS dataset (HDFS100k is for demo only).

|       |            | HDFS |     |
| :----:|:----:|:----:|:----:|
| **Model** | **Precision** | **Recall** | **F1** |
| LR| 0.955 |	0.911 |	0.933 |
| Decision Tree | 0.998 |	0.998 |	0.998 |
| SVM| 0.959 |	0.970 |	0.965 |
| LOF | 0.967 | 0.561 | 0.710 |
| One-Class SVM | 0.995 | 0.222| 0.363 |
| Isolation Forest |  0.830 | 0.776 | 0.802 |
| PCA | 0.975 | 0.635 | 0.769|
| Invariants Mining | 0.888 | 0.945 | 0.915|
| Clustering | 1.000 | 0.720 | 0.837 |

## Contributors
+ [Shilin He](https://shilinhe.github.io), The Chinese University of Hong Kong
+ [Jieming Zhu](https://jiemingzhu.github.io), The Chinese University of Hong Kong, currently at Huawei Noah's Ark Lab
+ [Pinjia He](https://pinjiahe.github.io/), The Chinese University of Hong Kong, currently at ETH Zurich


## Feedback
For any questions or feedback, please post to [the issue page](https://github.com/logpai/loglizer/issues/new). 


## History
* May 14, 2016: initial commit 
* Sep 21, 2017: update code and readme 
* Mar 21, 2018: rewrite most of the code and add detailed comments
* Feb 18, 2019: restructure the repository with hands-on demo
