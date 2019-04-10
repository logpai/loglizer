# Demo

### Dependency

The loglizer toolkit is implemented with Python and requires a number of dependency requirements installed. 

+ python 3.6
+ scipy
+ numpy
+ scikit-learn=0.20.3
+ pandas

We recommend users to use Anaconda, which is a popular Python data science platform with many common packages pre-installed. The virtual enviorment can be set up via `conda`:

```
$ conda create -n py36 -c anaconda python=3.6
$ conda activate py36
```

For ease of reproducing our benchmarking results, we have also built a docker image for the running evironment. If you have docker installed, you can easily pull and run a docker container as follows:

```
$ mkdir loglizer
$ git clone https://github.com/logpai/loglizer.git loglizer/
$ docker run --name loglizer -v loglizer:/loglizer -it logpai/anaconda:py3.6 bash
$ cd /loglizer/demo
```

### Run loglizer
You can try the demo scripts of loglizer on HDFS_100k.log_structured.csv as follows:

```
# Clone the project from Github
$ git clone https://github.com/logpai/loglizer.git

# Run PCA demo
$ cd loglizer/demo/
$ python PCA_demo.py

# Or run InvariantsMiner demo
$ python InvariantsMiner_demo.py

# If you want to apply loglizer to your own log data, and even have no label data, 
# you can follow the following script to run an unsupervised anomaly detection model. 
$ python PCA_demo_without_labels.py
$ python InvariantsMiner_demo_without_labels.py
```

