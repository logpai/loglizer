## Demo

### Dependency

The loglizer toolkit is implemented with Python and requires a number of dependency requirements installed. 

+ python 3.6
+ scipy
+ numpy
+ scikit-learn
+ pandas

We recommend users to use Anaconda, which is a popular Python data science platform with many common packages pre-installed. The enviorment can be set up via `conda`:

```
$ conda create -n py36 -c anaconda python=3.6
$ conda activate py36
```

For ease of reproducing our benchmark results, we have also build a docker image for the running evironments. If you have docker installed, you can easily pull and run a docker container as follows:

```
$ mkdir loglizer
$ git clone https://github.com/logpai/loglizer.git loglizer/
$ docker run --name loglizer -it -v loglizer:/loglizer logpai/anaconda:py3.6 bash
$ cd /loglizer/demo
```

### Run loglizer
You can try the demos of loglizer on HDFS_2k.log as follows:

```
$ mkdir loglizer
$ git clone https://github.com/logpai/loglizer.git loglizer/
$ cd loglizer/demo/
$ python PCA_demo.py
```

