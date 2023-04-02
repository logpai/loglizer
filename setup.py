from setuptools import setup, find_packages

setup(
    name='loglizer',
    version='1.0.0',
    url='https://github.com/logpai/loglizer.git',
    author='LogPAI',
    author_email='info@logpai.com',
    description='Machine learning-based log analysis toolkit for automated anomaly detection',
    packages=find_packages(),
    install_requires=['sklearn', 'pandas', 'torch', 'numpy'],
)