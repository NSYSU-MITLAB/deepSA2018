# deepSA2018 at SemEval-2018 Task 1 Affect in Tweets V-oc English competition
The system is based on the our previous work in the SemEval-2018 competition ([deepSA2018 at SemEval-2018 Task 1: Multi-task Learning of Different Label for Affect in Tweets](http://aclweb.org/anthology/S18-1034)).

## SemEval-2018 Task 1 Subtask V-oc
Given a tweet, classify it into one of seven ordinal classes [-3,3].

## Introduction
We improve the system which is based on the our previous work.
Our system consists of five class models, namely three class model, negative model, neutral class model, positive class model, and seven class model.
Different labels are used in different sub-models to learn the polar representation of tweets.
We change the usage of data, add the class weights, add the lexicon features, and train own word vector.
In addition, we retrain the [DeepMoji](https://github.com/bfelbo/DeepMoji) model with transfer learning, and weighted average with polar classification results of our system.
The Pearson correlation coefficient is 0.806 on the test data and could be ranked 4th in the competition.

## Data
We use two datasets for training the system.

1. The dataset is provided for the SemEval-2018 shared task [SemEval-2018 Task 1: Affect in Tweets](http://saifmohammad.com/WebDocs/semeval2018-task1.pdf).
2. The dataset is provided for the SemEval-2017 shared task [SemEval-2017 task 4: Sentiment analysis in Twitter](http://www.aclweb.org/anthology/S17-2088).

The SemEval-2018 includes training, development, and test datasets.<br>
And all datasets of SemEval-2017 are regarded as training data.

## Pre-processing
Tweets are pre-processed using [ekphrasis](https://github.com/cbaziotis/ekphrasis) tool.
```bash
python preprocessing.py
```

## Embedding
We use four embedding sets which are publicly available and one our own embedding set.<br>
Pre-training is performed on aggregated global word-word co-occurrence statistics from a corpus. [GloVe](https://nlp.stanford.edu/projects/glove/)<br>
Pre-training word vector by skip-gram architectures. [ACL-2015](https://www.fredericgodin.com/software/), [word2vec](https://code.google.com/archive/p/word2vec/)<br>
The our own embedding set is our own collection of 140 million tweets from the Twitter API, and is pre-trained using the skip-gram model.<br>
The **self** embedding set available in the links: [Download link](http://)

## Usage
```bash
python system.py <usage of data> <embedding> <class weights> <lexicon features>
```
The parameter &lt;usage of data&gt; can take three possible values:
  * train-18: Using the SemEval-2018 training data to train the system.<br>
  * train-all: Using the SemEval-2018 training data and SemEval-2017 data to train the system.<br>
  * train: Only using the SemEval-2018 training data and SemEval-2017 data to train three class model in the system.<br> Other class models are trained by SemEval-2018 training data. (The best method for training.)
  
The parameter &lt;embedding&gt; can take five possible values:
  * glove-t: 200 dimension.
  * glove-g: 300 dimension.
  * acl2015: 400 dimension.
  * word2vec: 300 dimension.
  * self: 400 dimension.
  
The parameter &lt;class weights&gt; can take two possible values:
  * True: Training the system with skewness robust class weights.
  * False: Training the system without skewness robust class weights.
  
The parameter &lt;lexicon features&gt; can take two possible values:
  * True: Adding the lexicon features, and concatenating with word vector.
  * False: Not using lexicon features.
  
**The best result of training the system is**
```bash
python system.py train self True True
```
## Ensemble Methods
We use a simple ensemble method to further boost the performance of tweet polarity classification system.
### Weighted Average
```bash
python weightedaverage.py
```

### Stacking
```bash
python stacking.py
```

## Prerequisites
* Tensorflow 
* Keras 
* Numpy
* Python

## References

