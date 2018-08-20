# deepSA2018 at SemEval-2018 Task 1 Affect in Tweets V-oc English competition
The system is based on the our previous work in the SemEval-2018 competition ([deepSA2018 at SemEval-2018 Task 1: Multi-task Learning of Different
Label for Affect in Tweets](http://aclweb.org/anthology/S18-1034)).
## Introduction
We improve the system, and the Pearson correlation coefficient is 0.806 on the test data and could be ranked 4th in the competition.

## Data

## Pre-processing

## Embedding

## Usage
```bash
python system.py <usage of data> <embedding> <class weights> <lexicon features>
```
The parameter &lt;usage of data&gt; can take three possible values:
  * train-18 : Using the SemEval-2018 training data to train the system.<br>
  * train-all : Using the SemEval-2018 training data and SemEval-2017 data to train the system.<br>
  * train : Only using the SemEval-2018 training data and SemEval-2017 data to train three class model in the system. Other class models are trained by SemEval-2018 training data. (The best method for training.)
  
The parameter &lt;embedding&gt; can take five possible values:
  * GloVe-T :
  * GloVe-G :
  * ACL-2015 :
  * Word2Vec :
  * Self : The embedding set is our own collection of 140 million tweets from the Twitter API, and is pre-trained using the Skip-gram model.
  
The parameter &lt;class weights&gt; can take two possible values:
  * True : Training the system with skewness robust class weights.
  * False : Training the system without skewness robust class weights.
  
The parameter &lt;lexicon features&gt; can take two possible values:
  * True : Adding the lexicon features, and concatenating with word vector.
  * False : Not using lexicon features.
  
## Prerequisites
