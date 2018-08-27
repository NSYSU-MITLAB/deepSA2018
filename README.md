# deepSA2018 at SemEval-2018 Task 1 Affect in Tweets V-oc English competition
The system is based on the our previous work in the SemEval-2018 competition ([deepSA2018 at SemEval-2018 Task 1: Multi-task Learning of Different Label for Affect in Tweets](http://aclweb.org/anthology/S18-1034)).

## SemEval-2018 Task 1 Subtask V-oc
Given a tweet, classify it into one of seven ordinal classes [-3,3].

## Introduction
We improve the system which is based on the our previous work.
Our system consists of five class models, namely three class model, negative model, neutral class model, positive class model, and seven class model.
Different labels are used in different sub-models to learn the polar representation of tweets.
We change the usage of data, add the class weights, add the lexicon features, and train own word vector.
In addition, we retrain the [DeepMoji](https://github.com/bfelbo/DeepMoji) [1] model with transfer learning, and weighted average with polar classification results of our system.
The Pearson correlation coefficient is 0.806 on the test data and could be ranked 4th in the competition.

## Data
We use two datasets for training the system.

1. The dataset is provided for the SemEval-2018 shared task [SemEval-2018 Task 1: Affect in Tweets](http://saifmohammad.com/WebDocs/semeval2018-task1.pdf).
2. The dataset is provided for the SemEval-2017 shared task [SemEval-2017 task 4: Sentiment analysis in Twitter](http://www.aclweb.org/anthology/S17-2088).

The SemEval-2018 includes training, development, and test datasets.<br>
And all datasets of SemEval-2017 are regarded as training data.

## Pre-processing
Tweets are pre-processed using [ekphrasis](https://github.com/cbaziotis/ekphrasis) [2] tool.
```bash
python preprocessing.py
```

## Embedding
We use four embedding sets which are publicly released and one our own embedding set.<br>
Pre-training is performed on aggregated global word-word co-occurrence statistics from a corpus. [GloVe](https://nlp.stanford.edu/projects/glove/) [3]<br>
Pre-training word vector by skip-gram architectures. [ACL-2015](https://www.fredericgodin.com/software/) [4], [word2vec](https://code.google.com/archive/p/word2vec/) [5]<br>
The our own embedding set is our own collection of 140 million tweets from the Twitter API, and the processed tweets are pre-trained using the skip-gram model.<br>

<table>
  <tr>
    <th>Sets</th><th>Algorithm</th><th>Corpus</th><th>Dimension</th><th>Vocabulary</th>
  </tr>
  <tr>
    <td>GloVe-T</td><td rowspan=2>GloVe</td><td>Twitter</td><td>200</td><td>1.2M</td>
  </tr>
   <tr>
    <td>GloVe-G</td><td>General</td><td>300</td><td>2.2M</td>
  </tr>
   <tr>
    <td>ACL-2015</td><td rowspan=3>Skip-Gram</td><td>Twitter</td><td>400</td><td>3M</td>
  </tr>
   <tr>
    <td>Word2Vec</td><td>Google News</td><td>300</td><td>3M</td>
  </tr>
   <tr>
    <td>Self</td><td>Twitter</td><td>400</td><td>0.46M</td>
  </tr>
</table>

The **self** embedding set is available in the links: [Download link](https://drive.google.com/file/d/15zgPiqPS2Zu1Y7jx9izyQeR11dv7K0cN/view?usp=sharing)

## Lexicons
We use four sentiment lexicons, including [AFINN](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010) [6], [Sentiment140](https://github.com/okugami79/sentiment140) [7], [Sentistrength](http://sentistrength.wlv.ac.uk/) [8] and [Vader](https://github.com/cjhutto/vaderSentiment) [9].

## Training
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
* Tensorflow (>= 1.3.0)
* Keras (>= 2.1.0)
* Numpy
* Python (>= 2.7.6)
* [ekphrasis](https://github.com/cbaziotis/ekphrasis)

## References
[1] Bjarke Felbo, Alan Mislove, Anders Søgaard, Iyad Rahwan, and Sune Lehmann, “Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm,” arXiv preprint arXiv:1708.00524, 2017.

[2] Christos Baziotis, Nikos Pelekis, and Christos Doulkeridis, “Datastories at semeval-2017 task 4: Deep lstm with attention for message-level and topic-based sentiment analysis,” in Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), Vancouver, Canada, August 2017, pp. 747–754, Association for Computational Linguistics.

[3] Jeffrey Pennington, Richard Socher, and Christopher D.Manning, “Glove: Global vectors for word representation,” in Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1532–1543.

[4] Timothy Baldwin, Marie-Catherine de Marneffe, Bo Han, Young-Bum Kim, Alan Ritter, and Wei Xu. 2015. Shared tasks of the 2015 workshop on noisy user-generated text: Twitter lexical normalization and named entity recognition. In Proceedings of the Workshop on Noisy User-generated Text, pages 126–135.

[5] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean, “Distributed representations of words and phrases and their compositionality,” in Advances in neural information processing systems, 2013, pp. 3111–3119.

[6] F. &Aring;. Nielsen, “A new anew: Evaluation of a word list for sentiment analysis in microblogs,” arXiv preprint arXiv:1103.2903, 2011.

[7] S. M. Mohammad, S. Kiritchenko, and X. Zhu, “Nrc-canada: Building the state-of-the-art in sentiment analysis of tweets,” arXiv preprint arXiv:1308.6242, 2013.

[8] M. Thelwall, K. Buckley, G. Paltoglou, D. Cai, and A. Kappas, “Sentiment strength detection in short informal text,” Journal of the American Society for Information Science and Technology, vol. 61, no. 12, pp. 2544–2558, 2010.

[9] C. H. E. Gilbert, “Vader: A parsimonious rule-based model for sentiment analysis of social media text,” in Eighth International Conference on Weblogs and Social Media (ICWSM-14). Available at (20/04/16) http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf, 2014.
