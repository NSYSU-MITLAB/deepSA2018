from __future__ import print_function

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

import string
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)
#---------------------------------------------------------------------------------------------------------------------------------
#data file (.txt)
data_file = '2018-Valence-oc-En-train'

tweet = []
label = []
header = True
with open('./Data/original/' + data_file + '.txt', 'r') as File:
	for line in File:
		if header :
			header = False
			continue
		linesp = line.split('\t')
		tweet.append(" ".join(text_processor.pre_process_doc(linesp[1])))
		label.append(linesp[3].split(':')[0])
File.close()

#save preprocessed data
Fd = open('./Data/processed/' + data_file + '-data.tok','w')
Fd.write('\n'.join(tweet))
Fd.close()
#save label
Fl = open('./Data/processed/' + data_file + '-label.txt','w')
Fl.write('\n'.join(label))
Fl.close()


