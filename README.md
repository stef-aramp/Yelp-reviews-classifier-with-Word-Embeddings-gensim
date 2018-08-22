# Yelp-reviews-classifier-with-Word-Embeddings-gensim
Using word embeddings from a pre-trained word2vec model to classify yelp reviews.


## Yelp reviews sentiment analysis 


As textual data comes in then:
- non-english reviews are gone
- reviews get cleaned and tokenised
- word embeddings are created form gensim’s word vectors
- Train 4 different NN architectures: 
	- Simple LSTM 
	- Bidirectional LSTM
	- Stacked LSTM
	- CNN

(**CNN-LSTM to be done**)

(**jupyter notebook with results and visuals to be uploaded**)


### Dependencies
————————————————

- python 3.6
- keras (tensorflow backend)
- gensim
- spacy
- pandas/numpy
- RegEx
- seaborn/matplotlib
- sic-kit learn

### Contents
————————————

- yelp classifier using **gensim** (skip gram) for word vectors.
- **expressions** module is used to detect standard expressions with repeated (or not) characters.
- **emoticons** module is used to detect and replace basic emoticons with text
- **slang** module is used to re-define some slang words and acronyms.
- pre-work contains a few lines of code to create final data set.


