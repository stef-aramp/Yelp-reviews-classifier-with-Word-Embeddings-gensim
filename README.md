# Yelp-reviews-classifier-with-Word-Embeddings-gensim
Using word embeddings from a pre-trained word2vec model to classify yelp reviews.


## Yelp reviews sentiment analysis 


As textual data comes in then:
- non-english reviews are gone
- reviews get cleaned and tokenized
- Word Vectors are created from gensim’s word2vec model.
- Train 4 different NN architectures: 
	- Simple LSTM 
	- Bidirectional LSTM
	- CNN

(**CNN-LSTM to be done**)




### Dependencies
————————————————

- Python 3.6
- keras (tensorflow backend)
- gensim
- spaCy
- Pandas/Numpy
- RegEx
- Seaborn/Matplotlib
- Sci-Kit Learn

### Contents
————————————

- yelp classifier using **gensim** (skip gram) for word vectors.
- **expressions** module is used to detect standard expressions with repeated (or not) characters.
- **emoticons** module is used to detect and replace basic emoticons with text
- **slang** module is used to re-define some slang words and acronyms.
- pre-work contains a few lines of code to create final data set.


