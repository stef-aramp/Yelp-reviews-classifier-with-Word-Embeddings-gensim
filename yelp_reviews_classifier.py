#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 23:42:28 2018

@author: stephanosarampatzes
"""

import pandas as pd

df = pd.read_csv('yelp800K.csv')

df = df.drop('Unnamed: 0', axis = 1)

# load libraries for cleaning

import numpy as np
import re

import spacy
import en_core_web_sm as en
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
nlp = en.load()

import warnings
warnings.filterwarnings('ignore')

# remove a couple of phrases in german (testing)

df.text[119825] = re.sub(" \n\nich bin ein berliner","", df.text[119825])
df.text[18060] = re.sub("Alle den Torte und Kuchen schmecken gut--mein lieblings ist die Bär-Kralle!","", df.text[18060])

# collect reviews that are not in english
non_eng = []
for i in df.index:
    if re.findall(r"\b(?:der|ich|und|ein|wir|bei|mit|uns|kann|nein|sie|für|nicht\
                       |avec|bien|qu|je|c'est|dans|não|une|au|nous|vous|bonne|les\
                       |tout|et|avoir|mais|pas|nada)\b", str(df['text'][i])) != []:
        #print(i)
        non_eng.append(i)
        
# keep some reviews in englih contains non-english words        
eng = []
for i in non_eng:
    if df.text[i].find('with') != -1 or df.text[i].find('this') != -1:
        eng.append(i)

# create a list with indices to drop
drop_list = [ind for ind in non_eng if ind not in eng]
print('length of droplist: ', len(drop_list))

df = df.drop(df.index[drop_list]).reset_index(drop = True)
        
# drop a couple of rows that put text_tokenizer in infinite loop
df = df.drop(df.index[[230425,478224]]).reset_index(drop = True)  
        
# delete temporary lists
del [non_eng, eng, drop_list]

# checking for missing values
print("Do I have NaN's? : ", df.isna().sum)

### CLEANING AND TOKENIZING REVIEWS

# Remove URLS
def remove_urls(text):
    pattern = r'(https?:\\/\\/[a-zA-Z0-9.\\/\_?=&-]+)'
    return(re.sub(pattern, '', text))

# Module for repeated characters
from expressions import expressions

# Module for Emoticons
from emoticons import emoticons

# Lemmatization
def lemmas(text):
    text = nlp(text)
    text = [t.lemma_.lower().strip() if t.lemma_ != '-PRON-' else t.lower_ for t in text]
    return(text)

# Remove Stop Words
def stopWords(text):
    text = ' '.join([t for t in text if t not in set(list(STOP_WORDS)+
                     list(punctuation) + [ "--", "---", "...", "“", "”"])])
    return(text)

# Module for Slangs & Acronyms
from slang import slanger, slang_map
# search every word separately

def slang_in_corpus(text):
    text = ' '.join([slanger(w) if w in list(slang_map.keys()) else w for w in text])
    return(text)

# Remove left chars, nums, symbols
def leftChars(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    # remove single characters
    text = ' '.join([w for w in text.split() if len(w)>=2])
    return(text)

def text_normalizer(text, strip_text = False, sub_suffix = True, urls = True, expression = True, emojis = True ,
                    lemmatization = True, stop_words = False, split = False,
                    slangs = True, leftos = True, white_spaces = True):
    
    if strip_text:
        text = text.strip().lower()
        
    if sub_suffix:
        text = re.sub("n't", ' not', text)
        text = re.sub("'d", ' would', text)
        text = re.sub("'ll", ' will', text)
        text = re.sub("'m", ' am', text)
        text = re.sub("Im", 'I am', text)
        text = re.sub("'ve", ' have', text)
        text = re.sub("its", 'it is', text)
        text = re.sub("it's", 'it is', text)
        text = re.sub("dont", 'do not', text)
        text = re.sub('yucky', 'yuck', text)

    if urls:
        text = remove_urls(text)
        
    # use functions in a specific order for better results        
    if expression:
        text = expressions(text)
        
    # replace emojis before any symbol is gone forever
    if emojis:
        text = emoticons(text)

    if lemmatization:
        text = lemmas(text)
        
    #if stop_words:
    #    text = stopWords(text)
    
    if split:
        text = text.split()
        
    # replace slangs & acronyms 
    if slangs:
        text = slang_in_corpus(text)
        
    if leftos:
        text = leftChars(text)
    
    if white_spaces:
        text = re.sub(' +', ' ', text)
    
    return(text.strip().split())

# create labels: stars >= 4 ->positive review , stars <=3 -> negative review
# I want to keep my problem binary, because I want as much as possible balance 
# in my classes

def label(x):
    if x > 3:
        val = 1
        #print('+')
    else:
        val = 0
        #print('-')
    return(val)
    
df['labels'] = df.stars.apply(lambda row: label(row))

# ratio of labels
print('ratio of labels: \n\n{}'.format(df['labels'].value_counts()/len(df)*100))

# apply text_normalizer function
from tqdm import tqdm
import pickle

# tokenize and clean sentences / store them in a list
def tokenize_reviews(list_of_sentences):
    reviews = []
    
    for text in tqdm(list_of_sentences):
        txt = text_normalizer(text)
        reviews.append(txt)
    
    with open("reviews_final.txt", "wb") as fp:
        pickle.dump(reviews, fp)
        
    return(reviews)

# create a corpus with sentences
train_sents = list(df['text'].values)
# collect tokens
reviews = tokenize_reviews(train_sents)

# CHECK FOR EMPTY REVIEWS (LISTS)
count_empties = len([empty for empty in reviews if empty == []])

for r in reviews:
    if r == []:
        r.append('unknown sentiment')

#####
# Create Word Vectors to Train Word2Vec Model with Gensim

import gensim
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases

# keep some useful bigrams with two basic parameters for count and score
bigram = Phraser(Phrases(reviews, min_count=50, threshold=120))
#bigram.phrasegrams
print('Length of bigrams: ', len(bigram.phrasegrams))

# find and replace bigrams in reviews
reviews_bigram = []
for review in reviews:
    reviews_bigram.append(bigram[review])

del reviews


### Word2Vec Model / 100 dimensions word vecotrs space
### Skip Gram with 10 contextual words and 5 words for negative sampling 
model = Word2Vec(reviews_bigram, size=100, window=10, min_count=5, workers=12, sg=1, negative=5)


# model.save('yelp_bigram_model.w2v')
# model = gensim.models.Word2Vec.load('yelp_bigram_model.w2v')

# Examine if model is trained well
print("most similars words to 'steak' :", model.most_similar('steak'))
# king + woman - man = queen (?)
print(model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man']))
# father + daughter - son = mother (?)
print(model.wv.most_similar_cosmul(positive=['daughter', 'father'], negative=['son']))


word_vectors = model.wv
print('length of word vectors: {}'.format(len(word_vectors.vocab)))


### INITIALIZE THE EMBEDDINGS

###########
# I need a vocabulary equal to the number of word vectors and a fixed length
# of reviews sequences. What is the most suitable value for sequence length ?
# Few lines of code analyse length of reviews 

import matplotlib.pyplot as plt
import seaborn as sns

lens = [len(sen) for sen in reviews_bigram]

plt.figure(figsize=(18,8))
sns.boxplot(lens)
plt.title('Length of reviews', fontsize = 15)
plt.show()

print('Q1, Median, Q3:', np.percentile(lens, [25,50,75]), 'and Mean:', np.mean(lens))

# proportion of reviews with number of words under or equal to the mean
print(round(len([i for i in lens if i <108])/len(lens)*100,2),'%')

# proportion between mean and Q3
print(round(len([i for i in lens if (i>108 and i <138)])/len(lens)*100,2),'%')

del lens
###########

max_n_words = len(word_vectors.vocab) #40549 vectors
max_sequence_len = 120

# update vocabulary
from collections import Counter
vocab = Counter()

vocab.update([word for sent in reviews_bigram for word in sent])

import keras
from keras.preprocessing.sequence import pad_sequences

# give an index to each word starting from most frequent
word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(max_n_words))}

word_index = {k:(v+2) for k,v in word_index.items()}
word_index['PAD'] = 0
word_index['START'] = 1
word_index['Unknown'] = 2

# convert sequence of words to sequence of numbers
sequences = [[word_index.get(t,0) for t in review] for review in reviews_bigram]

# labels
labels = df['labels'].values

# split to train and test(validation) sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size = 0.2, random_state = 42)

x_train = pad_sequences(sequences=x_train, maxlen=max_sequence_len, padding='pre', truncating='post')
x_test = pad_sequences(sequences=x_test, maxlen= max_sequence_len, padding='pre', truncating='post')


### create the embedding matrix

embed_size = 100 # size of embedding layer
n_words = min(max_n_words, len(word_vectors.vocab)) # number of words used as vocabulary

word_vectors_mtrx = (np.random.rand(n_words+3, embed_size) - 0.5)/5.0 # initialize weights

for word, ind in word_index.items():
    if ind >= max_n_words:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        word_vectors_mtrx[i] = embedding_vector
    except:
        pass


    
from keras.layers import (Dense, LSTM, Embedding, Dropout, SpatialDropout1D, GlobalMaxPool1D, Conv1D)
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
    
### NN Models ###
# 1. Simple LSTM
# 2. Bidirectional LSTM
# 3. CNN
# 4. Stacked LSTM
#################

# 1. Simple LSTM
import keras.backend as K
K.clear_session()

model1 = Sequential()
model1.add(Embedding(n_words+3, embed_size, weights = [word_vectors_mtrx], input_length=max_sequence_len, trainable = False))
model1.add(SpatialDropout1D(0.2))
model1.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))

model1.add(Dense(1, activation='sigmoid'))

model1.summary()

model1.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001), metrics=['accuracy'])

results = model1.fit(x_train, y_train, epochs = 5, batch_size=128, verbose=1,
                    validation_split= 0.2, validation_data = (x_test, y_test), shuffle = True)


# 2. Bidirectional LSTM
K.clear_session()

model2 = Sequential()
model2.add(Embedding(n_words+3, embed_size, weights = [word_vectors_mtrx], input_length=max_sequence_len, trainable = False))
model2.add(SpatialDropout1D(0.2))
model2.add(Bidirectional(LSTM(256, dropout=0.2)))

model2.add(Dense(1, activation='sigmoid'))

model2.summary()

model2.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001), metrics=['accuracy'])

results2 = model2.fit(x_train, y_train, epochs = 5, batch_size=128, verbose=1,
                    validation_split= 0.2, validation_data = (x_test, y_test), shuffle = True)


# 3. CNN

K.clear_session()

model3 = Sequential()
model3.add(Embedding(n_words+3, embed_size, weights = [word_vectors_mtrx], input_length=max_sequence_len, trainable = False))
model3.add(SpatialDropout1D(0.2))
model3.add(Conv1D(256, 3, activation='relu'))
model3.add(GlobalMaxPool1D())
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.2))

model3.add(Dense(1, activation='sigmoid'))

model3.summary()

model3.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001), metrics=['accuracy'])

results3 = model3.fit(x_train, y_train, epochs = 5, batch_size=128, verbose=1,
                    validation_split= 0.2, validation_data = (x_test, y_test), shuffle = True)

# 4. Stacked LSTM

K.clear_session()

model4 = Sequential()
model4.add(Embedding(n_words+3, embed_size, weights = [word_vectors_mtrx], input_length=max_sequence_len, trainable = False))
model4.add(SpatialDropout1D(0.2))
model4.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
model4.add(Bidirectional(LSTM(64, dropout=0.2)))

model4.add(Dense(1, activation = 'sigmoid'))

model4.summary()

model4.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001), metrics=['accuracy'])

results4 = model4.fit(x_train, y_train, epochs = 5, batch_size=128, verbose=1,
                    validation_split= 0.2, validation_data = (x_test, y_test), shuffle = True)




### PLOTS
# Validation Loss
plt.figure(1, figsize=(10,10))
plt.plot(results.history['val_loss'])
plt.plot(results2.history['val_loss'])
plt.plot(results3.history['val_loss'])
plt.plot(results4.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['LSTM', 'biDir LSTM', 'CNN', 'Stacked LSTM'])
plt.show()

# Validation Accuracy
plt.figure(2, figsize=(10,10))
plt.plot(results.history['val_acc'])
plt.plot(results2.history['val_acc'])
plt.plot(results3.history['val_acc'])
plt.plot(results4.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('val_accuracy')
plt.legend(['LSTM', 'biDir LSTM', 'CNN', 'Stacked LSTM'])
plt.show()

# Acurracy
plt.figure(3, figsize=(10,10))
plt.plot(results.history['acc'])
plt.plot(results2.history['acc'])
plt.plot(results3.history['acc'])
plt.plot(results4.history['acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['LSTM', 'biDir LSTM', 'CNN', 'Stacked LSTM'])
plt.show()

