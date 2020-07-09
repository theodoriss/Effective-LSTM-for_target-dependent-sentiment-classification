#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import string
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer


def create_dataframe(file_path):
    sentences=[] # list of lists, where every sentence is a list of the words that form it. will be used for splitting right and left of target
    #sentences_text=[]  # list of strings-sentences. replace target symbol with target, to be used in tokenization
    target=[]
    label=[]
    with open(file_path,encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip('\n').lower().split()) 
            target.append(next(file).strip('\n'))
            label.append(int(next(file).strip('\n')))


    words_left=[]# list of lists
    words_right=[]# list of lists
    for target_number,sentence in enumerate(sentences):
        for position,word in enumerate(sentence):
            if word=='$t$':
                words_left.append(sentence[:position]+[target[target_number]])
                words_right.append([target[target_number]]+sentence[position+1:])
                break

    data=pd.DataFrame(columns=['sentence','words_left','words_right','target','label'])
    data['sentence']=sentences
    data['words_left']=words_left
    data['words_right']=words_right
    data['target']=target
    data['label']=label

    data['sentence']= data.apply(lambda x: ' '.join(x['sentence']).replace('$t$',x['target']).split(),axis=1) # replace $t$ symbol with its target value

    return data


# In[ ]:


def preprocessing(list_of_sentences,dataframe):
    processed_sentences=[]
    stops = set(stopwords.words("english"))
    ps = PorterStemmer()
    for s in list_of_sentences:
       # print(str(s).translate(str.maketrans('', '', string.punctuation)))
        text = [w for w in s if not w in stops and len(w)>2]
       # print(' '.join(text))
        temp=[]
        for w in text:
            if w not in ' '.join(list(dataframe['target'])):
                rootword=ps.stem(w)
                temp.append(rootword)
            else:
                temp.append(w)
        processed_sentences.append(str(temp).translate(str.maketrans('', '', string.punctuation)))
    return processed_sentences


# In[ ]:


def create_sequences(tokenizer,train_dataframe,test_dataframe):
    #MAX_SEQUENCE_LENGTH = 14 # largest sequence length for left,right words from both train and test sets
   # MAX_SEQUENCE_LENGTH_sentences = 17 # largest sequence for the whole sentence
    


    train_tokens_left = tokenizer.texts_to_sequences(train_dataframe['words_left'])
    train_tokens_right = tokenizer.texts_to_sequences(train_dataframe['words_right'])
    train_tokens_all = tokenizer.texts_to_sequences(train_dataframe['sentence'])
    train_tokens_left= sequence.pad_sequences(train_tokens_left, padding='pre',maxlen= len(max(train_tokens_left,key=len)))
    train_tokens_right= sequence.pad_sequences(train_tokens_right, padding='post',maxlen=len(max(train_tokens_right,key=len)))
    train_tokens_all= sequence.pad_sequences(train_tokens_all, padding='pre',maxlen=len(max(train_tokens_all,key=len)))



    test_tokens_left = tokenizer.texts_to_sequences(test_dataframe['words_left'])
    test_tokens_right = tokenizer.texts_to_sequences(test_dataframe['words_right'])
    test_tokens_all = tokenizer.texts_to_sequences(test_dataframe['sentence'])
    test_tokens_left= sequence.pad_sequences(test_tokens_left, padding='pre',maxlen= len(max(train_tokens_left,key=len)))
    test_tokens_right= sequence.pad_sequences(test_tokens_right, padding='post',maxlen=len(max(train_tokens_right,key=len)))
    test_tokens_all= sequence.pad_sequences(test_tokens_all, padding='pre',maxlen=len(max(train_tokens_all,key=len)))





    return train_tokens_left,train_tokens_right,train_tokens_all,test_tokens_left,test_tokens_right,test_tokens_all


# In[ ]:


def create__inputs_outputs(train_dataframe,test_dataframe,num_of_tokenizer):
    number_of_words=num_of_tokenizer
    tokenizer = text.Tokenizer(num_words=number_of_words)
    tokenizer.fit_on_texts(train_dataframe['sentence'])
    
    train_tokens_left,train_tokens_right,train_tokens_all,test_tokens_left,test_tokens_right,test_tokens_all=create_sequences(tokenizer,train_dataframe,test_dataframe)

    
    labels_train= np.asanyarray(pd.get_dummies(train_dataframe['label'],prefix=['label']))
    labels_test= np.asanyarray(pd.get_dummies(test_dataframe['label'],prefix=['label']))

    
    return tokenizer,train_tokens_left,train_tokens_right,train_tokens_all,test_tokens_left,test_tokens_right,test_tokens_all,labels_train,labels_test

    


# In[ ]:


def create_matrix_L(tokenizer,word2vec):
    df=np.zeros((len(tokenizer.index_word),word2vec.vector_size),dtype='float')
    for number,word in enumerate(tokenizer.index_word.values()):
        if word in word2vec.wv.vocab:
            df[number]=word2vec[word]
    return df

