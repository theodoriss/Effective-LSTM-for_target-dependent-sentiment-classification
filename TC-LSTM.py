#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM,Concatenate,concatenate,Dropout,Input,RepeatVector
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
import gensim.downloader as api
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import main



# In[2]:





# In[3]:


train_data=main.create_dataframe('train.raw')
test_data=main.create_dataframe('test.raw')



# In[4]:


train_processed = train_data.apply(lambda x: main.preprocessing(x,train_data) if x.name in ['sentence', 'words_left','words_right'] else x)
test_processed = test_data.apply(lambda x: main.preprocessing(x,test_data) if x.name in ['sentence', 'words_left','words_right'] else x)


# In[5]:



# In[28]:





# In[30]:


word2vec = api.load("glove-twitter-100")


# In[6]:


def create_target_vector_array(dataframe,word2vec):
    target_vector=np.empty([len(dataframe),word2vec.vector_size])
    for number,target in enumerate(dataframe.target):
        y=np.zeros(word2vec.vector_size) 
        for t in target.split():
            if t in word2vec.wv.vocab.keys():
                y+= word2vec.wv[t]
        target_vector[number]= y/len(target.split())
    return target_vector


# In[7]:


tokenizer,X_train_left,X_train_right,X_train,X_test_left,X_test_right,X_test,Y_train,Y_test=main.create__inputs_outputs(train_processed,test_processed,num_of_tokenizer=250)


# In[8]:


L=main.create_matrix_L(tokenizer,word2vec) # create the word embedding matrix


# In[9]:


target_vector_array_train=create_target_vector_array(train_processed,word2vec)


# In[11]:


stopping = EarlyStopping(monitor='val_loss', patience=4,restore_best_weights=False,verbose=0)


# In[12]:


def LSTM_TC_model_left(L,train_tokens,target_vector_array,num_classes=3):
    MAX_SEQUENCE_LENGTH = len(max(train_tokens,key=len))
    input_tokens=Input(shape=MAX_SEQUENCE_LENGTH,dtype='int32')
    input_target_vector=Input(shape=target_vector_array.shape[1],dtype='float32')
    
    embb= Embedding(L.shape[0],L.shape[1], input_length=train_tokens.shape[1],embeddings_initializer=Constant(L),trainable=False)(input_tokens)
    repeat_target= RepeatVector(MAX_SEQUENCE_LENGTH)(input_target_vector) #plut the word vector to each input of the LSTM
    conc=Concatenate(axis=-1)([embb,repeat_target])
    lstm= LSTM(100, dropout=0.3, recurrent_dropout=0.1,recurrent_regularizer='l2')(conc)
    #lstm=Dropout(0.3)(lstm)
    
    
    #dense=Dense(64)(lstm)
    #dense=BatchNormalization()(dense)

    #dense=Dropout(0.3)(dense)


    output = Dense(num_classes, activation='softmax')(lstm)
    model=Model([input_tokens,input_target_vector],outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[13]:


def LSTM_TC_model_right(L,train_tokens,target_vector_array,num_classes=3):
    MAX_SEQUENCE_LENGTH = len(max(train_tokens,key=len))

    input_tokens=Input(shape=MAX_SEQUENCE_LENGTH,dtype='int32')
    input_target_vector=Input(shape=target_vector_array.shape[1],dtype='float32')
    
    embb= Embedding(L.shape[0],L.shape[1], input_length=train_tokens.shape[1],embeddings_initializer=Constant(L),trainable=False)(input_tokens)
    repeat_target= RepeatVector(MAX_SEQUENCE_LENGTH)(input_target_vector)
    conc=Concatenate(axis=-1)([embb,repeat_target])
    lstm= LSTM(100, dropout=0.3, recurrent_dropout=0.1,go_backwards=True)(conc)
    #lstm=Dropout(0.3)(lstm)
    
    
    #dense=Dense(64)(lstm)
    #dense=BatchNormalization()(dense)

    #dense=Dropout(0.3)(dense)


    output = Dense(num_classes, activation='softmax')(lstm)
    model=Model([input_tokens,input_target_vector],outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[14]:


model_LSTM_left=LSTM_TC_model_left(L,X_train_left,target_vector_array_train)


# In[16]:


model_LSTM_left.fit([X_train_left,target_vector_array_train],Y_train,batch_size=128,epochs=30,verbose=0,validation_split=0.2,use_multiprocessing=True,workers=0,callbacks=[stopping])


# In[20]:


model_LSTM_right=LSTM_TC_model_right(L,X_train_right,target_vector_array_train)


# In[21]:


model_LSTM_right.fit([X_train_right,target_vector_array_train],Y_train,batch_size=128,epochs=30,verbose=0,validation_split=0.2,use_multiprocessing=True,workers=0,callbacks=[stopping])


# In[22]:


def TC_LSTM(model_left,model_right,num_classes=3):
    left_vector=model_left.layers[-2].output
    right_vector=model_right.layers[-2].output 
    
    merged=Concatenate()([left_vector,right_vector]) # concatenate the two LSTM layers of the previous models
    output=Dropout(0.3)(merged)
    
    output= Dense(num_classes, activation='softmax')(output)

    
    merged_model=Model([model_left.input,model_right.input],output)


    #model= Dense(num_classes, activation='softmax')(merged)
    for idx,layer in enumerate(merged_model.layers):
        if idx+1<len(merged_model.layers): # train only the last layer,activated with softmax
            layer.trainable = False
        #print(idx+1,len(merged_model.layers))
    merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return merged_model


# In[62]:


model_concatenate=TC_LSTM(model_LSTM_left,model_LSTM_right)


# In[63]:


model_concatenate.summary()


# In[64]:


model_concatenate.fit([X_train_left,target_vector_array_train,X_train_right,target_vector_array_train],Y_train,batch_size=128,epochs=30,verbose=1,validation_split=0.2,workers=0,use_multiprocessing=True,callbacks=[stopping])


# In[31]:


target_vector_array_test=create_target_vector_array(test_processed,word2vec)


# In[32]:


model_concatenate.evaluate([X_test_left,target_vector_array_test,X_test_right,target_vector_array_test],Y_test,batch_size=128,verbose=0,workers=0,use_multiprocessing=True)


# In[45]:


preds = model_concatenate.predict([X_test_left,target_vector_array_test,X_test_right,target_vector_array_test],verbose=0)

Ypred = np.argmax(preds, axis=1)

print(classification_report(np.argmax(Y_test,axis=1), Ypred, labels=None,target_names=['negative','neutral','positive'], digits=3))


# In[46]:


print(confusion_matrix(np.argmax(Y_test,axis=1),Ypred))


# In[65]:



def plot_history(model,name):

    # summarize history for accuracy
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title(name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


name="TC-LSTM"
plot_history(model_concatenate, name)


# In[ ]:




