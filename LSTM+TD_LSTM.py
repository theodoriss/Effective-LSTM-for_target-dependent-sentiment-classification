#!/usr/bin/env python
# coding: utf-8




import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM,Concatenate,concatenate,Dropout,Input
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




train_data=main.create_dataframe('train.raw')
test_data=main.create_dataframe('test.raw')


# ### Preprocessing of data




train_processed = train_data.apply(lambda x: main.preprocessing(x,train_data) if x.name in ['sentence', 'words_left','words_right'] else x)
test_processed = test_data.apply(lambda x: main.preprocessing(x,test_data) if x.name in ['sentence', 'words_left','words_right'] else x)



tokenizer,X_train_left,X_train_right,X_train,X_test_left,X_test_right,X_test,Y_train,Y_test=main.create__inputs_outputs(train_processed,test_processed,num_of_tokenizer=250)


word2vec = api.load("glove-twitter-100")


L=main.create_matrix_L(tokenizer,word2vec)


del word2vec,tokenizer,train_processed,test_processed

stopping = EarlyStopping(monitor='val_loss', patience=4,restore_best_weights=False,verbose=1)


# ### Normal LSTM model

# In[10]:


def LSTM_model(L,train_tokens,num_classes=3):
    model = Sequential()
    model.add(Embedding(L.shape[0],L.shape[1], input_length=train_tokens.shape[1],embeddings_initializer=Constant(L),trainable=False))
    #model.add(BatchNormalization())
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.1,recurrent_activation='sigmoid',recurrent_regularizer='l2'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[11]:


model_LSTM=LSTM_model(L,X_train)


# In[12]:


model_LSTM.fit(X_train,Y_train,batch_size=128,epochs=30,verbose=0,validation_split=0.2,use_multiprocessing=True,workers=0,callbacks=[stopping])


# In[13]:


model_LSTM.evaluate(X_test,Y_test,batch_size=128,verbose=0,workers=0,use_multiprocessing=True)


# In[14]:


preds = model_LSTM.predict([X_test],verbose=0)

Ypred = np.argmax(preds, axis=1)
#Ytest = test_generator.classes  # shuffle=False in test_generator
print('LSTM classification Report and confusion matrix')
print(classification_report(np.argmax(Y_test,axis=1), Ypred, labels=None,target_names=['-1','0','1'], digits=3))


# In[15]:


print(confusion_matrix(np.argmax(Y_test,axis=1),Ypred))


# ### LSTM-left

# In[16]:


def LSTM_model_left(L,train_tokens,num_classes=3):
    model = Sequential()
    model.add(Embedding(L.shape[0],L.shape[1], input_length=train_tokens.shape[1],embeddings_initializer=Constant(L),trainable=False))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.1,recurrent_activation='sigmoid',recurrent_regularizer='l2'))
    model.add(Dropout(0.3))


    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[17]:


model_LSTM_left=LSTM_model_left(L,X_train_left)


# In[18]:


model_LSTM_left.fit(X_train_left,Y_train,batch_size=128,epochs=30,verbose=0,validation_split=0.2,use_multiprocessing=True,workers=0,callbacks=[stopping])


# ### LSTM-right

# In[19]:


def LSTM_model_right(L,train_tokens,num_classes=3):
    model = Sequential()
    model.add(Embedding(L.shape[0],L.shape[1], input_length=train_tokens.shape[1],embeddings_initializer=Constant(L),trainable=False))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.1,recurrent_activation='sigmoid',recurrent_regularizer='l2',go_backwards=True))
   # model.add(Dropout(0.3))


    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[20]:


model_LSTM_right=LSTM_model_right(L,X_train_right)


# In[21]:


model_LSTM_right.fit(X_train_right,Y_train,batch_size=128,epochs=30,verbose=0,validation_split=0.2,use_multiprocessing=True,workers=0,callbacks=[stopping])


# ### Merge the 2 LSTM layers and feed them to another netowrk

# In[22]:


def TD_LSTM(model_left,model_right,num_classes=3):
    left_vector=model_left.layers[-2].output
    right_vector=model_right.layers[-2].output # concatenate the 2nd to last layers
    
    merged=Concatenate()([left_vector,right_vector])
    output=Dropout(0.1)(merged)
    
    output= Dense(num_classes, activation='softmax')(output)

    
    merged_model=Model([model_left.input,model_right.input],output)


    for idx,layer in enumerate(merged_model.layers):
        if idx+1<len(merged_model.layers): # train only the last layer,activated by softmax
            layer.trainable = False
    merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return merged_model

    


# In[23]:


model_TD=TD_LSTM(model_LSTM_left,model_LSTM_right)


# In[24]:


model_TD.summary()


# In[25]:


model_TD.fit([X_train_left,X_train_right],Y_train,batch_size=128,epochs=30,verbose=1,validation_split=0.2,workers=0,use_multiprocessing=True,callbacks=[stopping])


# In[27]:


model_TD.evaluate([X_test_left,X_test_right],Y_test,batch_size=128,verbose=0,workers=0,use_multiprocessing=True)


# In[30]:



preds = model_TD.predict([X_test_left,X_test_right],verbose=0)

Ypred = np.argmax(preds, axis=1)
#Ytest = test_generator.classes  # shuffle=False in test_generator

print('TD-LSTM classification Report and confusion matrix')


print(classification_report(np.argmax(Y_test,axis=1), Ypred, labels=None,target_names=['-1','0','1'], digits=3))


# In[31]:


print(confusion_matrix(np.argmax(Y_test,axis=1),Ypred))


# In[43]:



def plot_history(models_list,names_list):
    for n, model in enumerate (models_list):
    # summarize history for accuracy
        #plt.plot(model.history.history['accuracy'])
        plt.plot(model.history.history['val_accuracy'])
        plt.title( names_list[n]+ 'validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
    plt.legend([names_list[n-1],names_list[n]], loc='upper left')
    plt.show()


name="TD-LSTM"
plot_history([model_LSTM,model_TD], ['LSTM','TD-LSTM'])


# In[ ]:




