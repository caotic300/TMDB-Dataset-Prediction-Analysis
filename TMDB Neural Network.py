# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:52:39 2020

@author: Husam
"""
import json
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mlp

def convert(df, columns): 
    for c in columns:
        # Convert json format to python list
        df[c]=df[c].apply(json.loads)
        
        # Obtain first 10 from columns cast and crew
        if (c == 'cast' or 'crew' or 'production_companies'): 
            for index,i in zip(df.index,df[c]):
                limit = 5
                if len(i) < 5:
                    limit = len(i)
                
                temp_list=[]
                for j in range(limit):
                    # Json format of 'id' & 'name'
                    temp_list.append((i[j]['name'])) 
                df.loc[index,c]= str(temp_list)

        # For any other columns
        else:    
            for index,i in zip(df.index,df[c]):
                temp_list=[]
                for j in range(len(i)):
                    temp_list.append((i[j]['name'])) 
                df.loc[index,c]= str(temp_list)
    
         
        df[c] = df[c].str.strip('[]')       # Remove Sqr Brackets
        df[c] = df[c].str.replace(' ','')   # Remove empty space 
        df[c] = df[c].str.replace("'",'')   # Remove quotations
        df[c] = df[c].str.split(',')        # Format into list
        
        # Sort elements 
        for i,j in zip(df[c],df.index):
            temp_list = i
            temp_list.sort()
            df.loc[j,c]=str(temp_list)
            
        df[c] = df[c].str.strip('[]')       
        df[c] = df[c].str.replace(' ','')    
        df[c] = df[c].str.replace("'",'')   
       
        lst = df[c].str.split(',')        
        if len(lst) == 0:
            df[c] = None
        else:
            df[c]= df[c].str.split(',')
            
    return df

path1 = "D:\Computing\Computer Science Year 3\IN3062 Introduction to Artificial Intelligence\TMDB-home"
filename_read = os.path.join(path1,"tmdb_5000_movies.csv")
movie = pd.read_csv(filename_read,na_values=['NA','?'])

path2 = "D:\Computing\Computer Science Year 3\IN3062 Introduction to Artificial Intelligence\TMDB-home"
filename_read = os.path.join(path2,"tmdb_5000_credits.csv")
credit = pd.read_csv(filename_read,na_values=['NA','?'])

movies = movie.merge(credit, left_on='id', right_on='movie_id', how='left')

movies = movies.drop(columns=['homepage','original_language','title_y', 'title_x',
                              'overview','production_countries','release_date',
                              'runtime','spoken_languages','status','tagline', 
                              ], axis=1)

print(movies.dtypes)

test = convert(movies,  ['genres', 'keywords', 'production_companies', 'cast', 'crew'])

# Removing entries with a score of 0
mask_avg = (test['vote_average'] != 0)
test = test[mask_avg]

# Removing entries with empty cast/crew/companies
drop = []
for i in test.index:
    if (test['production_companies'][i] == [''] and test['cast'][i] == [''] and 
       test['crew'][i] == ['']): 
       drop.append(i)
       
test = test.drop(drop, axis = 0)
print(test.shape)
print(test.dtypes)

temp = test.copy()

temp['vote_round'] = temp.vote_average.round()
temp = temp.drop(columns = 'vote_average')

countCol = temp.groupby("vote_round")["vote_round"].transform(len)
mask = (countCol >= 10)

temp = temp[mask]

print(temp.shape)
print(temp.dtypes)
print(temp['production_companies'].shape)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(sparse_output=True) 

def encode_list(dataf, feature):
    enc_df = dataf.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(temp.pop(feature)),
                                                     index = dataf.index,
                                                     columns = mlb.classes_))
    return enc_df
    
enc_df = encode_list(temp, feature = 'genres')
#enc_df = encode_list(temp, feature = 'production_companies')
#enc_df = encode_list(temp, feature = 'cast')
#enc_df = encode_list(temp, feature = 'crew')


print(enc_df.shape)
print(enc_df.dtypes)

encoded = enc_df.copy()

from sklearn.model_selection import train_test_split

# Remove irrelavant features 
# Vote_Avg has decent correlation with: Popularity, Revenue, Budget
encoded = encoded.drop(columns=['id', 'keywords', 'original_title',
                                 'vote_count','movie_id',
                                'production_companies', 'cast', 'crew']) 
                                # , 'genres', 'cast', 'crew', 'production_companies'


U = encoded.drop(columns=['vote_round'])
v = encoded['vote_round']
v_flat = v.values.ravel()

print(U.dtypes, U.shape)
print(v_flat.shape)

# Converting datatypes for encoded genres 
for i in range(3, 24):
    name = U.columns[i]
    U[name] = np.asarray(U[name]).astype('float32')
    
from keras.utils import to_categorical
v_flat = to_categorical(v_flat) 

X_train, X_test, y_train, y_test = train_test_split(U, v_flat, test_size=0.25, random_state=9)

from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

print("X_train: ", X_train.dtypes)
print("y_train: ", y_train.shape)
print(U.shape)


callbacks = [EarlyStopping(monitor='loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]

model = Sequential()
model.add(Dense(24, input_dim=U.shape[1], activation = 'sigmoid')) #Input Layer -> Hidden Layer 1
model.add(Dense(16, activation = 'sigmoid')) # Hidden Layer 1 -> Hidden Layer 2
model.add(Dense(9,activation='softmax')) # Hidden Layer 2 -> Output Layer
model.compile(loss='categorical_crossentropy', optimizer='adam') # Choosing Loss & Optimizer 
model.fit(X_train, y_train, verbose = 2,epochs = 200, callbacks = callbacks, batch_size = 100) # Training data
pred = model.predict(X_test)  # make predictions 
pred = np.argmax(pred,axis=1) # now pick the most likely outcome
y_compare = np.argmax(y_test,axis=1) 

#calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))









































enc_df.to_csv(r'D:\Computing\Computer Science Year 3\IN3062 Introduction to Artificial Intelligence\TMDB-home\new-encoded.csv',
            index = False)

print(test.dtypes)
print(test['genres_cat'])

print(temp.shape)

encoded = enc_df.copy()



"""
Use columns for X: 
    - budget 
    - Popularity 
    - Revenue 
    - genres 
    
Predict vote_avg 

"""
print(encoded.dtypes)
from sklearn.model_selection import train_test_split

encoded = encoded.drop(columns=['id', 'keywords', 'original_title',
                                 'vote_count','movie_id', 'genres',
                                'production_companies']) 
                                # , 'genres', 'cast', 'crew', 


U = encoded.drop(columns=['vote_round'])
v = encoded['vote_round']
v_flat = v.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(U, v_flat, test_size=0.25, random_state=9)


# Predicting Contiunous Data 

# Linear Regression 
# Regression Trees
# Neural Network 
# Use binning to put rating in categories



"""
from sklearn.ensemble import RandomForestClassifier

accuracy_data = []
nums = []
for i in range(len(y)):
    rf_model = RandomForestClassifier(n_estimators=i,criterion="entropy")
    rf_model.fit(X_train, y_train)
    y_model = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_model)
    accuracy_data.append(accuracy)
    nums.append(i)

print(y.dtypes)
print(U.dtypes)
print(v_flat.dtypes)
"""

















































