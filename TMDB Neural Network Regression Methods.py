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

path1 = "./"
filename_read = os.path.join(path1,"tmdb_5000_movies.csv")
movie = pd.read_csv(filename_read,na_values=['NA','?'])

path2 = "./"
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
#enc_df_two = encode_list(test.copy(), feature = 'genres')
#print(end_df)
print(enc_df.shape)
print(enc_df.dtypes)

encoded = enc_df.copy()
encoded_two = test.copy()
from sklearn.model_selection import train_test_split

# Remove irrelavant features 
# Vote_Avg has decent correlation with: Popularity, Revenue, Budget
encoded = encoded.drop(columns=['id', 'keywords', 'original_title',
                                 'vote_count','movie_id',
                                'production_companies', 'cast', 'crew']) 
                                # , 'genres', 'cast', 'crew', 'production_companies'
encoded_two = encoded_two.drop(columns=['id', 'keywords', 'original_title',
                                 'vote_count','movie_id',
                                'production_companies', 'cast', 'crew', 'genres']) 

U = encoded.drop(columns=['vote_round'])
v = encoded['vote_round']

v_flat = v.values.ravel()

print(U.dtypes, U.shape)
print(v_flat.shape)

# Converting datatypes for encoded genres
def encode_genres(df): 
    for i in range(3, 24):
        name = U.columns[i]
        U[name] = np.asarray(U[name]).astype('float32')

encode_genres(U)    
from keras.utils import to_categorical
v_flat = to_categorical(v_flat) 

X_train, X_test, y_train, y_test = train_test_split(U, v_flat, test_size=0.25, random_state=9)


#Linear Regression
#enc_two_df = encode_list(test, feature = 'genres')

print(enc_df.shape)
print(enc_df.dtypes)


#encoded = enc_two_df

U_two = encoded_two.drop(columns=['vote_average'])
v_two = encoded_two['vote_average']
v_flat_two = v_two.values.ravel()

print(U_two.dtypes, U_two.shape)
print(v_flat_two.shape)

#v_flat_two = to_categorical(v_flat_two)
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    


X_train, X_test, y_train, y_test = train_test_split(U_two, v_flat_two, test_size=0.25, random_state=9)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

model = LinearRegression()
params = {'fit_intercept':[True, False], 'normalize':[True, False], 'copy_X':[True, False]}
grid_search = GridSearchCV(model, param_grid=params, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params)
print(best_score)

y_pred = grid_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_square =r2_score(y_test, y_pred)

chart_regression(y_pred[:100].flatten(), y_test[:100].flatten(), sort=True)    
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R2 Square', r2_square)

print(y_pred.shape)
print(y_test.shape)

## Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

model_Ridge = Ridge()
#prepare a range of alpha values to test
alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
normalizes = ([True, False])

grid_search_Ridge = GridSearchCV(estimator=model_Ridge,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='neg_mean_squared_error',
                         n_jobs=-1)


grid_search_Ridge.fit(X_train, y_train)
print(grid_search_Ridge.best_params_)
print(grid_search_Ridge.best_score_)

y_pred = grid_search_Ridge.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_square =r2_score(y_test, y_pred)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R2 Square', r2_square)

print(y_pred.shape)
print(y_test.shape)

chart_regression(y_pred[:100].flatten(), y_test[:100].flatten(), sort=True)


##ElasticNet Regression
model_Elastic = ElasticNet()
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

grid_search_elastic = GridSearchCV(estimator=model_Elastic,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='neg_mean_squared_error',
                         n_jobs=-1)
grid_search_elastic.fit(X_train, y_train)

print(grid_search_elastic.best_params_)
print(grid_search_elastic.best_score_)

y_pred = grid_search_elastic.predict(X_test)

chart_regression(y_pred[:100].flatten(), y_test[:100].flatten(), sort=True)

#Neural Network
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
#print(test['genres_cat'])

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

#print(reg.intercept_)
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

















































