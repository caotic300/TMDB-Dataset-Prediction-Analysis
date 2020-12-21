# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:23:30 2020

@author: Husam
"""

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
#print(movies.dtypes)
#print(movies.shape)

#mask = (movies['vote_average'] != 0)    # Remove any entries with 0 as vote avg
#movies = movies[mask]
#print(movies.shape)

test = convert(movies,  ['genres', 'keywords', 'production_companies', 'cast', 'crew'])
print(test[:5])

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

"""
test = test[test['production_companies'].map(lambda i: len(i)) > 0]
test = test[test['cast'].map(lambda i: len(i)) > 0]
test = test[test['crew'].map(lambda i: len(i)) > 0]

test = test[(test['production_companies'].str.len() != 0) or
            (test['cast'].str.len() != 0) or
            (test['crew'].str.len() != 0)]

test = test[~test.production_companies.str.len().eq(0)]
test = test[~test.crew.str.len().eq(0)]
test = test[~test.cast.str.len().eq(0)]


test.to_csv(r'D:\Computing\Computer Science Year 3\IN3062 Introduction to Artificial Intelligence\TMDB-home\test.csv',
            index = False)

print(test['genres'])


#test['genres'] = test['genres'].astype('category')
#test['genres_cat'] = test['genres'].cat.codes


from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()

test['genres_cat'] = test['genres'] 
#test['genres_cat'] = test['genres_cat'].apply(vec.fit_transform(test['genres_cat']))
 
for index, row in test.iterrows():
    row['genres_cat'] = vec.fit_transform(row['genres_cat'])

for i in test.index: 
  #print(df.loc[i, "Name"], df.loc[i, "Age"]) 
  test['genres_cat'][i] = vec.fit_transform(test['genres_cat'][i])
  
print(test['genres_cat'])
"""

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
    
#enc_df = encode_list(temp, feature = 'genres')
enc_df = encode_list(temp, feature = 'production_companies')
#enc_df = encode_list(temp, feature = 'cast')
#enc_df = encode_list(temp, feature = 'crew')


print(enc_df.shape)
print(enc_df.dtypes)

encoded = enc_df.copy()

from sklearn.model_selection import train_test_split

# Remove relavant features 
encoded = encoded.drop(columns=['id', 'keywords', 'original_title',
                                 'vote_count','movie_id',
                                'genres', 'cast', 'crew']) 
                                # , 'genres', 'cast', 'crew', 'production_companies'


U = encoded.drop(columns=['vote_round'])
v = encoded['vote_round']
v_flat = v.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(U, v_flat, test_size=0.25, random_state=9)

# Predicting vote_average based on prodcution companies etc. 
# Example of some models that may be useful

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm = 'auto', 
                            metric = 'minkowski', 
                            n_neighbors = 6,
                            p = 2,
                            weights = 'uniform')

knn.fit(X_train, y_train)
#print("Predictions form the classifier:")
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))






























































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

















































