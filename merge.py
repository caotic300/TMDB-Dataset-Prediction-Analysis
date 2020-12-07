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
        if (c == 'cast' or 'crew'): 
            for index,i in zip(df.index,df[c]):
                limit = 10
                if len(i) < 10:
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

path1 = "."
filename_read = os.path.join(path1,"tmdb_5000_movies.csv")
movie = pd.read_csv(filename_read,na_values=['NA','?'])

path2 = "."
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
"""

test.to_csv(r'.test.csv',
            index = False)

print(test['genres'])





































