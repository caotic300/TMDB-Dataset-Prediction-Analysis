# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:48:30 2020

@author: Husam
"""
import pandas as pd
import os
import sys

path = "."

filename_read = os.path.join(path,"tmdb_5000_movies.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'])

#movie = df['title']
#print(movie)
#print(df.head(5))
#print(df.dtypes)

#dict = [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]

#print (df.dtypes)

obj_df = df.select_dtypes(include = ['object'].copy())
#print(obj_df.head())


# Label Encoding
obj_df['genres'] = obj_df['genres'].astype('category')
#print (obj_df.dtypes)

obj_df['genres_cat'] = obj_df['genres'].cat.codes


# One Hot Encoding would create too many unique columns

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


#print_full(obj_df['genres_cat'])

sys.stdout = open("genres_cat.txt", "w")
print_full(obj_df['genres_cat'])
sys.stdout.close()














