# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:38:30 2020

@author: Husam
"""
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows', 21)


                        # Read Data    

path = "D:\Computing\Computer Science Year 3\IN3062 Introduction to Artificial Intelligence\TMDB-home"

filename_read = os.path.join(path,"tmdb_5000_movies.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'])

obj_df = df.select_dtypes(include = ['object'].copy())

                        # Label Encoding
"""
obj_df['genres'] = obj_df['genres'].astype('category')
obj_df['genres_cat'] = obj_df['genres'].cat.codes


#print(obj_df.dtypes)

countCol = obj_df.groupby("genres_cat")["genres_cat"].transform(len)
mask = (countCol >= 10)

x = obj_df[mask]

print(obj_df.shape)
print(x.shape)

"""
"""
sys.stdout = open("mask.txt", "w")
print_full(x)
sys.stdout.close()
"""



                        # Encoding inlcuding all features
                        
test_df = df
print(test_df.shape)

test_df['genres'] = test_df['genres'].astype('category')
test_df['genres_cat'] = test_df['genres'].cat.codes

countCol = test_df.groupby("genres_cat")["genres_cat"].transform(len)
mask = (countCol >= 100)

y = test_df[mask]
print(y.shape)

                        # Dropping unecessary columns
                        
print(y.dtypes)

y = y.drop(columns=['genres','homepage','keywords','production_companies',
                    'title','id','original_language',
                    'original_title','overview','production_countries',
                    'release_date','runtime','spoken_languages',
                    'status','tagline'], axis=1)

print(y.dtypes)

                        # Splitting dataset
                        
from sklearn.model_selection import train_test_split

U = y.drop(columns=['genres_cat'])
v = y[['genres_cat']]
v_flatten = v.values.ravel()
#print(v.values.ravel())

X_train, X_test, y_train, y_test = train_test_split(U, v_flatten, test_size=0.25)

"""
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
z = iris.target

print(X.shape)
print(z.shape)
print(v.shape)
print(U.shape)
"""

            # Using KNN with sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


knn = KNeighborsClassifier(algorithm = 'auto', 
                            metric = 'minkowski', 
                            n_neighbors = 13,
                            p = 2,
                            weights = 'uniform')

knn.fit(X_train, y_train)
print("Predictions form the classifier:")

y_pred = knn.predict(X_test)
#print(y_pred)

print(accuracy_score(y_test, y_pred))


            # Naive Bayes with sklearn
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print(accuracy_score(y_test, y_pred))



                        # Plotting Features 

fig, ax1 = plt.subplots()

u = y["genres_cat"]
v1 = y["budget"]
v2 = y["revenue"]

#ax2 = ax1.twinx()

yAxis = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 
         1.2, 0.4, 0.6, 0.8, 2.0,
         2.2, 2.4, 2.6, 2.8, 3.0]

ax1.set_xlabel("Category")
ax1.set_ylabel("Budget")
#ax1.set_yticks(yAxis)
ax1.plot(u, v1, 'g.')

#ax2.plot(u, v2, 'b-')

#y[['ISP.MI','Ctrv']].plot()


fig, ax2 = plt.subplots()


ax2.set_xlabel("Category")
ax2.set_ylabel("Revenue")
#ax1.set_yticks(yAxis)
ax2.plot(u, v2, 'b.')

#ax2.plot(u, v2, 'b-')















































