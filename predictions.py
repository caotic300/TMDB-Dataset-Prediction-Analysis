#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:55:43 2020

@author: pedrojose
"""
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # plot formatting
path = "."

filename_read = os.path.join(path,"tmdb_5000_movies.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'])

obj_df = df.select_dtypes(include = ['object'].copy())


                        # Label Encoding

obj_df['genres'] = obj_df['genres'].astype('category')
obj_df['genres_cat'] = obj_df['genres'].cat.codes


#print(obj_df.dtypes)

countCol = obj_df.groupby("genres_cat")["genres_cat"].transform(len)
mask = (countCol >= 10)

x = obj_df[mask]

print(obj_df.shape)
print(x.shape)


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
print(test_df)
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import LeaveOneOut

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
X = y.drop(columns=['genres_cat'])
v = y[['genres_cat']]
y = v.values.ravel()
#print(v.values.ravel())



def plot_confucion_matrix(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(mat, square=True, annot=True ,cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    
def plot_graph(X_train, X_test, y_train, y_test):

    colours = ("r", "b")
    newX = []
    for iclass in range(3):
        newX.append([[], [], []])
    
        for i in range(len(X_train)):
            if y_train[i] == iclass:
                newX[iclass][0].append(X_train[i][0])
                newX[iclass][1].append(X_train[i][1])
                newX[iclass][2].append(sum(X_train[i][2:]))
                colours = ("r", "g", "y")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for iclass in range(3):
            ax.scatter(newX[iclass][0], newX[iclass][1], newX[iclass][2], c=colours[iclass])
        plt.show()
            







#print(X.columns)

#Custom Crossval creates a custom crossvalidation by sending the data
#The data must use a label_col to label the cross validation.
def make_custom_cross_val(data, cv_label):
    myCViterator = []
    for i in range(cv_label):
        trainIndices = data[ data[cv_label] !=i ].index.values.astype(int)
        testIndices =  data[ data[cv_label]== i ].index.values.astype(int)
        myCViterator.append( (trainIndices, testIndices) )




# split the data with 25% in each set to use holdout
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.40, random_state= 42,  shuffle=True)

#Setting knn model
knn_one = KNeighborsClassifier(algorithm = 'auto', 
                            metric = 'minkowski', 
                            n_neighbors = 13,
                            p = 2,
                            weights = 'distance')
# usin g knn alone
knn_one.fit(X_train, y_train)
#y_test_pred = knn.fit(X_test, y_test).predict(X_test)

y_pred = knn_one.predict(X_test)

print("Predictions from the classifier using holdout method:")

print(X.shape, y.shape)
print(accuracy_score(y_test, y_pred))



##Using gridsearch with holdout
#Set grid_params to find best params
knn_two = KNeighborsClassifier()
param_grid = [{'weights': ["uniform", "distance"], 'p': [1, 2], 
               'n_neighbors': [10, 20, 30, 40, 50]}]

#set grid_search to find best_params
grid_search = GridSearchCV(knn_two, param_grid, cv=2, verbose=3)

# usin g knn alone
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
#y_test_pred = knn.fit(X_test, y_test).predict(X_test)

y_pred = grid_search.predict(X_test)

print("Predictions from the classifier using holdout with Gridsearch method:")

# split the data with 25% in each set to use two fold cross  validation
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.35, random_state= 42,  shuffle=True)

knn_three = KNeighborsClassifier(algorithm = 'auto', 
                            metric = 'minkowski', 
                            n_neighbors = 13,
                            p = 2,
                            weights = 'distance')

y_train_pred = knn_three.fit(X_train, y_train).predict(X_train)
y_test_pred = knn_three.fit(X_test, y_test).predict(X_test)
#plot_confucion_matrix(y_test, y_test_pred)

#using two-fold cross_validation
print(X.shape, y.shape)

print(y_train_pred.shape)
print(y_test_pred.shape)

#accuracy_score(stDt.label,ret1)

print(accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred))

print()
knn_four = KNeighborsClassifier()
#using crossvalidation with leaveOneOut
print('using crossvalidation with leaveOneOut\n')
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.35, random_state= 42,  shuffle=True)


grid_search = GridSearchCV(knn_four, param_grid, cv=LeaveOneOut(), verbose=3)
#grid_search = GridSearchCV(knn_four, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print(grid_search.best_params_)
print(grid_search.best_score_)



print(accuracy_score(y_test, y_pred))


from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
from sklearn.impute import SimpleImputer
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

print(X.shape, y.shape)
print(X.isnull().sum())
plt.scatter(X_train[:,0], y_train, color='black')
axis = plt.axis()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
for degre in [1, 3, 5]:
    y_test_pred = PolynomialRegression(degree).fit(X_train, y_train).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree=={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')








