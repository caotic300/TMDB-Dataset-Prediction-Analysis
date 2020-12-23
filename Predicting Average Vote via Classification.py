# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:52:43 2020

@author: Husam
"""

import json
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn.preprocessing import MultiLabelBinarizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def encode_list(dataf, feature):
    enc_df = dataf.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(temp.pop(feature)),
                                                     index = dataf.index,
                                                     columns = mlb.classes_))
    return enc_df

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
                              'spoken_languages','status','tagline'
                              ], axis=1) # , 'runtime', , 'genres'


test = convert(movies,  ['genres', 'keywords', 'production_companies', 'cast', 'crew'])

# Removing entries with empty cast/crew/companies
drop = []
for i in test.index:
    if (test['production_companies'][i] == [''] and test['cast'][i] == [''] and 
       test['crew'][i] == ['']): 
       drop.append(i)
test = test.drop(drop, axis = 0)

# Removing entries with a score of 0
mask_avg = (test['vote_average'] != 0)
test = test[mask_avg]
temp = test.copy()

test.shape
test.dtypes

temp['vote_round'] = temp.vote_average.round()
temp = temp.drop(columns = 'vote_average')

countCol = temp.groupby("vote_round")["vote_round"].transform(len)
mask = (countCol >= 10)
temp = temp[mask]

temp.vote_round.value_counts()

mlb = MultiLabelBinarizer(sparse_output=True) 
enc_df = encode_list(temp, feature = 'genres')




encoded = enc_df.copy()

encoded.dtypes
encoded.shape
# Remove unecessary columns
encoded = encoded.drop(columns=['id', 'keywords', 'original_title','movie_id',
                                'production_companies', 'cast', 'crew', 'runtime']) 
encoded.isnull().any()
# Converting datatypes for encoded genres 
for i in range(5, 26):
    name = encoded.columns[i]
    encoded[name] = np.asarray(encoded[name]).astype('float32')

U = encoded.drop(columns=['vote_round'])
v = encoded['vote_round'].astype('int32')

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(v)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_v = onehot_encoder.fit_transform(integer_encoded)

U.shape
v.shape
onehot_v.shape
onehot_v[:10]
v[:10]

encoded['vote_round'].value_counts()

v_flat = v.values.ravel()
v_flat = to_categorical(v_flat) 
v_flat.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(U, onehot_v, test_size=0.25, random_state=5)

X_train.shape
y_train.shape

X_test.shape
y_test.shape

############################################## NEURAL NETWORK ###################################################

from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

callbacks = [EarlyStopping(monitor='loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]

opt = SGD(learning_rate = 0.01, momentum=0.9, decay=0.01)

model = Sequential()
model.add(Dense(25, input_dim = U.shape[1], activation = 'sigmoid')) #Input Layer -> Hidden Layer 1
model.add(Dense(17, activation = 'sigmoid')) # Hidden Layer 1 -> Hidden Layer 2
model.add(Dense(7,  activation = 'softmax')) # Hidden Layer 2 -> Output Layer
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # Choosing Loss & Optimizer 
net = model.fit(X_train, y_train, verbose = 2, epochs = 200)#, callbacks = callbacks, batch_size = 64) # Training data
pred = model.predict(X_test)  # make predictions 
pred = np.argmax(pred,axis=1) # now pick the most likely outcome
y_compare = np.argmax(y_test,axis=1) 

#calculate accuracy
score = metrics.accuracy_score(y_compare, pred) 
print("Accuracy score: {}".format(score))

for key, value in net.history.items() :
    print (key)
    
plt.plot(net.history['loss'])
#plt.plot(net.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


plt.plot(net.history['accuracy'])
plt.title('Accuracy')
plt.ylabel('accruacy')
plt.xlabel('epoch')
plt.legend(['accruacy'], loc='upper left')
plt.show()

# Plotting of Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
cm = confusion_matrix(y_compare, pred)
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
pred[:10]

y_test.shape
print(y_test[:10])
print(pred[:20])
########################################### OTHER MODELS #####################################################

                                    # K-Nearest Neighbors Classification
                                    
# Doesn't work too well with higher dimensiosn as it is more difficult to calculate distances in high dimensions
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm = 'auto',           # Find the best algorithm 
                            metric = 'minkowski',        # Distance Metric E.g. Minkowski
                            n_neighbors = 7,             # 
                            p = 2,                       # Manhattan (1) / Euclidean (2) -  Distance 
                            weights = 'distance')        # distance weighted points 

knn.fit(X_train, y_train)
#print("Predictions form the classifier:")
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


                                    # Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion = 'entropy', max_features = 'auto',
                             n_estimators = 100, n_jobs = (4), oob_score=(True),
                             max_depth=(30)) #min_samples_leaf = 2)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


                                    # Gaussian Naive Bayes
                                    
U = encoded.drop(columns=['vote_round'])
v = encoded['vote_round'].astype('int32')
# run this for naive bayes 
X_train, X_test, y_train, y_test = train_test_split(U, v, test_size=0.25, random_state=3)
 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

                                    # Gaussian Mixture Model 
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(covariance_type = 'tied', n_components= 2, n_init = 10,
                      verbose = 2, verbose_interval = 5, random_state=4).fit(U)
proba = gmm.predict_proba(U)
svm = SVC().fit(proba, v)
y_pred = svm.predict(U)
print("Accuracy Score: ", metrics.accuracy_score(v, y_pred))



"""
            Doesnt work well with dataset 
                                    # KMEANS 
                                    CLUSTERING                    
 # run this for k-means                                     
X = enc_df.copy()
X.dtypes
X = X.drop(columns=['id', 'keywords', 'original_title','movie_id',
                                'production_companies', 'cast', 'crew', 'runtime']) 

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 7, n_init = 15, tol = 1e-4)
y_pred = kmeans.fit(X)
print(metrics.accuracy_score(kmeans, y_pred))

len(y_pred)


X_train.shape
y_train.shape
y_test.shape
"""

