#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:55:43 2020

@author: pedrojose
"""

path = "."

filename_read = os.path.join(path,"tmdb_5000_movies.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'])

obj_df = df.select_dtypes(include = ['object'].copy())
