#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 23:36:38 2018

@author: stephanosarampatzes
"""

import pandas as pd

# yelp reviews dataset was too big for my laptop, so I did split it to smaller 
# chunks of data from terminal. I did chose xaa.csv and xac.csv
#
# After a few manipulations to structure I concatenated them and saved as new
# csv file

df = pd.read_csv('xaa.csv', encoding = 'utf-8', engine = 'python', error_bad_lines = False)
df = df.drop(['review_id', 'user_id', 'business_id', 'date','useful', 'funny', 'cool'], axis = 1)

df2 = pd.read_csv('xac.csv', encoding = 'utf-8', engine = 'python', error_bad_lines = False, header = None)
df2 = df2.drop([0, 1, 2, 4, 6, 7, 8], axis = 1)
df2 = df2[1:]
df2.columns = ['stars', 'text']

new_df = pd.concat([df,df2]).reset_index(drop = True)
new_df.to_csv('yelp800.csv', encoding='utf-8', index = False)

del [df, df2]

import gc
gc.collect()
