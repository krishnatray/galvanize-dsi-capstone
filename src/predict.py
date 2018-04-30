# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:38:59 2018

@author: SS
"""
import pandas as pd
import sys
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
model = joblib.load('filename.pkl')

# data pipeline

#importing the processed dataset
file_name="corpus.csv"
data = pd.read_csv(file_name)

if len(sys.argv) > 1 and int(sys.argv[1]) > 0 :
    sample_size = int(sys.argv[1])

    print("Taking first", sample_size, "rows from dataset..")

    # split into train test
    print("Class counts:")
    print(pd.DataFrame(data['is_cardiologist'][:sample_size].value_counts()))
    y = data['is_cardiologist'].values[:sample_size]
    X_train, X_test, y_train, y_test = train_test_split(data.corpus[:sample_size], y,test_size=0.25, random_state=0)
else:
    # split into train test
    print("Class counts:")
    print(pd.DataFrame(data['is_cardiologist'].value_counts()))
    y = data['is_cardiologist'].values
    X_train, X_test, y_train, y_test = train_test_split(data.corpus, y,test_size=0.25, random_state=0)


# Bag of words model
print ("TFIDF..." )


cv = TfidfVectorizer(max_features = 100, stop_words= "english")
