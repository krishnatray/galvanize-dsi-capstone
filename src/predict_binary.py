# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:38:59 2018

@author: SS
"""

import pandas as pd
from sklearn.externals import joblib
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# import sys
# from sklearn.feature_extraction.text import TfidfVectorizer



def data_pipeline(data_file, tfidf_file):
    data = pd.read_csv(data_file)
    print("Data size:", data.shape)

    X_raw = data.corpus

    # Bag of words model
    saved_tfidf = joblib.load( tfidf_file )
    print(saved_tfidf)

    #
    X_tfidf = saved_tfidf.transform(X_raw).toarray()

    return data, X_tfidf



if __name__ == "__main__":

    input_file = "data/corpus.csv"
    tfifd = "model/tfidf_binary.pkl"
    saved_model =  "model/multiclass.pkl"

    df, X_tfidf = data_pipeline(input_file, tfifd)

    mymodel = joblib.load(saved_model)

    print(mymodel.predict(X_tfidf[:100]))

    #le.inverse_transform(mymodel.predict(X_test))

    # y_pred_proba = model.predict_proba(X_new)

    #print(print(pd.DataFrame(y_pred, y)))
