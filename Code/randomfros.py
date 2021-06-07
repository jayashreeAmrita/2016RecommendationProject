# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
import numpy as np
from numpy import genfromtxt
import csv
import psycopg2

def predict():
    with open('cluster_result.csv', 'rb') as f:
        reader = csv.reader(f)
        cluster = list(reader)

    with open('rms_cluster_result.csv', 'rb') as f:
        reader = csv.reader(f)
        rms_cluster = list(reader)

    indx = genfromtxt('test_index.csv', delimiter=',',dtype=int)
    centroid = genfromtxt('centroid.csv', delimiter=',',dtype=int)
    test_compute = genfromtxt('test1.csv', delimiter=',',dtype=int)



    readIn = pd.read_csv('train1.csv', sep=',',header=None)
    l1=list(readIn.columns.values)
    test = pd.read_csv('test1.csv', sep=',',header=None)
    p = readIn[l1[len(l1)-1]]
    readIn = readIn.drop(l1[len(l1)-1], axis=1)
    forest = RandomForestClassifier(n_estimators=10,max_depth=10,max_features=2000,max_leaf_nodes=25,n_jobs=2)
    forest = forest.fit(readIn, p)
    joblib.dump(forest, 'fo', compress=5)
    result = forest.predict(test)
    print result


    scores=cross_val_score(forest,readIn, p,cv=10)
    accute=(" %0.2f (+/- %0.2f)"
          % (scores.mean(), scores.std()*2))
    print scores.mean()
    # conn = psycopg2.connect(database="postgres", user="postgres", password="as", host="127.0.0.1", port="5432")
    # cur = conn.cursor()
    # cur.execute("INSERT INTO accu (val) VALUES (%s)", (scores.mean()))
    # conn.commit()
    # conn.close()

    pred=np.zeros(shape=(1,len(result)), dtype=np.int)
    obt=np.zeros(shape=(1,len(result)), dtype=np.int)
    pred1 = np.zeros(shape=(indx[1]-indx[0],3952), dtype=np.float)


    i_indx=0
    k_indx=0
    for i in range (indx[0],indx[1]):
        for j in range (0,len(rms_cluster)):
            if(str(i) in rms_cluster[j]):
                k_indx=0
                for k in centroid[j]:
                    pred1[i_indx][k_indx]=k
                    k_indx+=1


    print pred1
    #
    #
    #
    # a=pred1.shape
    # test_total=0
    # sum_rat=0
    # dif_pr=0
    # for i in range(0,a[0]):
    #     for j in range (0,a[1]):
    #         if(test_compute[i][j]!=0):
    #             test_total+=1
    #             dif_pr= pow((pred1[i][j]-test_compute[i][j]),2)
    #             sum_rat+=dif_pr
    #
    #
    # sq_sum_rat= sqrt((1/test_total)*sum_rat)
    # print sq_sum_rat



    # print "RMSE :"+str( (sqrt(mean_squared_error(test_compute, pred1)))+.1)

    # conn = psycopg2.connect(database="postgres", user="postgres", password="as", host="127.0.0.1", port="5432")
    # cur = conn.cursor()
    # cur.execute("INSERT INTO name (id, nam) VALUES (%s,%s)", (a - 1, str(b)))
    # conn.commit()
    # conn.close()





predict()






