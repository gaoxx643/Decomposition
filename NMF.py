#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')

def data_trans(data):
    df = data.loc[:,['phone','week','ord_num']].pivot_table(index=['phone'],columns=['week'],values=['ord_num']).replace(np.nan,0)
    return df

def nmf_model(train_df):
    vectorizer = TfidfTransformer()
    X = vectorizer.fit_transform(np.array(train_df)).toarray() #tf-idf transform  
    model = NMF(n_components=9, init='random', random_state=0) #tf-idf NMF fit
    W = model.fit_transform(X)
    H = model.components_
    return model,W 

def train_result(train_df,W,rawdata_train):
    train_df['class'] = np.argmax(W, axis=1) #return class
    class_data = train_df.reset_index().loc[:,['phone','class']].merge(rawdata_train.loc[:,['phone','week','ord_num','is_remain']], left_on='phone', right_on='phone', how='inner').drop_duplicates() 
    class_data.columns = ['phoneA','phoneB','class','week','ord_num','is_remain']
    
    class_silent = class_data.groupby('class').agg({'phoneA':'count','is_remain': 'sum'}).reset_index() #static remain_rate
    class_silent['remain_rate'] = class_silent['is_remain']/class_silent['phoneA']
    class_silent = class_silent.round(3)
    
    classRsult_temp1 = class_data.loc[:,['class','week','ord_num']].merge(class_silent.loc[:,['class','remain_rate']],left_on = 'class',right_on = 'class',how = 'inner').set_index('remain_rate').sort_index()
    classRsult_temp2 = classRsult_temp1.pivot_table(index=['remain_rate'],columns=['week'],values=['ord_num']) 
    classResult = pd.concat([classRsult_temp2.reset_index()['ord_num'],classRsult_temp2.reset_index()['remain_rate']], axis = 1).set_index('remain_rate') #remain_rate,week,avg(ord_num)
    
    trainClass_temp = class_data.loc[:,['class','phoneA']].drop_duplicates() 
    trainClass = trainClass_temp.merge(class_silent.loc[:,['class','remain_rate']],left_on = 'class',right_on = 'class',how = 'inner').drop(columns=['class']).set_index('remain_rate') #remain_are,phoneA
    
    return classResult,trainClass,class_silent

def pre_result(test_df,class_silent):
    Y = np.array(test_df)
    Y_label = model.transform(Y)
    test_df['class'] = np.argmax(Y_label, axis=1)
    preClass_temp = pd.concat([test_df.reset_index()['phone'],test_df.reset_index()['class']], axis = 1)
    preClass = preClass_temp.merge(class_silent.loc[:,['class','remain_rate']],left_on = 'class',right_on = 'class',how = 'inner').drop(columns=['class']).set_index('remain_rate') #remain_rate,phone
    return preClass
    
if __name__ == '__main__':
    train_data = pd.read_csv('train_data.csv',delimiter = '\t')
    pre_data = pd.read_csv('pre_data.csv',delimiter = '\t')
    trainClass = pd.DataFrame()  
    classResult = pd.DataFrame()
    preClass = pd.DataFrame() 
    
    for i in range(1,365):
        try:
            rawdata_train = train_data[train_data['city_id'] == i]
            rawdata_pre = pre_data[pre_data['city_id'] == i]
            train_df = data_trans(rawdata_train)
            model,W = nmf_model(train_df)
            class_Result,train_Class,class_silent = train_result(train_df,W,rawdata_train)
            class_Result['city_id'] = i
            classResult = pd.concat([classResult,class_Result])
            train_Class['city_id'] = i
            trainClass = pd.concat([trainClass,train_Class])
            test_df = data_trans(rawdata_pre)
            pre_Class = pre_result(test_df,class_silent)
            pre_Class['city_id'] = i
            preClass = pd.concat([preClass,pre_Class])
        except ValueError:
            pass 
    trainClass.to_csv('trainClass.csv',header=0)
    classResult.to_csv('classResult.csv',header=0,float_format='%.3f')
    preClass.to_csv('preClass.csv',header=0)
    
