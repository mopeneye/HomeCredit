#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import math


import importlib
import googletrans
importlib.reload(googletrans)
from googletrans import Translator
import pandas as pd
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option("display.max_rows", None, "display.max_columns", None)
translator = Translator()

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# matplotlib and seaborn for plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap

from scipy import stats
from scipy.stats import norm, skew

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer



def read(csv):
    df = pd.read_csv(csv)
    return df



def class_synchronization(train, test):
    train_unique =train.select_dtypes('object').nunique() 
    test_unique = test.select_dtypes('object').nunique()
    check = (train_unique - test_unique )
    diff = check[check>0].index
    for i in range(0, len(diff)):
        a =train[diff [i]].value_counts().index
        b =test[diff [i]].value_counts().index
        train.drop(train[train[diff [i]]==a.difference(b)[0]].index, inplace = True, axis =0)
        train.drop(test[test[diff [i]]==a.difference(b)[0]].index, inplace = True, axis =0)





def collect_features(train, test, key):
    for df in (train, test):
        list_ = list(df.columns[df.columns.str.contains(key,regex=True)].sort_values(ascending = True))
        df[key] = df[list_].sum(axis=1)
        df.loc[(df[key]!=0), [key]] = 1
        df.drop(list_, inplace = True, axis =1)





def drop_group_of_features(train, test, key):
    for df in (train, test):
        list_ = list(df.columns[df.columns.str.contains(key,regex=True)].sort_values(ascending = True))
        df.drop(list_, inplace = True, axis =1)





def collect_features_and_binary(train, test, key):
    for df in (train, test):
        list_ = list(df.columns[df.columns.str.contains(key,regex=True)].sort_values(ascending = True))
        df[key] = df[list_].sum(axis=1)
        df[key+"_Binary"] = df[key]
        df.loc[(df[key+"_Binary"]!=0), [key+"_Binary"]] = 1 
        df.drop(list_, inplace = True, axis =1)





def row_cross_replace(train, test,feature1,feature2, key1, key2):
    for df in(train, test):
        df.loc[(df[feature1]==key1), [feature2]] = key2





def row_rename(train, test,feature, key, class_list):
    for df in (train, test):
        for j in class_list:
            df.loc[(df[feature]==j), [feature]] = key





def occupation_rare():
    White_collar = ["IT staff", "HR staff", "Accountants","Managers", "Core staff","Medicine staff", "Realty agents"]
    Blue_collar= ["Security staff", "Cooking staff", "Cleaning staff", "Secretaries","Sales staff","High skill tech staff",
                  "Waiters/barmen staff","Private service staff"]
    Laborers = ["Laborers", "Low-skill Laborers"]
    return White_collar,Blue_collar, Laborers





def personal_asset(train,test,feature1, feature2,new_feature_name):
    for df in (train, test):
        for feature in (feature1, feature2):
            df.loc[(df[feature]=="N"), [feature]] = 0
            df.loc[(df[feature]=="Y"), [feature]] = 1
            df[feature] = df[feature].astype(int)
        df[new_feature_name] = df[feature1] * df[feature2]
        df[new_feature_name] = df[new_feature_name].astype(int)




def education_years(train, test):
    map_ = {"Lower secondary":9, 'Secondary / secondary special':12, 'Incomplete higher':14, 
            'Higher education':18, 'Academic degree':22}
    for df in (train,test):
            df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].replace(map_)




def days_features(train,test):
    for df in (train,test):
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        Days = list(df.columns[df.columns.str.contains('DAYS',regex=True)])
        for col in Days:
            df[col] = df[col] / -365





def outlier_thresholds(df, feature):  
    quartile1 = df[feature].quantile(0.05)
    quartile3 = df[feature].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit





def has_outliers(train):   
    updated_outliers = []
    numerical_feats = train.dtypes[train.dtypes != "object"].index
    num_feats =[col for col in train[numerical_feats] if len(train[list(numerical_feats)][col].unique())> 10 and col not in "SK_ID_CURR"]
    for col in num_feats:
        low_limit, up_limit = outlier_thresholds(train, col)
        if train[(train[col] > up_limit) | (train[col] < low_limit)].any(axis=None):     
            updated_outliers.append(col)
    return updated_outliers




def outliers_update(train, test):   
    updated_outliers = has_outliers(train)
    low, up = outlier_thresholds(train, updated_outliers)
    for df in (train, test):
        for col in updated_outliers:
            low, up = outlier_thresholds(df, col)
            df[col] = df[col].apply(lambda x: low if (x<low) else x and up if (x>up) else x)





def log_transformation(train, test):
    log_list =['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
        'DAYS_EMPLOYED','DAYS_REGISTRATION','APARTMENTS_MEDI','COMMONAREA_MEDI',
         'LANDAREA_MEDI','NONLIVINGAREA_MEDI','DAYS_LAST_PHONE_CHANGE', 'CREDIT_INCOME_PERCENT',
        'DAYS_EMPLOYED_PERCENT','INCOME_PER_PERSON','ANNUITY_INCOME_PERCENT']
    for df in [train, test]:
        for i in log_list:
            df[i] = np.log1p(df[i])




def generate_binary_from_numerical(train, test, feature):
    for df in (train, test):
        df[feature+"_Binary"] = df[feature]
        df.loc[(df[feature+"_Binary"]!=0), [feature+"_Binary"]] = 1 





def new_features_from_EXT_features(train, test):
    
    #https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
    
    # Make a new dataframe for polynomial features
    poly_features = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
    poly_features_test = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    # imputer for handling missing values
    imputer = SimpleImputer(strategy = 'median')

    poly_target = poly_features['TARGET']
    poly_features = poly_features.drop(columns = ['TARGET'])

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.fit_transform(poly_features_test)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree = 3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)

    poly_features = pd.DataFrame(poly_features, 
                                 columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                               'EXT_SOURCE_3', 'DAYS_BIRTH']))
    # Add in the target
    poly_features['TARGET'] = train['TARGET']

    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test, 
                                      columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                    'EXT_SOURCE_3', 'DAYS_BIRTH']))
    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = train['SK_ID_CURR']
    app_train_poly  = train.merge(poly_features[poly_features.columns.difference(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                        'EXT_SOURCE_3', 'DAYS_BIRTH'])], 
                        on = 'SK_ID_CURR', how = 'left')

    # Merge polnomial features into testing dataframe
    poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
    app_test_poly  = test.merge(poly_features_test[poly_features_test.columns.difference(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH'])], 
                      on = 'SK_ID_CURR', how = 'left')

    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)
    app_train_poly["TARGET"] = train["TARGET"]
    return app_train_poly, app_test_poly





def new_features_domain_knowledge(train, test):
    for df in (train, test):
        df["REGION_RATING"] =(df["REGION_RATING_CLIENT"]**2)*df["REGION_POPULATION_RELATIVE"]
        df["OWN_CAR_AGE"] = (1/df["OWN_CAR_AGE"])
        df.loc[df["OWN_CAR_AGE"]== np.inf, ["OWN_CAR_AGE"]] =0
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['CREDIT_INCOME_PERCENT'] = df['AMT_INCOME_TOTAL'] / train['AMT_CREDIT']
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['CREDIT_GOOD_PERCENT'] = df["AMT_GOODS_PRICE"]  / df["AMT_CREDIT"]
        df["RATIO_OBS_60_to_OBS_30"] = df["OBS_60_CNT_SOCIAL_CIRCLE"]/df["OBS_30_CNT_SOCIAL_CIRCLE"]
        df["RATIO_DEF_60_to_DEF_30"] = df["DEF_60_CNT_SOCIAL_CIRCLE"]/df["DEF_30_CNT_SOCIAL_CIRCLE"]



def columns_dtypes(df):
    categorical_feats = df.dtypes[df.dtypes =="object"].index
    numerical_feats = df.dtypes[df.dtypes !="object"].index
    return categorical_feats, numerical_feats




def drop_features(train, test):
    drop_list = ["HOUR_APPR_PROCESS_START", "LIVE_CITY_NOT_WORK_CITY", "WEEKDAY_APPR_PROCESS_START", "REGION_RATING_CLIENT_W_CITY",
    "ORGANIZATION_TYPE","NAME_TYPE_SUITE","LIVINGAPARTMENTS_MEDI","LIVINGAREA_MEDI","ELEVATORS_MEDI","FLOORSMIN_MEDI",
    "BASEMENTAREA_MEDI","FLOORSMAX_MEDI","ENTRANCES_MEDI", 'APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG',
    'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG','FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
    'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 
    'BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE','ELEVATORS_MODE', 'ENTRANCES_MODE',
    'FLOORSMAX_MODE', 'FLOORSMIN_MODE','LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE', 
    'NONLIVINGAREA_MODE', 'FONDKAPREMONT_MODE','HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE',
    "REGION_POPULATION_RELATIVE", "CNT_FAM_MEMBERS",'FLAG_MOBIL','FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 
    'FLAG_PHONE','FLAG_EMAIL',"LIVE_REGION_NOT_WORK_REGION", "YEARS_BEGINEXPLUATATION_MEDI"]    
    for df in (train, test):
        df.drop(drop_list, inplace = True, axis = 1)


def test_model(X_train, X_test, y_train, y_test):
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    print('LGBM accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)), '\n')
    print(classification_report(y_test, y_pred))
    lgbm_cm = metrics.confusion_matrix( y_test,y_pred, [1,0])
    sns.heatmap(lgbm_cm, annot=True, fmt='.2f',xticklabels = ["Risk", "No Risk"] , yticklabels = ["Risk", "No Risk"] )
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title('LightGBM Model')
    
    
def feature_importance(X_train, X_test, y_train, y_test):
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)
    Importance = pd.DataFrame({'Importance': lgbm.feature_importances_ * 100,'Feature': X_train.columns}).sort_values(by="Importance", ascending =False).head(25)
    fig=plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    sns.barplot(x=Importance.Importance,y='Feature',data=Importance,color='blue',ax=ax1)
    plt.xticks(Importance.index,  rotation=60)
    plt.show()


def one_hot_encoder(train, test, categorical_cols, nan_as_category=True):
    original_columns_train = list(train.columns)
    original_columns_test = list(test.columns)
    train = pd.get_dummies(train, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    test =  pd.get_dummies(test, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in train.columns if c not in original_columns_train]
    return train, test, new_columns


def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1

    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


def implement_robust_function(train, test):
    Robust =['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 
    'DAYS_EMPLOYED','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE','EXT_SOURCE_1', 
    'EXT_SOURCE_2', 'EXT_SOURCE_3','APARTMENTS_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI','LANDAREA_MEDI', 
    'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI','DAYS_LAST_PHONE_CHANGE', 'ANNUITY_INCOME_PERCENT', 
    'CREDIT_INCOME_PERCENT','CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT', 'INCOME_PER_PERSON','CREDIT_GOOD_PERCENT']

    for df in (train, test):
        for col in Robust:
            df[col] = robust_scaler(df[col])






