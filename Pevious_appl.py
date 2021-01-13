# imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# To display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None)




def Last_prev_prepare():
    df = pd.read_csv(r'/data\previous_application.csv')

    Date_cols = list(df.columns[df.columns.str.contains('DAYS', regex=True)])
    # # To convert date values to absolute form
    df.loc[:, Date_cols] = np.absolute(df[Date_cols])
    # # To select last previous application
    temp_df = df.groupby('SK_ID_CURR').agg({'DAYS_DECISION': 'min'})
    # Drop duplicates
    df = df.drop_duplicates(subset=['SK_ID_CURR', 'DAYS_DECISION'])

    temp_df = temp_df.reset_index()

    df = pd.merge(df, temp_df, how='inner', left_on=['SK_ID_CURR', 'DAYS_DECISION'],
                  right_on=['SK_ID_CURR', 'DAYS_DECISION'])

    return df

def Drop_columns(df):
    drop_list = ['FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NAME_TYPE_SUITE', 'WEEKDAY_APPR_PROCESS_START',
             'HOUR_APPR_PROCESS_START', 'RATE_INTEREST_PRIVILEGED', 'AMT_GOODS_PRICE', 'DAYS_LAST_DUE', 'SELLERPLACE_AREA',
             'RATE_DOWN_PAYMENT', 'NAME_CASH_LOAN_PURPOSE', 'NAME_GOODS_CATEGORY', 'NAME_SELLER_INDUSTRY', 'PRODUCT_COMBINATION']

    df.drop(drop_list, axis=1, inplace=True)

    return df

def Join_aggregated_lastprevappl(df1, df2):
    df1['SK_ID_CURR'] = df1.index
    Joined_df = df2.join(df1, on='SK_ID_CURR', how='left', lsuffix="_LAST_APPL").reset_index()
    Joined_df.drop(['index', 'SK_ID_CURR_LAST_APPL'], axis=1, inplace=True)

    return Joined_df

def one_hot_encoderr(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def Previous_applications():
    prev = pd.read_csv(r'/data\previous_application.csv')
    prev = Drop_columns(prev)
    # Convert NAME_CONTRACT_STATUS
    prev['NAME_CONTRACT_STATUS'] = prev['NAME_CONTRACT_STATUS'].replace('Canceled', 'Refused')
    prev['NAME_CONTRACT_STATUS'] = prev['NAME_CONTRACT_STATUS'].replace('Unused offer', 'Approved')
    # Operation after Rare Analyzer
    prev.loc[(prev['NAME_CONTRACT_TYPE'] == 'XNA'), 'NAME_CONTRACT_TYPE'] = 'Cash loans'
    prev.loc[(prev['CODE_REJECT_REASON'].apply(
        lambda x: x in ['XNA', 'LIMIT', 'SCO', 'SCOFR', 'SYSTEM', 'VERIF'])), 'CODE_REJECT_REASON'] = 'HC'
    prev.loc[(prev['NAME_PORTFOLIO'] == 'Cars'), 'NAME_PORTFOLIO'] = 'Cards'
    prev.loc[(prev['CHANNEL_TYPE'].apply(
        lambda x: x in ['Car dealer', 'Channel of corporate sales'])), 'CHANNEL_TYPE'] = 'Country-wide'

    # Prepare Last Previous Applications
    df_last_prev_appl = Last_prev_prepare(num_rows)
    # Drop columns for Min data
    df_last_prev_appl = Drop_columns(df_last_prev_appl)
    prev, cat_cols = one_hot_encoderr(prev, nan_as_category=True)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    # prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # ADD FEATURES
    # Down payment status
    prev['DOWN_PAYMENT_STATUS'] = np.where(prev['AMT_DOWN_PAYMENT'] > 0, 1, 0)
    # Value asked / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['sum', 'mean'],
        'AMT_APPLICATION': ['mean', 'sum'],
        'AMT_CREDIT': ['mean', 'sum'],
        'APP_CREDIT_PERC': ['mean', 'var'],
        'DOWN_PAYMENT_STATUS': ['sum', 'mean'],
        'AMT_DOWN_PAYMENT': ['mean'],
        'DAYS_DECISION': ['mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'RATE_INTEREST_PRIMARY': 'mean'}

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev

    # Prepare Last Previous Applications
    df_last_prev_appl = Last_prev_prepare()

    # Drop columns for Min data
    df_last_prev_appl = Drop_columns(df_last_prev_appl)

    Joined_df = Join_aggregated_lastprevappl(prev_agg, df_last_prev_appl)

    return Joined_df

Joined_df = Previous_applications()