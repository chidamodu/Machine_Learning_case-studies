import pandas as pd
import numpy as np

def df_necessary(df,col_names, to_drop = False):
    '''
    to return the dataframe with only the necessary columns
    '''
    if to_drop:
        return df.drop(col_names,axis=1)
    return df[col_names]

def dummify(df, cols, constant_and_drop=False):
    '''
        Given a dataframe, for all the columns which are not numericly typed already,
        create dummies. This will NOT remove one of the dummies which is required for
        logistic regression (in such case, constant_and_drop should be True).
        To run outside of class
    '''
    df = pd.get_dummies(df, columns=cols, drop_first=constant_and_drop)
    if constant_and_drop:
        const = np.full(len(df), 1)
        df['constant'] = const
    return df

def time_between_user_event_created(df):
    df['time_bet_user_eventcreated'] = (df.event_created - df.user_created)
    df = df.drop(['user_created','event_created'], axis=1)
    return df
