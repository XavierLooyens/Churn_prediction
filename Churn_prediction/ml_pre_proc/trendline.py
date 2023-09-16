#Imports
import pandas as pd
import numpy as np

def trendline_merger(transactions_0_df,transactions_df,train_0_df,train_df):
    #concat vertically
    #drop duplicates
    #merging tables
    transactions_df_concat = pd.concat([transactions_0_df, transactions_df], axis=0)
    train_df_concat = pd.concat([train_0_df, train_df], axis=0)

    transactions_df_global = transactions_df_concat.drop_duplicates(keep="first")
    train_df_global = train_df_concat.drop_duplicates(keep="first")

    transactions_train_df = pd.merge(train_df_global, transactions_df_global, on='msno', how='left')

    return transactions_train_df

def trendline_preproc(transactions_train_df):
    #convert to datetime
    # Sort the DataFrame by 'msno' and 'transaction_date'
    transactions_train_df.loc[:,'transaction_date'] = pd.to_datetime(transactions_train_df['transaction_date'], format='%Y%m%d')
    transactions_train_df.loc[:,'membership_expire_date'] = pd.to_datetime(transactions_train_df['membership_expire_date'], format='%Y%m%d')

    transactions_train_df = transactions_train_df.sort_values(by=['msno', 'transaction_date'], ascending=[True, True],)

    return transactions_train_df

def trendline_compute(transactions_train_df):
    #compute previous period churns
    transactions_train_df['transaction_date_-1'] = transactions_train_df.groupby('msno')['transaction_date'].shift(+1)
    transactions_train_df['membership_expire_date_-1'] = transactions_train_df.groupby('msno')['membership_expire_date'].shift(+1)
    transactions_train_df['period_0'] = transactions_train_df['membership_expire_date_-1']-transactions_train_df['transaction_date']
    transactions_train_df['period_0_churn'] = np.where(transactions_train_df['period_0'] <= pd.Timedelta(days=-30), 1, np.where(transactions_train_df['period_0'].isna(), np.nan, 0))

    transactions_train_df['transaction_date_-2'] = transactions_train_df.groupby('msno')['transaction_date'].shift(+2)
    transactions_train_df['membership_expire_date_-2'] = transactions_train_df.groupby('msno')['membership_expire_date'].shift(+2)
    transactions_train_df['period_-1'] = transactions_train_df['membership_expire_date_-2']-transactions_train_df['transaction_date_-1']
    transactions_train_df['period_-1_churn'] = np.where(transactions_train_df['period_-1'] <= pd.Timedelta(days=-30), 1, np.where(transactions_train_df['period_-1'].isna(), np.nan, 0))


    transactions_train_df['transaction_date_-3'] = transactions_train_df.groupby('msno')['transaction_date'].shift(+3)
    transactions_train_df['membership_expire_date_-3'] = transactions_train_df.groupby('msno')['membership_expire_date'].shift(+3)
    transactions_train_df['period_-2'] = transactions_train_df['membership_expire_date_-3']-transactions_train_df['transaction_date_-2']
    transactions_train_df['period_-2_churn'] = np.where(transactions_train_df['period_-2'] <= pd.Timedelta(days=-30), 1, np.where(transactions_train_df['period_-2'].isna(), np.nan, 0))


    transactions_train_df['transaction_date_-4'] = transactions_train_df.groupby('msno')['transaction_date'].shift(+4)
    transactions_train_df['membership_expire_date_-4'] = transactions_train_df.groupby('msno')['membership_expire_date'].shift(+4)
    transactions_train_df['period_-3'] = transactions_train_df['membership_expire_date_-4']-transactions_train_df['transaction_date_-3']
    transactions_train_df['period_-3_churn'] = np.where(transactions_train_df['period_-3'] <= pd.Timedelta(days=-30), 1, np.where(transactions_train_df['period_-3'].isna(), np.nan, 0))


    transactions_train_df['transaction_date_-5'] = transactions_train_df.groupby('msno')['transaction_date'].shift(+5)
    transactions_train_df['membership_expire_date_-5'] = transactions_train_df.groupby('msno')['membership_expire_date'].shift(+5)
    transactions_train_df['period_-4'] = transactions_train_df['membership_expire_date_-5']-transactions_train_df['transaction_date_-4']
    transactions_train_df['period_-4_churn'] = np.where(transactions_train_df['period_-4'] <= pd.Timedelta(days=-30), 1, np.where(transactions_train_df['period_-4'].isna(), np.nan, 0))


    transactions_train_df['transaction_date_-6'] = transactions_train_df.groupby('msno')['transaction_date'].shift(+6)
    transactions_train_df['membership_expire_date_-6'] = transactions_train_df.groupby('msno')['membership_expire_date'].shift(+6)
    transactions_train_df['period_-5'] = transactions_train_df['membership_expire_date_-6']-transactions_train_df['transaction_date_-5']
    transactions_train_df['period_-5_churn'] = np.where(transactions_train_df['period_-5'] <= pd.Timedelta(days=-30), 1, np.where(transactions_train_df['period_-5'].isna(), np.nan, 0))

    return transactions_train_df

def trendline_is_churn(transactions_train_df):
    #only keeping rows with the latest transactions
    #removing unnecessary columns
    latest_transaction_indexes = transactions_train_df.groupby('msno')['transaction_date'].idxmax()
    latest_transactions_with_churn = transactions_train_df.loc[latest_transaction_indexes]

    trendline_df = latest_transactions_with_churn.drop(axis=1, columns=[
        'transaction_date_-1',
        'membership_expire_date_-1',
        'period_0',
        'transaction_date_-2',
        'membership_expire_date_-2',
        'period_-1',
        'transaction_date_-3',
        'membership_expire_date_-3',
        'period_-2',
        'transaction_date_-4',
        'membership_expire_date_-4',
        'period_-3',
        'transaction_date_-5',
        'membership_expire_date_-5',
        'period_-4',
        'transaction_date_-6',
        'membership_expire_date_-6',
        'period_-5',
        'is_churn',
        'payment_method_id',
        'payment_plan_days',
        'plan_list_price',
        'actual_amount_paid',
        'is_auto_renew',
        'transaction_date',
        'membership_expire_date',
        'is_cancel'
    ])

    return trendline_df
