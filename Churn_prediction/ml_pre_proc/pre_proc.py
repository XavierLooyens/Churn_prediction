#Imports
import pandas as pd
import numpy as np

def transactions_preproc(transactions_df):
    ###### converting transaction_date and membership_date to datetime object
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'], format='%Y%m%d')
    transactions_df['membership_expire_date'] = pd.to_datetime(transactions_df['membership_expire_date'], format='%Y%m%d')

    # removing duplicates and leaving only the latest transaction date
    transactions_df_lt = transactions_df.loc[transactions_df.groupby('msno').transaction_date.idxmax()]

    # creating remaining plan duration column
    transactions_df_lt['remaining_plan_duration'] = transactions_df_lt['membership_expire_date'] - transactions_df_lt['transaction_date']

    # creating discount column
    transactions_df_lt['is_discount'] = transactions_df_lt.apply(lambda x: '0' if (x['actual_amount_paid'] -x['plan_list_price'])>=0 else '1', axis=1)

    return transactions_df_lt

def user_logs(user_logs_df,transactions_df_lt):
    # Converting date into datetime object
    user_logs_df['date'] = pd.to_datetime(user_logs_df['date'], format='%Y%m%d')

    # create new dataframe from transactions table with only msno and latest transaction date
    latest_transactions_per_msno = transactions_df_lt[['msno', 'transaction_date']]

    # Merge user logs with latest transaction date
    merged_df = user_logs_df.merge(latest_transactions_per_msno, on='msno', how='left')

    # drop msno's where transaction date is NaN
    merged_df = merged_df.dropna(subset=['transaction_date'])

    # removing rows where the user log data is before the last transaction date
    user_logs_atd= merged_df.loc[merged_df['date']>=merged_df['transaction_date']]

    # removing data column
    user_logs_atd = user_logs_atd.drop(columns=['date','transaction_date'])

    # groupby msno and summing all values
    user_logs_atd = user_logs_atd.groupby('msno').sum()

    return user_logs_atd

def merger(transactions_df_lt,user_logs_atd,members_df,trendline_df):
    # Merging transactions, user logs and members data to Train dataframe
    train_df = train_df.merge(transactions_df_lt, on='msno', how='left')
    train_df = train_df.merge(user_logs_atd, on='msno', how='left')
    train_df = train_df.merge(members_df, on='msno', how='left')
    train_df = train_df.merge(trendline_df, on='msno', how='left')

    return train_df

def feautures_eng(train_df):
    # Drop Gender/Nan
    train_df = train_df.drop(['gender'],axis=1)
    train_df = train_df.dropna()

    # Changing bd outliers to NaN
    train_df['bd'] = train_df['bd'].apply(lambda x: np.nan if x <14 or x > 75 else x)

    # Converting remaining plan duration to int
    train_df['remaining_plan_duration'] = train_df['remaining_plan_duration'].dt.days

    # Converting total seconds to hours
    train_df['total_secs']= round(train_df['total_secs']/3600,2)

    # Creating average usage (hours) per day from latest transaction
    train_df['usage_from_ltd'] = round(train_df['total_secs']/train_df['remaining_plan_duration'],2)
    # replacing Nan with 0
    train_df['usage_from_ltd'].replace([np.nan], 0, inplace=True)
    # replacing inf values with the totalsecs usage.
    train_df['usage_from_ltd'] = np.where(train_df['usage_from_ltd'] == np.inf, train_df['total_secs'],train_df['usage_from_ltd'])

    # converting registration time to datetime object
    train_df['registration_init_time'] = pd.to_datetime(train_df['registration_init_time'], format='%Y%m%d')

    # creating discount percentage column
    train_df['discount_percentage'] = round((train_df['plan_list_price'] - train_df['actual_amount_paid'])/train_df['plan_list_price'],2)
    train_df['discount_percentage'].replace([np.nan,-0.01,0.01], 0, inplace=True)

    return train_df

def cyclic_encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def date_encoding(train_df):
    # cyclic encoding transaction date
    train_df['last_transaction_year'] = train_df['transaction_date'].dt.year
    train_df['last_transaction_month'] = train_df['transaction_date'].dt.month
    train_df = cyclic_encode(train_df, 'last_transaction_month', 12)
    train_df['last_transaction_day'] = train_df['transaction_date'].dt.day
    train_df= cyclic_encode(train_df, 'last_transaction_day', 31)
    train_df =train_df.drop(['last_transaction_month', 'last_transaction_day'],axis=1)

    # cyclic encoding membership expire date
    train_df['expire_year'] = train_df['membership_expire_date'].dt.year
    train_df['expire_month'] = train_df['membership_expire_date'].dt.month
    train_df = cyclic_encode(train_df, 'expire_month', 12)
    train_df['expire_day'] = train_df['membership_expire_date'].dt.day
    train_df= cyclic_encode(train_df, 'expire_day', 31)
    train_df =train_df.drop(['expire_month', 'expire_day'],axis=1)

    # cyclic encoding registration date
    train_df['registration_year'] = train_df['registration_init_time'].dt.year
    train_df['registration_month'] = train_df['registration_init_time'].dt.month
    train_df = cyclic_encode(train_df, 'registration_month', 12)
    train_df['registration_day'] = train_df['registration_init_time'].dt.day
    train_df= cyclic_encode(train_df, 'registration_day', 31)
    train_df =train_df.drop(['registration_month', 'registration_day'],axis=1)

    # droping date columns to avoid data leakege
    train_df =train_df.drop(['registration_init_time', 'membership_expire_date', 'transaction_date'],axis=1)

    return train_df

def under_balancing(train_df):
    df_no_churn = train_df[train_df['is_churn'] == 0]
    df_churn = train_df[train_df['is_churn'] == 1]
    df_no_churn = df_no_churn.sample(27997, random_state=42)
    train_underbalancing = pd.concat([df_no_churn, df_churn], axis=0)

    return train_underbalancing
