import pandas as pd
import numpy as np

def transactions_preproc(transactions_df):
    """
    Processes transactions data:
        - Converts date features to datetime object
        - removes duplicates
        - creates new features "remaining_plan_duration" and "is_discount"

    Returns:
        - Dataframe with processed transactions data

    """
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'], format='%Y%m%d')
    transactions_df['membership_expire_date'] = pd.to_datetime(transactions_df['membership_expire_date'], format='%Y%m%d')

    transactions_df_lt = transactions_df.loc[transactions_df.groupby('msno').transaction_date.idxmax()]

    transactions_df_lt['remaining_plan_duration'] = transactions_df_lt['membership_expire_date'] - transactions_df_lt['transaction_date']

    transactions_df_lt['is_discount'] = transactions_df_lt.apply(lambda x: '0' if (x['actual_amount_paid'] -x['plan_list_price'])>=0 else '1', axis=1)

    return transactions_df_lt

def user_logs(user_logs_df,transactions_df_lt):
    """
    Processes user logs data:
        - Converts date features to datetime object
        - Merges user logs data to processed transactions data.
        - Removes rows where usage is before last transaction date
        - groups by "msno" and performs sum


    Returns:
     - Dataframe of processed user logs
    """

    user_logs_df['date'] = pd.to_datetime(user_logs_df['date'], format='%Y%m%d')
    latest_transactions_per_msno = transactions_df_lt[['msno', 'transaction_date']]
    merged_df = user_logs_df.merge(latest_transactions_per_msno, on='msno', how='left')
    merged_df = merged_df.dropna(subset=['transaction_date'])
    user_logs_atd= merged_df.loc[merged_df['date']>=merged_df['transaction_date']]
    user_logs_atd = user_logs_atd.drop(columns=['date','transaction_date'])
    user_logs_atd = user_logs_atd.groupby('msno').sum()

    return user_logs_atd

def merger(transactions_df_lt,user_logs_atd,members_df,trendline_df,train_data):
    """
    Merging transactions, user logs and members data to Train dataframe
    """
    train_df = train_data.merge(transactions_df_lt, on='msno', how='left')
    train_df = train_df.merge(user_logs_atd, on='msno', how='left')
    train_df = train_df.merge(members_df, on='msno', how='left')
    train_df = train_df.merge(trendline_df, on='msno', how='left')

    return train_df

def feautures_eng(train_df):
    """
    Processes final train data:
        - Removes unnecessary columns
        - Drops Nan values
        - New feature of average usage per day fom latest transaction
        - New feature discount percentage

    Returns:
        - Dataframe of processed train data
    """

    train_df.drop(['bd','gender','payment_method_id', 'city', 'registered_via'],axis =1)
    train_df = train_df.dropna()

    train_df['remaining_plan_duration'] = train_df['remaining_plan_duration'].dt.days
    train_df['total_secs']= round(train_df['total_secs']/3600,2)

    train_df['usage_from_ltd'] = round(train_df['total_secs']/train_df['remaining_plan_duration'],2)
    train_df['usage_from_ltd'].replace([np.nan], 0, inplace=True)
    train_df['usage_from_ltd'] = np.where(train_df['usage_from_ltd'] == np.inf, train_df['total_secs'],train_df['usage_from_ltd'])

    train_df['registration_init_time'] = pd.to_datetime(train_df['registration_init_time'], format='%Y%m%d')

    train_df['discount_percentage'] = round((train_df['plan_list_price'] - train_df['actual_amount_paid'])/train_df['plan_list_price'],2)
    train_df['discount_percentage'].replace([np.nan,-0.01,0.01], 0, inplace=True)

    return train_df

def cyclic_encode(data, col, max_val):
    """
    Creates new Columns with cyclic encoding(sin,cos)
    """
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def date_encoding(train_df):
    """
    Performs the cyclic encoding of the date features.

    Returns:
        - Dataframe with cyclic encoded features and removes previous date features
    """
    train_df['last_transaction_year'] = train_df['transaction_date'].dt.year
    train_df['last_transaction_month'] = train_df['transaction_date'].dt.month
    train_df = cyclic_encode(train_df, 'last_transaction_month', 12)
    train_df['last_transaction_day'] = train_df['transaction_date'].dt.day
    train_df= cyclic_encode(train_df, 'last_transaction_day', 31)
    train_df =train_df.drop(['last_transaction_month', 'last_transaction_day'],axis=1)

    train_df['expire_year'] = train_df['membership_expire_date'].dt.year
    train_df['expire_month'] = train_df['membership_expire_date'].dt.month
    train_df = cyclic_encode(train_df, 'expire_month', 12)
    train_df['expire_day'] = train_df['membership_expire_date'].dt.day
    train_df= cyclic_encode(train_df, 'expire_day', 31)
    train_df =train_df.drop(['expire_month', 'expire_day'],axis=1)

    train_df['registration_year'] = train_df['registration_init_time'].dt.year
    train_df['registration_month'] = train_df['registration_init_time'].dt.month
    train_df = cyclic_encode(train_df, 'registration_month', 12)
    train_df['registration_day'] = train_df['registration_init_time'].dt.day
    train_df= cyclic_encode(train_df, 'registration_day', 31)
    train_df =train_df.drop(['registration_month', 'registration_day'],axis=1)

    train_df =train_df.drop(['registration_init_time', 'membership_expire_date', 'transaction_date'],axis=1)

    return train_df

def under_balancing(train_df):
    """
    Splits dataframe by target ---> "is_churns"

    Returns:
        - Balanced Dataframe

    """
    df_no_churn = train_df[train_df['is_churn'] == 0]
    df_churn = train_df[train_df['is_churn'] == 1]
    df_no_churn = df_no_churn.sample(27997, random_state=42)
    train_underbalancing = pd.concat([df_no_churn, df_churn], axis=0)

    return train_underbalancing
