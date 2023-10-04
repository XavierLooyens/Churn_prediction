import numpy as np
import pandas as pd
import os
from ml_pre_proc.trendline import trendline_merger,trendline_preproc,trendline_compute,trendline_is_churn
from ml_pre_proc.pre_proc import transactions_preproc,user_logs,merger,feautures_eng,date_encoding,under_balancing
from ml_pre_proc.data import get_data
from ml_pre_proc.model import initialize_model
import joblib


def data_processing(transactions_0_data,transactions_data,train_0_data,train_data,user_logs_data, members_data):
    """
    Processed and balances the data.

    Returns:
        - balanced/processed Dataframe
    """

    trendline_df = trendline_merger(transactions_0_data,transactions_data,train_0_data,train_data)
    trendline_df = trendline_preproc(trendline_df)
    trendline_df = trendline_compute(trendline_df)
    trendline_df = trendline_is_churn(trendline_df)

    transactions_df = transactions_preproc(transactions_data)

    user_logs_df = user_logs(user_logs_data, transactions_df)

    churn_df = merger(transactions_df,user_logs_df,members_data,trendline_df,train_data)
    churn_df = feautures_eng(churn_df)

    churn_df_encoded = date_encoding(churn_df)
    churn_df_balanced = under_balancing(churn_df_encoded)

    return churn_df_balanced


def train_model(data_df):
    """
    Trains the model and saves model results as a pickle file.
    """
    data_df = data_df.drop(['msno'], axis=1)

    X = data_df.drop(['is_churn'], axis=1)
    y = data_df['is_churn']

    model = initialize_model()
    result = model.fit(X, y)

    joblib.dump(result, 'model.pkl')

    return result



if __name__ == '__main__':
    print("Starting the process")
    gcp_project="churn-prediction-398917"
    cache_path = "/Users/andretomaz/code/XavierLooyens/Churn_prediction/raw_data/Cache/churn_df_balanced.csv"

    if not os.path.isfile(cache_path):
        members_data, transactions_0_data, transactions_data, user_logs_data, train_0_data, train_data = get_data(gcp_project)
        print("Data Saved")
        churn_df_balanced = data_processing(transactions_0_data,transactions_data,train_0_data,train_data,user_logs_data,members_data)
        churn_df_balanced.to_csv(cache_path)
    print("Data Processed")

    churn_df_balanced = pd.read_csv(cache_path)
    train_model(churn_df_balanced)
    print("Model saved")
