import numpy as np
import pandas as pd
import os
from ml_pre_proc.trendline import trendline_merger,trendline_preproc,trendline_compute,trendline_is_churn
from ml_pre_proc.pre_proc import transactions_preproc,user_logs,merger,feautures_eng,date_encoding,under_balancing
from ml_pre_proc.data import get_data
from ml_pre_proc.model import initialize_model
import pickle

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/andretomaz/code/XavierLooyens/GCP/churn-prediction-398917-8a95102c50a6.json"

def data_processing(transactions_0_data,transactions_data,train_0_data,train_data,user_logs_data, members_data):
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

#model
def train_model(data_df):
    # dropping unnecessary columns
    data_df = data_df.drop(['msno', 'bd', 'payment_method_id', 'city', 'registered_via'], axis=1)

    #splitting features and target
    X = data_df.drop(['is_churn'], axis=1)
    y = data_df['is_churn']

    #instatiating model and fitting
    model = initialize_model()
    result = model.fit(X, y)

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(result, open(filename, 'wb'))

    return result



if __name__ == '__main__':
    print("starting the process")
    #importing data
    #gcp_project="churn-prediction-398917"
    #members_data, transactions_0_data, transactions_data, user_logs_data, train_0_data, train_data = get_data(gcp_project)
    #importing 1st transactions
    transactions_0_data= pd.read_csv("../raw_data/transactions.csv")
    #Importing Transactions data
    transactions_data= pd.read_csv("../raw_data/transactions_v2.csv")
    #Importing user logs data
    user_logs_data = pd.read_csv("../raw_data/user_logs_v2.csv")
    #importing members data
    members_data= pd.read_csv("../raw_data/members_v3.csv")
    # import first training dataset
    train_0_data= pd.read_csv("../raw_data/train.csv")
    # import training dataset
    train_data = pd.read_csv("../raw_data/train_v2.csv")
    print("data downloaded")
    # pre processing
    churn_df_balanced = data_processing(transactions_0_data,transactions_data,train_0_data,train_data,user_logs_data,members_data)
    print("data Processed")
    train_model(churn_df_balanced)
    print("model saved")
