import numpy as np
import pandas as pd
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_pre_proc.trendline import trendline_merger,trendline_preproc,trendline_compute,trendline_is_churn
from ml_pre_proc.pre_proc import transactions_preproc,user_logs,merger,feautures_eng,date_encoding,under_balancing
from ml_pre_proc.data import get_data

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/andretomaz/code/XavierLooyens/GCP/churn-prediction-398917-8a95102c50a6.json"

def data_processing(transactions_0_data,transactions_data,train_0_data,train_data,user_logs_data, members_data):
    trendline_df = trendline_merger(transactions_0_data,transactions_data,train_0_data,train_data)
    trendline_df = trendline_preproc(trendline_df)
    trendline_df = trendline_compute(trendline_df)
    trendline_df = trendline_is_churn(trendline_df)

    transactions_df = transactions_preproc(transactions_data)

    user_logs_df = user_logs(user_logs_data, transactions_df)

    churn_df = merger(transactions_df,user_logs_df,members_data,trendline_df)
    churn_df = feautures_eng(churn_df)

    churn_df_encoded = date_encoding(churn_df)
    churn_df_balanced = under_balancing(churn_df_encoded)

    return churn_df_balanced

#model


#prediction


if __name__ == '__main__':
    #importing data
    members_data, transactions_0_data, transactions_data, user_logs_data, train_0_data, train_data = get_data(gcp_project="churn-prediction-398917",data_has_header=True)
    # pre processing
    data_processing(transactions_0_data,transactions_data,train_0_data,train_data,user_logs_data,members_data)
    model
    prediction
