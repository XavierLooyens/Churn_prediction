import pandas as pd
from google.cloud import bigquery
import os
from pathlib import Path


def get_data(gcp_project):
    """
    Queries Google big query to retrive needed data from the bucket

    Returns:
        - Dataframes: members_v3, transactions, transactions_v2, user_logs_v2, train, train_v2
    """
    query = """
            SELECT *
            FROM churn-prediction-398917.churn_prediction.members_v3
            """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    members_v3= result.to_dataframe()

    query = """
            SELECT *
            FROM churn-prediction-398917.churn_prediction.transactions
            """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    transactions = result.to_dataframe()

    query = """
            SELECT *
            FROM churn-prediction-398917.churn_prediction.transactions_v2
            """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    transactions_v2 = result.to_dataframe()

    query = """
            SELECT *
            FROM churn-prediction-398917.churn_prediction.user_logs_v2
            """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    user_logs_v2 = result.to_dataframe()

    query = """
            SELECT *
            FROM churn-prediction-398917.churn_prediction.train
            """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    train = result.to_dataframe()

    query = """
            SELECT *
            FROM churn-prediction-398917.churn_prediction.train_v2
            """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    train_v2 = result.to_dataframe()

    return members_v3, transactions, transactions_v2, user_logs_v2, train, train_v2
