import pandas as pd
from google.cloud import bigquery

def get_data(
        gcp_project:str,
        data_has_header=True
    ) -> pd.DataFrame:

    client = bigquery.Client(project=gcp_project)

    query = """SELECT * FROM churn-prediction-398917.churn_prediction.members_v3"""
    query_job = client.query(query)
    result = query_job.result()
    members_v3= result.to_dataframe()

    query = """SELECT * FROM churn-prediction-398917.churn_prediction.transactions"""
    query_job = client.query(query)
    result = query_job.result()
    transactions = result.to_dataframe()

    query = """SELECT * FROM churn-prediction-398917.churn_prediction.transactions"""
    query_job = client.query(query)
    result = query_job.result()
    transactions_v2 = result.to_dataframe()

    query = """SELECT * FROM churn-prediction-398917.churn_prediction.user_logs_v2"""
    query_job = client.query(query)
    result = query_job.result()
    user_logs_v2 = result.to_dataframe()

    query = """SELECT * FROM churn-prediction-398917.churn_prediction.train"""
    query_job = client.query(query)
    result = query_job.result()
    train = result.to_dataframe()

    query = """SELECT * FROM churn-prediction-398917.churn_prediction.train_v2"""
    query_job = client.query(query)
    result = query_job.result()
    train_v2 = result.to_dataframe()

    return members_v3, transactions, transactions_v2, user_logs_v2, train, train_v2

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)


    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
