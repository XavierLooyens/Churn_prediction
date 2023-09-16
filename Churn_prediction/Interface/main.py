import numpy as np
import pandas as pd

from ml_pre_proc.trendline import trendline_merger,trendline_preproc,trendline_compute,trendline_is_churn
from ml_pre_proc.pre_proc import transactions_preproc,user_logs,merger,feautures_eng,date_encoding,under_balancing

# importing data using bigquery
query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}"""

data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

#transactions_0_data =
#transactions_data =
#train_0_data =
#train_data =
#user_logs_data =
#members_data =



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




if __name__ == '__main__':
    data_processing(transactions_0_data,transactions_data,train_0_data,train_data,user_logs_data,members_data)
