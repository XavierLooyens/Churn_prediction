import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from joblib import dump



#Get the data from the csv file
## this will need to be tweaked to get the data from the other .py scripts
def get_data():
    data_df= pd.read_csv("../raw_data/churn_df_underbalanced.csv")
    return data_df

#drop columns
def drop_columns(data_df):
    data_df = data_df.drop(['Unnamed: 0','msno', 'bd', 'payment_method_id', 'city', 'registered_via'], axis=1)
    return data_df

#create X and y
def get_X_y(data_df):
    X = data_df.drop(['is_churn'], axis=1)
    y = data_df['is_churn']
    return X, y

#Create lists of features for their relevant scalers
robust_features = ['remaining_plan_duration',
                   'usage_from_ltd',
                   'payment_plan_days',
                   'plan_list_price',
                   'actual_amount_paid',
                   'num_50',
                   'num_75',
                   'num_985',
                   'expire_year',
                   'last_transaction_year'
]
minmax_features = ['registration_year']

normal_features = [ 'num_25',
                    'num_100'
                    'num_unq'
                    'total_secs',
]

#Pipeline
def build_pipeline():
    #robust scaler for robust features that contain outliers
    robust_pipeline = make_pipeline(RobustScaler())
    #minmax scaler for features that have no outliers but are not normally distributed
    minmax_pipeline = make_pipeline(MinMaxScaler())
    #logarithmic transformation of the normal features and then standard scaler
    log_pipeline = make_pipeline(
        FunctionTransformer(np.log1p, validate=True),
        StandardScaler())

    #preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('Robust', robust_pipeline, robust_features),
            ('MinMax', minmax_pipeline, minmax_features),
            ('Log', log_pipeline, normal_features)
        ], remainder='passthrough'
    )

    #model
    model = LogisticRegression()


    #preprocessor + model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline

#fit the pipeline
def fit_pipeline(pipeline, X, y):
    result = pipeline.fit(X, y)
    return result

#predict churn
def predict_churn(pipeline, X_test):
    y_pred = pipeline.predict(X_test)
    return y_pred

#predict churn probability
def predict_churn_proba(pipeline, X_test):
    y_pred_proba = pipeline.predict_proba(X_test)
    return y_pred_proba

# saving the trained model
def save_model(result):
    dump(result, 'LogRegModel.joblib')
    return None
