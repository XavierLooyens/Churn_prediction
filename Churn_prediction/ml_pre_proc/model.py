import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from joblib import dump

def initialize_model():
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
                        'num_100',
                        'num_unq',
                        'total_secs'
    ]

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



