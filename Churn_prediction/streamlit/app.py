import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
for sckit
import joblib

st.set_page_config(
        page_title="Churn Prediction App",
        page_icon=":bar_chart",
        layout="wide"
    )
st.title('Churn Prediction App')


#package = joblib.load("/home/nazneen/code/nazneen78/Churn_prediction_front_end/model /package.pkl")
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#loaded_preproc = package["preprocessor"]

def main():

    # upload CSV file
    st.sidebar.header('User Input')
    uploaded_file = st.file_uploader("Upload a csv file here", type=["csv"])

    st.sidebar.markdown("""
    **Instructions:**
    1. Upload a CSV file containing the data you want to predict.
    2. The file should have the same columns as the training data.
    3. After uploading, click the 'Predict' button to see predictions.

    Example CSV format:
    ```csv
    Unnamed: 0,msno,feature1,feature2, ...
    0,12345,0.5,0.3, ...
    1,67890,0.2,0.7, ...
    ```
    """)


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        ids= df.msno
        X_test= df.drop(['msno'], axis=1)

        st.sidebar.header("Please filter here: ")

        X_columns = X_test.columns.to_list()
        #pre_processor = predict_pipeline()
        #X_transformed = pre_processor.fit_transform(X_test)
        #X_transformed = pd.DataFrame(X_transformed,columns=X_columns)

        # make predictions
        predict = loaded_model.predict_proba(X_test)*100
        new = pd.DataFrame({'id': ids, 'prediction percentage': predict[:,1]})

        # Get feature coefficients
        feature_coefficients = pd.DataFrame({'Feature': X_test.columns, 'Coefficient': abs(loaded_model.coef_[0])})
        feature_coefficients = feature_coefficients.sort_values(by='Coefficient', ascending=False)




        st.subheader("predictions:")
        st.table(new)

        st.subheader('Feature Coefficients:')
        feature_importance_df= pd.DataFrame(feature_coefficients)
        fig1 = px.bar(
            feature_importance_df,
            x='Feature',
            y='Coefficient',
            # color='Sign',
            title='Feature Importance Based on Coefficients'
        )

        # Customize the layout
        fig1.update_layout(
            xaxis_title='Feature',
            yaxis_title='Coefficient',
            showlegend=True,
            barmode='relative',
        )
        # st.plotly_chart(fig1)

        st.subheader("Churn Statistics")

        churn_count = new['prediction percentage'].apply(lambda x: 'Churn' if x >= 50 else 'No Churn')
        churn_counts = churn_count.value_counts()

        fig = px.pie(new, values=churn_counts.values, names=churn_counts.index, title='Churn Distribution')
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=[0.1, 0.1],
            hole=0.3,
            )
        fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
            )
        # st.plotly_chart(fig)

        left_column, right_column = st.columns(2)
        left_column.plotly_chart(fig1, use_container_width=True)
        right_column.plotly_chart(fig, use_container_width=True)

        # # Hide streamlit style

        hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}
        </style>"""

        st.markdown(hide_st_style, unsafe_allow_html=True)


main()
