<img width="1500" alt="Screenshot 2023-10-04 at 18 13 31" src="https://github.com/XavierLooyens/Churn_prediction/assets/110831321/8b448a1b-0439-4557-8b95-dd97c6be2252">





# CHURN?
Churn rate is a measure of the proportion of individuals or items moving out of a group over a specific period. 
Based on that, churners are the people leaving the group. Many businesses know what their churn rate is, however they do not know who is churning. Being able to identify which users are more likely to churn offers an opportunity to save money for businesses. 

The objective of this specific project was to predict probability for users of the music streaming plattform - KKBOX

## Solution Structure 
 - Data stored and retrieved from Google Cloud Platform (GCP) using BigQuery
 - Used previous periods of data to build a churn trendline in order to identify if a user had previously churned.
 - Month and days were cyclic encoded. 
 - Given the variation of outliers in the different features the data was scaled using different scallers (Standard, MiMax and Robust Scaler)
 - Considering the dataset was unbalanced we underbalanced the dataset. We tried SMOTE but the synthetic rows caused overfitting. 
 - Logistic regression model used.
 - XGboost classifier and Random Forest models were tried but were overfitting. 
 - Used data relating to over 50K users to train the model. 
 - Churning predictions made in the form of a probability.
 - Website created using Streamlit to present predictions.
 - Streamlit app devides churners into buckets of likelihood off churning:
    - <50% likelihood: low risk
    - >50% and <90% likelihood: medium risk
    - >90% likelihood: high risk
 - Streamlit app provides options to download user ID's to use for targeting. 

## Confusion matrix 
We used the confusion matrix to decide what score metric was the most relevant in this case.

<img width="1431" alt="Screenshot 2023-10-14 at 18 15 36" src="https://github.com/XavierLooyens/Churn_prediction/assets/130698577/3d508c78-981b-4c28-94b9-e7c1632e878a">

## Score Metric
Recall was the most relevant score metric; a high recall score means that you are working towards a low number of false negatives. 
False negatives is this case means that we are predicting that a user will not churn but they are actually going to churn. The risk here is that the business loses money. We want to reduce that risk. 

<img width="1432" alt="Screenshot 2023-10-14 at 18 15 48" src="https://github.com/XavierLooyens/Churn_prediction/assets/130698577/92352425-6959-43d4-a7fc-c2b1b0bd7a71">


## Feature Importance
is_cancel is the biggest indicator of a person potentially churning. It is valuable to know that a cancel only turns into a churn if the user does not resubscribe within 30 days. 
is_auto_renew is the biggest indicator of a person potentially not churning. 

<img width="1148" alt="Screenshot 2023-10-04 at 18 52 39" src="https://github.com/XavierLooyens/Churn_prediction/assets/110831321/23ffe1c6-f73b-4136-8e03-40baeef9ec49">

## Links
Link to Streamlit: https://churnpredictfrontend-splvw2mkwctb829dm24nc5.streamlit.app/
Link to Streamlit GitHub: https://github.com/XavierLooyens/Churn_prediction_frontend
Link to presentation: https://www.canva.com/design/DAFvRLK95Q4/XiRhcHtFOIWwD1UywEjBvQ/view
Link to data: https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
