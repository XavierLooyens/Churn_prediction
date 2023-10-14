<img width="1500" alt="Screenshot 2023-10-04 at 18 13 31" src="https://github.com/XavierLooyens/Churn_prediction/assets/110831321/8b448a1b-0439-4557-8b95-dd97c6be2252">





# CHURN?
Churn rate is a measure of the proportion of individuals or items moving out of a group over a specific period. 
Based on that, churners are the people leaving the group. Many businesses know what their churn rate is, however they do not know who is churning. Being able to identify which users are more likely to churn offers an opportunity to save money for businesses. 

The objective of this specific project was to predict probability for users of the music streaming plattform - KKBOX

## Solution Structure
 - Data store and retrieved from Google Cloud Platform (GCP) using BigQuery
 - Given the variation of outliers in the different features the data was scaled using different scallers (Standard, MiMax and robust scaler)
 - Utilised previous periods of data to build a trendline in order to identify if a user had previously churned.
 - Considering the dataset was umbalanced and to prevent data leakage the data set was randomly sampled and reduced.
 - Logistic regression model used.
 - Used data relating to over 50K users to train the model.
 - Churning predictions made in the form of a probability.
 - Website created using Streamlit to present predictions. A recommendation given for what actions to take based on the probability of a user churning.

##Links
Link to Streamlit: https://churnpredictfrontend-splvw2mkwctb829dm24nc5.streamlit.app/
Link to Streamlit GitHub: https://github.com/XavierLooyens/Churn_prediction_frontend
Link to presentation: https://www.canva.com/design/DAFvRLK95Q4/XiRhcHtFOIWwD1UywEjBvQ/view
Link to data: https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data


## Feature Importance

<img width="1148" alt="Screenshot 2023-10-04 at 18 52 39" src="https://github.com/XavierLooyens/Churn_prediction/assets/110831321/23ffe1c6-f73b-4136-8e03-40baeef9ec49">


