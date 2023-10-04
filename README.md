<img width="1500" alt="Screenshot 2023-10-04 at 18 13 31" src="https://github.com/XavierLooyens/Churn_prediction/assets/110831321/8b448a1b-0439-4557-8b95-dd97c6be2252">





# CHURN?
Churn rate is a measure of the proportion of individuals or items moving out of a group over a specific period. The objective of this project was to predict the churning probability for users of the music streaming plattform - KKBOX

## Solution Structure
 - Data store and retrieved from Google Cloud Platform (GCP) using BigQuery
 - Given the variation of outliers in the different features the data was scaled using different scallers (Standard, MiMax and robust scaler)
 - Utilised previous periods of data to build a trendline in order to identify if a user had previously churned.
 - Considering the dataset was umbalanced and to prevent data leakage the data set was randomly sampled and reduced.
 - Logistic regression model used.
 - Used data relating to over 50K users to train the model.
 - Churning predictions made in the form of a probability.
 - Website created using Streamlit to present predictions. A recommendation given for what actions to take based on the probability of a user churning.

## Feature Importance

<img width="1148" alt="Screenshot 2023-10-04 at 18 52 39" src="https://github.com/XavierLooyens/Churn_prediction/assets/110831321/23ffe1c6-f73b-4136-8e03-40baeef9ec49">


