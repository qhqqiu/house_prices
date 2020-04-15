# house_prices
Codes for Kaggle contest of house-price predictions: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

I'm trying to implement Xgboost here.

A rough list of I've got for now:

1. Loading train and test data

2. Preprocessing the train data
* For features that have string data, replacing nulls with "None"
* For features that have numeric data, replacing nulls with "0"
* Categorical features are represented as numerical, so I converted them to strings. I then used label encoding to label these categorical features.
* Filtering out all the numerical features and check the skewness. Then I used log1p to normalize skewed features (setting the skewness threshold abs(skewness) >= 0.5)

3. Implementing Xgboost on the train data
* Tuning parameters with gola of small learning rate and large n_estimator 

