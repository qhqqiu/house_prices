# house_prices
Codes for Kaggle contest of house-price predictions: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Predictions with Xgboost.

#Steps

1. Loading train and test data

2. Data Preprocessing and Feature Engineering
* Concatenating train and test data
* Replacing nulls with 0
* Splitting categorical features and numeric features
* Label encoding then one hot encoding categorical features
* Generating polynomial features by numeric features with degree=2
* Concatenating categorical features and numeric features
* Spliting train and test data
* Transforming train and test data to DMatrix

3. Importing Xgboost and setting parameters 

4. Fiting train DMatrix, and predicting for testDMatrix

My results got a score of 0.12460, top 24% of all the competitors, which is not bad.
