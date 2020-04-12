import numpy as np
import pandas as pd

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#briefly checking the training data
train_data.head()

train_data.columns
#notice that 'SalePrice' is the y 

train_data.info

#data cleaning 
aa = train_data.isnull().sum() #checking how many nulls each columns has
aa[aa>0].sort_values(ascending=False) #only listing the columns that have nulls, and listing in descending order 

#for part of the features, replacing nulls with "None"
columns_1= ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageYrBlt", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtExposure", "BsmtFinType1", "BsmtCond",   "MasVnrType", "Electrical"]
for column in columns_1:
    train_data[column].fillna("None", inplace=True)

#for other features, replacing nulls with "0"
columns_2 = ["BsmtQual","MasVnrArea", 'LotFrontage']
for column in columns_2:
    train_data[column].fillna("0", inplace=True)

aa = train_data.isnull().sum()
print(aa)

#categorical features are represented as numerical, so I need to convert them to string
NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    train_data[col] = train_data[col].astype(str)

#then, I use label encoding to label these categorical features
from sklearn.preprocessing import LabelEncoder
Columns = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in Columns:
    labelencoder = LabelEncoder()
    labelencoder.fit(list(train_data[col].values))
    train_data[col] = labelencoder.transform(list(train_data[col].values))
print("Shape of train_data: {}".format(train_data.shape))
train_data.head()

integer_features = train_data.dtypes[train_data.dtypes != "int64"].index
print(interger_features)

# Check the skew of all numerical features
from scipy.stats import skew
skewness = train_data[integer_features].apply(lambda x: skew(x))
skew = pd.DataFrame({'Skew' :skewness})
skew.head(10)
#normal distributed data, the skew is 0
#skew less thatn 0, meaning there's more wight on the left tail of the distrubution 
#skew greater thatn 0, meaning there's more wight on the right tail of the distrubution  

#use log1p to normalize skewed features
skewness_features = skewness[abs(skewness) >= 0.5].index #target features that skewness absolutes over than 0.5
#train_data[skewed_features.index] = np.log1p(train_data[skewed_features])
print(skewness_features)
train_data[skewness_features] = np.log1p(train_data[skewness_features])
train_data.head()

#Convert categorical variable into dummy/indicator variables.
train_data = pd.get_dummies(data=train_data,dummy_na=True) 
train_data.head()
train_data.shape

X_train = train_data.iloc[:, 1:-1]
print(X_train)
print(X_train.shape)
X_train = X_train.as_matrix()

y_train = train_data.iloc[:, -1]
print(y_train)
print(y_train.shape)
#y_train = y_train.as_matrix()

from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train, verbose=False)

#tuning prarameters of xgboost
#generally small learning rate and large n_estimator makes better xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
model = XGBRegressor(n_estimator = 1000)
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate = learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold) #setting n_jobs to makes it run faster
grid_result = grid_search.fit(X_train, y_train)
print("Best: %f of using %s" % (grid_result.best_score_, grid_result.best_params_))

aa = test_data.isnull().sum() #checking how many nulls each columns has
aa[aa>0].sort_values(ascending=False) #only listing the columns that have nulls, and listing in descending order 
columns_1= ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageYrBlt", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtExposure", "BsmtFinType1", "BsmtCond",   "MasVnrType", "Electrical"]
for column in columns_1:
    test_data[column].fillna("None", inplace=True)
    
columns_2 = ["BsmtQual","MasVnrArea", 'LotFrontage']
for column in columns_2:
    test_data[column].fillna("0", inplace=True)
aa = train_data.isnull().sum()
print(aa)

NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    test_data[col] = test_data[col].astype(str)

from sklearn.preprocessing import LabelEncoder
Columns = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in Columns:
    labelencoder = LabelEncoder()
    labelencoder.fit(list(test_data[col].values))
    test_data[col] = labelencoder.transform(list(test_data[col].values))
print("Shape of train_data: {}".format(test_data.shape))

test_data.head()

numeric_features = test_data.dtypes[test_data.dtypes != "object"].index
print(numeric_features)
from scipy.stats import skew
skewness = test_data[numeric_features].apply(lambda x: skew(x))
skew = pd.DataFrame({'Skew' :skewness})
skew.head(10)

skewness_features = skewness[abs(skewness) >= 0.5].index #target features that skewness absolutes over than 0.5
#train_data[skewed_features.index] = np.log1p(train_data[skewed_features])
print(skewness_features)
test_data[skewness_features] = np.log1p(test_data[skewness_features])
test_data.head()
test_data.shape

test_data = pd.get_dummies(data=test_data,dummy_na=True) 
test_data.head()
test_data.shape

X_test = test_data.drop(["Id"], axis=1)
print(X_test)
print(X_test.shape)
X_test.head()
X_test = X_test.as_matrix()

predictions = grid_search.predict(X_test)