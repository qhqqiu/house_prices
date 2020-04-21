import numpy as np
import pandas as pd
from dfply import *
#adjust display to get a better view of columns and rows
pd.options.display.max_rows = 100
pd.options.display.max_columns = 500
print(pd.options.display.max_rows)
print(pd.options.display.max_columns)

train_0 = pd.read_csv("train.csv")
test_0 = pd.read_csv("test.csv")

#briefly checking the training data
train_0.head()

#check data types
types=np.array(train_0.dtypes)
types_df=pd.DataFrame(train_0.dtypes)
types_df.columns=['types']
types_df_count=pd.DataFrame(types_df['types'].value_counts())
types_df_count.columns=['counts']
types_df_count

len(train_0.columns)

train_0=train_0.set_index(['Id'])
test_0= test_0.set_index(['Id'])

train_0_X = train_0.iloc[:, 1:79]
train_0_y = train_0['SalePrice']
test_0_X =test_0.copy()

all=pd.concat([train_0_X,test_0_X])

all.fillna(0,inplace=True)

categorical = pd.DataFrame()
numeric = pd.DataFrame()
for k in all.columns:
    dtype_k = all[k].dtype
    if dtype_k == 'object':
        categorical[k] = all[k]
    else:
            numeric[k] = all[k]

categorical.head()
print(categorical.columns)
print(categorical.shape)
numeric.head()

import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
from sklearn import preprocessing
Columns = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2',
       'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',
       'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond',
       'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC',
       'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
       'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood',
       'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition',
       'SaleType', 'Street', 'Utilities']
for col in Columns:
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(list(categorical[col].values))
    categorical[col] = labelencoder.transform(list(categorical[col].values))
    
print("Shape of categorical data: {}".format(categorical.shape))
print(categorical.head())

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
onehotencoder = OneHotEncoder(sparse=False,dtype=np.int)
categorical = pd.DataFrame(onehotencoder.fit_transform(categorical), index = categorical.index)

print(categorical.head())

pf = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
numeric = pd.DataFrame(pf.fit_transform(numeric), index = numeric.index)

all = pd.concat([categorical, numeric], axis=1)
all.columns = np.arange(all.shape[1])
all.head()

train_1_X=all.loc[train_0_X.index,:]
test_1_X=all.loc[test_0_X.index,:]
train_1_y=train_0_y.copy()

print(train_1_X.shape)
print(test_1_X.shape)

import xgboost as xgb
D_train = xgb.DMatrix(train_1_X, label=train_1_y)
D_test = xgb.DMatrix(test_1_X)

params = {
            'objective': 'reg:gamma',
            'eta': 0.002,
            'seed': 0,
            'missing': -999,
            'silent' : 1,
            'gamma' : 0.02,
            'subsample' : 0.5,
            'alpha' : 0.045,
            'max_depth':4,
            'min_child_weight':1
            }
num_rounds=20000

#attention: The eta parameter is importaant! It gives us a chance to prevent overfitting.
#objective = the loss function being used

model = xgb.train(params, D_train, num_rounds)

preds = pd.Series(model.predict(D_test),index=test_1_X.index)
results = pd.DataFrame(preds,columns=['SalePrice'])
results.to_csv("results.csv", index=False)
print(pd.read_csv("results.csv"))