# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:45:16 2022

@author: chitr
"""


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

 

df_chitra = pd.read_csv("C:/Users/chitr/OneDrive/Documents/KSI.csv")

print('Type and Names of columns: ')
print(df_chitra.dtypes)

print("\nStatistics per column: \n")
print(df_chitra.describe())

print('\nUnique Values per Column')
print(df_chitra.nunique())

# Some columns have too many unique values and hence will have toto high varience to be useful
uniq_cols=['ObjectId','ACCNUM','DATE','TIME','STREET1','STREET2','NEIGHBOURHOOD','WARDNUM','DIVISION']

for col in df_chitra.columns:
    df_chitra[col].replace('<Null>', np.nan, inplace=True)
    df_chitra[col].replace('unknown', np.nan, inplace=True)
    df_chitra[col].replace('Unknown', np.nan, inplace=True)

print('\nMissing Values per Column')
print(df_chitra.isna().sum())

# Many columns have too much data missing hence we cannot use these columns
null_cols=['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND','PEDESTRIAN',
            'CYCLIST','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','PASSENGER','SPEEDING','REDLIGHT',
            'ALCOHOL','DISABILITY']

# Removing unusable columns
df_chitra = df_chitra.drop(null_cols,axis='columns')
df_chitra = df_chitra.drop(uniq_cols,axis='columns')

df_chitra.head()

# Datatypes of columns
num_cols=['X','Y','YEAR','HOUR','LATITUDE','LONGITUDE','HOOD_ID']
cat_cols=['ROAD_CLASS','DISTRICT','LOCCOORD','ACCLOC','TRAFFCTL','VISIBILITY','LIGHT','RDSFCOND',
          'IMPACTYPE','INVTYPE','INVAGE','INJURY','INITDIR','VEHTYPE','MANOEUVER','DRIVACT',
          'DRIVCOND','AUTOMOBILE','AG_DRIV','POLICE_DIVISION']

# Handling Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
columns=df_chitra.columns
df_chitra = pd.DataFrame(imputer.fit_transform(df_chitra))
df_chitra.columns = columns
print('\nMissing Values per Column')
print(df_chitra.isna().sum())

# After Imputation all columns including numerical columns have been converted to categorical
# and hence we need to change them to numerical


# Converting to Numerical Columns
for col in num_cols:
    df_chitra[col] = pd.to_numeric(df_chitra[col])


# Handling Categorical Columns with get_dummies
data_cat=df_chitra[cat_cols]

#### Column Transformer
cat_col = df_chitra.drop('ACCLASS', axis = 1).select_dtypes('object').columns
num_col = df_chitra.select_dtypes(np.number).columns


for col in cat_col:
    df_chitra[col] = df_chitra[col].astype('category')
    
df_chitra.dtypes
df_chitra['ACCLASS'].unique()
df_chitra.drop(df_chitra.index[df_chitra['ACCLASS'] == 'Property Damage Only'], inplace = True)
df_chitra['ACCLASS'].unique()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer([
        ('encoder', encoder, cat_col),
        ('scaling', scaler, num_col)],
        remainder = 'passthrough',
        sparse_threshold = 0)

#dummies = pd.get_dummies(data_cat,dummy_na=False)
#data=pd.concat([df_albert,dummies],axis=1)
#data=data.drop(cat_cols,axis=1)

# Label Encoder for ACCLASS (y variable)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df_albert['ACCLASS'] = label_encoder.fit_transform(df_albert['ACCLASS'])

# Standard Scaling

# data.loc[num_cols] = scaler.fit_transform(data.loc[num_cols])
df_albert['ACCLASS'].value_counts()

# Splitting data into training and testing sets
X = df_albert.drop(['ACCLASS'],axis=1)
y = df_albert['ACCLASS']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)
