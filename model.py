import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
with open("../input/lending-club/accepted_2007_to_2018Q4.csv.gz", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

X = accepted.copy()
X.select_dtypes('object').head()
keep_list = ['addr_state', 'annual_inc', 'dti', 'earliest_cr_line', 'emp_length', 'fico_range_low', 'home_ownership',
             'initial_list_status', 'int_rate', 'loan_amnt', 'loan_status','mths_since_rcnt_il',
             'mths_since_recent_bc', 'mths_since_recent_inq', 'mort_acc', 'pub_rec', 'revol_bal', 
             'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status']
print(keep_list)
print(len(keep_list))
drop_list = [col for col in X.columns if col not in keep_list]
print(drop_list)
print(len(drop_list))
w = [col for col in keep_list if col not in X.columns]
X.drop(labels=drop_list, axis=1, inplace=True)
X.fillna(method='bfill', axis=0).fillna(0)
X1 = X.sample(10000)
plt.figure(figsize=(20,6))
sns.lineplot(data=X1['int_rate'])
X1 = X.dropna()
X1.loan_status
X1['loan_status'] = X1['loan_status'].astype(float)
from sklearn.model_selection import train_test_split
X1.groupby('loan_status').mean()
objects = ['term', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status',
      'title', 'addr_state', 'earliest_cr_line', 'initial_list_status']
X1_in[objects] = X1_in[objects].astype(str)
data_other_cols = X1_in.drop(['term','sub_grade', 'emp_length', 'home_ownership', 'verification_status','title', 'addr_state', 'earliest_cr_line', 'initial_list_status'], axis=1)
data_other_cols
from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['term', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status',
                    'title', 'addr_state', 'earliest_cr_line', 'initial_list_status'] 

onehotencoder = OneHotEncoder()

transformed_data = onehotencoder.fit_transform(X1_in[categorical_cols])

# the above transformed_data is an array so convert it to dataframe
encoded_data = pd.DataFrame(transformed_data, index=X1_in.index)

# now concatenate the original data and the encoded data using pandas
concatenated_data = pd.concat([data_other_cols, encoded_data], axis=1)
concatenated_data = concatenated_data[['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low','pub_rec',
                                      'revol_bal', 'revol_util','total_acc', 'mths_since_rcnt_il',
                                      'mort_acc', 'mths_since_recent_bc','mths_since_recent_inq'
feature_cols = ['loan_amnt', 'term', 'int_rate', 'sub_grade', 'emp_length', 'home_ownership', 
                'annual_inc', 'verification_status', 'title',
                'addr_state', 'dti', 'earliest_cr_line', 'fico_range_low', 'pub_rec',
                'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
                'mths_since_rcnt_il', 'mort_acc', 'mths_since_recent_bc','mths_since_recent_inq']
X = concatenated_data
y = X1['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred=logistic.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_rep = metrics.classification_report(y_test,y_pred)
print(class_rep)                                     