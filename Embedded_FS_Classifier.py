import time
import tracemalloc
import timeit

import numpy as np

start_time = time.time()
tracemalloc.start()
import os
import pandas as pd
os.chdir('C:/Users/bitel/PycharmProjects/CIND820_Project/CIND820_Project')
crime = pd.read_csv('cleanedcommunitiescrime.csv', sep=',')
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
#The target variable "ViolentCrimesPerPop" was divided into 5 quantile categorical variables.
x = crime.drop('ViolentCrimesPerPop', axis = 1)
y = crime['ViolentCrimesPerPop']
y = pd.qcut(y, q=5, labels=['Very Low Crime', 'Low Crime', 'Medium Crime', 'High Crime', 'Very High Crime'])
#Data was split into training (0.7) and test (0.3) data. The following preprocessing procedures and modelling will occur on the training data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.3, random_state=345)
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
x_train['OtherPerCap'] = pd.to_numeric(x_train['OtherPerCap'], errors='coerce')
###Converted the missing/null values into column median values.
x_train = x_train.fillna(0)
x_train = x_train.replace(0, x_train.median())
#Conducted Yeo and Johnson transformation on the dataset because numerous features were skewed distributions.
#In order to improve the predictor variables and possibly the model, this transformation was applied.
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
x_train = pd.DataFrame(pt.fit_transform(x_train), columns=x_train.columns)
#The feature selection embedded method Lasso was applied to the preprocessed training dataset.
#The top 10 features were selected.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
importances = rfc.feature_importances_
selection = x_train.columns
sorted_importances = pd.Series(importances, index=selection).sort_values(ascending=False)
top_features = sorted_importances[:10]
selected_features = list(top_features.index)
x_train_transform = x_train[selected_features]
x_test_transform = x_test[selected_features]
print(selected_features)
#k-fold cross validation was used to evaluate the Support Vector Classifier (SVC) model against 20 split samples.
#The average accuracy score was determined as a performance metric.
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = SVC()
score = cross_val_score(model, x_train_transform, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
mean_score = sum(score)/30
model.fit(x_train_transform, y_train)
print(mean_score)
y_pred = model.predict(x_test_transform)
mcc = matthews_corrcoef(y_test, y_pred)
print(mcc)
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(x_test_transform), zero_division= np.nan))
snapshot = tracemalloc.take_snapshot()
end_time = time.time()
tracemalloc.stop()
memory = snapshot.statistics('lineno')
for stat in memory[:10]:
    print(stat)
print(end_time - start_time)
t = timeit.timeit(stmt='''import time
import tracemalloc
import timeit
start_time = time.time()
tracemalloc.start()
import os
import pandas as pd
os.chdir('C:/Users/bitel/PycharmProjects/CIND820_Project/CIND820_Project')
crime = pd.read_csv('cleanedcommunitiescrime.csv', sep=',')
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
#The target variable "ViolentCrimesPerPop" was divided into 5 quantile categorical variables.
x = crime.drop('ViolentCrimesPerPop', axis = 1)
y = crime['ViolentCrimesPerPop']
y = pd.qcut(y, q=5, labels=['Very Low Crime', 'Low Crime', 'Medium Crime', 'High Crime', 'Very High Crime'])
#Data was split into training (0.7) and test (0.3) data. The following preprocessing procedures and modelling will occur on the training data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.3, random_state=345)
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
x_train['OtherPerCap'] = pd.to_numeric(x_train['OtherPerCap'], errors='coerce')
###Converted the missing/null values into column median values.
x_train = x_train.fillna(0)
x_train = x_train.replace(0, x_train.median())
#Conducted Yeo and Johnson transformation on the dataset because numerous features were skewed distributions.
#In order to improve the predictor variables and possibly the model, this transformation was applied.
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
x_train = pd.DataFrame(pt.fit_transform(x_train), columns=x_train.columns)
#The feature selection embedded method Lasso was applied to the preprocessed training dataset.
#The top 10 features were selected.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
importances = rfc.feature_importances_
selection = x_train.columns
sorted_importances = pd.Series(importances, index=selection).sort_values(ascending=False)
top_features = sorted_importances[:10]
selected_features = list(top_features.index)
x_train_transform = x_train[selected_features]
x_test_transform = x_test[selected_features]
print(selected_features)
#k-fold cross validation was used to evaluate the Support Vector Classifier (SVC) model against 20 split samples.
#The average accuracy score was determined as a performance metric.
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=1)
model = SVC()
score = cross_val_score(model, x_train_transform, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
mean_score = sum(score)/60
model.fit(x_train_transform, y_train)
print(mean_score)
y_pred = model.predict(x_test_transform)
mcc = matthews_corrcoef(y_test, y_pred)
print(mcc)
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(x_test_transform)))''', number=0)
print(t)