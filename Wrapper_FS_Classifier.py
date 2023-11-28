#Time and memory measures were collected right from the beginning of the code.
import time
import tracemalloc
start_time = time.time()
tracemalloc.start()
import os
import pandas as pd

#importing a csv file of communities and crime after removing the features from raw dataset.
os.chdir('C:/Users/bitel/PycharmProjects/CIND820_Project/CIND820_Project')
crime = pd.read_csv('cleanedcommunitiescrime.csv', sep=',')

#First column counts the number of observations. Removed first column.
#Removed target variable column from predictor variables
x = crime.drop(crime.columns[[0, 95]], axis = 1)

#The target variable "ViolentCrimesPerPop" was divided into 5 quantile categorical variables.
y = crime['ViolentCrimesPerPop']
y = pd.qcut(y, q=5, labels=['Very Low Crime', 'Low Crime','Medium Crime', 'High Crime', 'Very High Crime'])

#Data was split into training (0.7) and test (0.3) data. The following preprocessing procedures and modelling will occur on the training data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.3, random_state=225)

#One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value.
#Change feature to numeric value.
x_train['OtherPerCap'] = pd.to_numeric(x_train['OtherPerCap'], errors='coerce')

###Converted the missing/null values into column median values.
x_train = x_train.fillna(x_train.median())

#The feature selection wrapper method forward selection was applied to the preprocessed training dataset.
#The top 10 features were selected.
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
selection = SequentialFeatureSelector(SVC(kernel='rbf'), direction='forward', n_features_to_select=10)
selection.fit(x_train, y_train)
newx_train = selection.transform(x_train)
selected_features = x_train.columns[selection.get_support()].tolist()
print("Top 10 Selected Features:", selected_features)

#Use the trained data to determine it's accuracy scores based on zero cross validation.
#The model used to classify the trained data was the Support Vector Classifier (SVC).
#Accuracy measure will be used to determine variance score.
from sklearn.svm import SVC
from sklearn.metrics import classification_report
model = SVC()
model.fit(newx_train, y_train)
print("zero-fold scores:", classification_report(y_train, model.predict(newx_train)))

#k-fold cross validation was used to evaluate the Support Vector Classifier (SVC) model against 20 split samples.
#The average accuracy score was determined as a performance metric.
#The average variance from the 10-folds was calculated using the average accuracy score.
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = SVC()
score = cross_val_score(model, newx_train, y_train, scoring= 'accuracy', cv=cv, n_jobs=-1)
mean_score = sum(score)/30
print("R10-fold mean accuracy score:", mean_score)
print("R10-Fold accuracy variance:", mean_score*(1-mean_score))

#Computed matthew_corrcoef to determine MCC score of the test data.
from sklearn.metrics import matthews_corrcoef
model.fit(newx_train, y_train)
newx_test = selection.transform(x_test)
y_pred = model.predict(newx_test)
mcc = matthews_corrcoef(y_test, y_pred)
print("MCC score:", mcc)

#In addition, anoher classification report was used to evaluate the performance of the model against the test data.
print("Test data performance report:", classification_report(y_test, model.predict(newx_test)))

#End of time and memory tracking.
snapshot = tracemalloc.take_snapshot()
end_time = time.time()
tracemalloc.stop()
memory = snapshot.statistics('lineno')
for stat in memory[:10]:
    print("Memory Used:", stat)
print("Duration of execution (sec):", end_time - start_time)