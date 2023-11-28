import pandas as pd
import numpy
from scipy import stats
import os
os.chdir('C:/Users/bitel/PycharmProjects/CIND820_Project/CIND820_Project')
crime = pd.read_csv('cleanedcommunitiescrime.csv', sep= ',')
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
crime['OtherPerCap'] = pd.to_numeric(crime['OtherPerCap'], errors='coerce')
crime2 = crime.apply(stats.shapiro)
print(crime2)
crime2.to_csv('normalitytests.csv')
