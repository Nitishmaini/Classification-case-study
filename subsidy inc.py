import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import train_test_split from sklearn
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.linear_model import LogisticRegression
data_income = pd.read_csv('income.csv')
data=data_income.copy()
data.info()
data.isnull().sum() # There is no null values
# summary of numerical data 
summary_num=data.describe()
print(summary_num)
# summary of categorical variables
summary_cat=data.describe(include='O')
print(summary_cat)
data['JobType'].value_counts() # There is ? which is representing the missing value but we did not count it as Nan value so we have to fix it
data['EdType'].value_counts() # there is no missing value
data['occupation'].value_counts() # Similar as jobType
# checking for unique value
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

# Now we have to read ? value as Nan value so
data=pd.read_csv('income.csv',na_values=[" ?"])
# now check again about missing value
data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
#------points to notice
# 1) Missing values in jobtype=1809
# 2) Missing value in Occupation=1816
# 3) There are 1809 rows wheretwo specific columns i.e occupation and jobtype both have missing value
# 4) you still have occupation unfilled because job type is never worked
# 5) so best thing is drop all the data with missing value
 
data2=data.dropna(axis=0)
data3=data.dropna(axis=0)


# Relationship between inderpendent variable
Correlation=data3.corr()
# In our case non of the variable is coorelated with each other beacuse value is not near 1 or -1
# Relationship between categorical variables using cross tables and data visualization
data2.columns
gender=pd.crosstab(index=data2['gender'],columns='count',normalize=True)
print(gender)
# Gender vs salary status
gender_salstat=pd.crosstab(index=data2['gender'],columns=data2['SalStat'],normalize='index',margins=True)
print(gender_salstat)
# Frequency distribution of salary status
SalStat=sns.countplot(data2['SalStat'])
# 75% of people earn less than 50000
# where 25% of people earn greater than 50000

# Histogram of age
sns.distplot(data2['age'], bins=10,kde=False)
# People with age in between 20-45  are high in frequency
# Box plot Age vs salary status
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()
# people with age 35-50 are more likely to earn greater than 50000
# people with age 25-35 are more likely to earn less than 50000
sns.countplot(y="JobType",hue="SalStat",data=data2)
Job_sal=pd.crosstab(index=data2['JobType'],columns=data2['SalStat'],normalize='index',margins=True)
print(Job_sal)
#From the above table it is visible that 56000 self employed people earn more than 50000
# Hence it is the important variable in avoiding the misuse of subsidy
sns.countplot(y="EdType",hue="SalStat",data=data2)
Edu_sal=pd.crosstab(index=data2['EdType'],columns=data2['SalStat'],normalize='index',margins=True)
print(Edu_sal)
# From the above table we can see that people who have done Doctorate, Masters and Prof school are more likely to earn above 50000 Hence it is influensing variablein avoiding the misuse of subsides
sns.countplot(y="occupation",hue="SalStat",data=data2)
occ_sal=pd.crosstab(index=data2['occupation'],columns=data2['SalStat'],normalize='index',margins=True)
print(occ_sal)
# Those who make more than 50000 usd per year are more likely to work as mangers and professionalsheance an important variablein avoiding the misuse of subsidies
sns.distplot(data2['capitalgain'], bins=10,kde=False)
sns.distplot(data2['capitalloss'], bins=10,kde=False)
sns.boxplot('SalStat','hoursperweek',data=data2)
data2.groupby('SalStat')['hoursperweek'].median()
# From the plot it is clearly seen that those who make more than 50000 are more likely to spend 40 to 50 hours per week. this variable can contribute classyfying the individual salary status since there is association betwwen salary status and hours per week

#       logistic regression  
# Reindexing the salary status as 0 and 1
data3['SalStat']=data3['SalStat'].map({" less than or equal to 50,000":0," greater than 50,000":1})
print(data3['SalStat']) # integer encoding
# Using the pandas function f=get dummies we can convert categorical variable into dummy variable which is know as one hot encoding
# It is used to spliting the column which has categorical data in no. of columns depends on no of category
new_data=pd.get_dummies(data3,drop_first=True)
# column_list stores the list of all column names
column_list=list(new_data.columns)
print(column_list)
# Seprating the input variables from the data
features=list(set(column_list)-set(['SalStat']))

# storing the output variable in y
y=new_data['SalStat'].values

# Storing the input variable in x
x=new_data[features].values
# split the data into train amd test data
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0) 
# Make an istance of the model
logistic=LogisticRegression(solver='lbfgs')

# Fitting the model for x and y
logistic.max_iter=1000000

logistic.dual=False
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# prediction from test data
prediction=logistic.predict(test_x)
print(prediction)
 # confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

# calculate the accuracy
accuracy_score=accuracy_score(test_y,prediction)
accuracy_score

# .......logestic regression removing the insignificant variables.............
 # reindexing the salary status names 0, 1
data3['SalStat']=data3['SalStat'].map({" less than or equal to 50,000" : 0," greater than 50,000":1})
print(data3['SalStat'])

cols=['gender','nativecountry','JobType','race']
new_data1=data3.drop(cols,axis=1)
new_data1=pd.get_dummies(new_data1,drop_first=True)

column_list=list(new_data1.columns)
print(column_list)
# Seprating the input variables from the data
features=list(set(column_list)-set(['SalStat']))
print(features)
# storing the output variable in y
y=new_data1['SalStat'].values

# Storing the input variable in x
x=new_data1[features].values

# split the data into train amd test data
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0) 

# Make an instance of the model
logistic=LogisticRegression(solver='lbfgs')

# Fitting the model for x and y
logistic.max_iter=1000000

logistic.dual=False
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# prediction from test data
prediction1=logistic.predict(test_x)
print(prediction1)
test_y
prediction1
from sklearn.metrics import confusion_matrix
confusion_matrix1=confusion_matrix(test_y,prediction1)
print(confusion_matrix1)

# calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score1=accuracy_score(test_y,prediction)
accuracy_score1

# We can see there is no such difference between previous one and this one after removing the unnessary variable

# lets solve using Knn algorithm
# ........................knn.......................
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
prediction2=KNN_classifier.predict(test_x)
confusion_matrix2=confusion_matrix(test_y,prediction2)
print(confusion_matrix2)

accuracy_score2=accuracy_score(test_y,prediction2)
accuracy_score2

# We just randomly choose the k value so
# calculating error for k value between 1 to 20'
for i in range(1, 20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    print((test_y!=pred_i).sum())

print(Misclassified_sample)
# So at k=10 no of missclassified value is 1420 which is minimum so we can use k=10

from sklearn.neighbors import KNeighborsClassifier
KNN_classifier=KNeighborsClassifier(n_neighbors=10)
KNN_classifier.fit(train_x,train_y)
prediction2=KNN_classifier.predict(test_x)
confusion_matrix2=confusion_matrix(test_y,prediction2)
print(confusion_matrix2)

accuracy_score2=accuracy_score(test_y,prediction2)
accuracy_score2
# previsoly accuracy was 0.8341253177146646 and now=0.8430765830478506 
# So there is a little change 