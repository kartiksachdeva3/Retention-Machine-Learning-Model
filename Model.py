# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:39:51 2019

@author: Kartik
"""

## importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Importing the Cleaned data to DataFrame

dataframe2=pd.read_csv("DataFrame_cleaned.csv")
dataframe2=dataframe2.set_index("id")

### Fucntion to Check the Index of the dataset columns (for ease use)
def index_num(dataframex):
    for i in dataframex.columns :
        print(i, " " ,dataframex.columns.get_loc(i))

## Including the Explicative Variable vectors for training and testing object vecotr

x=dataframe2.iloc[:, [2,3,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]].values

x.astype('float')

y=dataframe2.loc[:,'churn'].values

# Crossvalidation of the Data for training and testing purposes
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , t_test =train_test_split(x,y,random_state=7,test_size=0.25)

## Scaling the instances to Normalized dataset
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.transform(x_test)

## Building the classifier for the LogisticRegression Model 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

## Getting the predictions for the test instances
y_pred=classifier.predict(x_test)

##Scoring the Classifier

print("Accuracy = " ,(classifier.score(x_test,t_test))*100,"%", )

## Deducing the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(t_test, y_pred)


## dictating the outputs to the final : Dictonary
unique, counts = np.unique(y_pred, return_counts=True)
final= dict(zip(unique, counts))

## piechart plot for the Training and testing data

plt.pie(counts , labels=["False", "True"], colors=["yellow","red"], explode = (0,0.1),autopct='%1.1f%%', shadow=True, startangle=140)
plt.suptitle("Percentage of Churn Costumers for next 3 months (Training Set)")
plt.axis('equal')
plt.tight_layout()
plt.show()

## Importing the Unsupervised Dataset for the Model Preditions

check_data=pd.read_csv("Check_new.csv")
check_data=check_data.set_index("id")
datafr_final=check_data

## Including the explicative Column vectors for Prediction

Final_X=check_data.iloc[:, [2,3,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]].values

Final_X.astype('float')
## predicting the new values with the help of build Classifier

datafr_final['y_pred1'] =classifier.predict(Final_X)

datafr_final.drop(datafr_final.columns[0:44], axis=1 , inplace=True )

datafr_final.rename(columns={'y_pred1' : 'Churn_predict'} , inplace=True)

##Binding the Predicted values to Dictonary Format

unique1, counts1 = np.unique(datafr_final['Churn_predict'], return_counts=True)
final1= dict(zip(unique1, counts1))
print(final1)

## Ploting the PieChart for the Fianl Values

plt.pie(counts1 , labels=["False", "True"], colors=["yellow","red"], explode = (0,0.1),autopct='%1.1f%%', shadow=True, startangle=140)
plt.suptitle("Percentage of Churn Costumers in next 3 months ")
plt.axis('equal')
plt.tight_layout()
plt.show()




