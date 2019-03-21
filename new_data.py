# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:52:58 2019

@author: Kartik
"""
##import the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## import the Dataset

dataset = pd.read_csv("Hackthon_case_training_data.csv")
price=pd.read_csv("Price_index.csv")
price_ind=price.set_index('id')

## Function for counting Missing data in every Column for the Dataframe........................

def counts(dataset):
    for i in range(len(dataset.columns)) :
        print(i,"Data empty in",dataset.columns[i], " : " ,  dataset.iloc[:,i].isnull().sum())
###.............................................................................................    

## Cleaning the Dataframe 

dataset['date_activ']=pd.to_datetime(dataset['date_activ'])
dataset['date_end']=pd.to_datetime(dataset['date_end'])
dataset['date_first_activ']=pd.to_datetime(dataset['date_first_activ'])

dataset['num_days']=dataset['date_end']-dataset['date_activ']
dataset.dropna(subset =['date_end'] , axis =0 , inplace=True)
dataset.drop(columns='campaign_disc_ele', inplace=True)


activ= pd.get_dummies(dataset['activity_new'])

data_out = pd.read_csv("Hackthon_case_training_output.csv")
datastream2=data_out.set_index("id")

##piechart for exsiting churn data

plt.pie(data_out['churn'].value_counts() , labels=["False", "True"], colors=["yellow","red"], explode = (0,0.1),autopct='%1.1f%%', shadow=True, startangle=140)
plt.suptitle("Percentage of Churn Costumers")
plt.axis('equal')
plt.tight_layout()
plt.show()
## for supervised feature dataset..

datastream1=dataset.set_index("id")

data= pd.merge(datastream1,datastream2 ,left_index=True ,right_index=True, how='inner')
data= pd.merge(data,price_ind ,left_index=True ,right_index=True, how='inner')

channels=pd.get_dummies(data['channel_sales'])
chanel=data['channel_sales'].value_counts().idxmax()
data['channel_sales'].replace(np.nan , chanel , inplace=True)

activity_max=data['activity_new'].value_counts().idxmax()
data['activity_new'].replace(np.nan , activity_max , inplace=True)

data['has_gas'].replace('f' , 0 , inplace=True)
data['has_gas'].replace('t' , 1, inplace=True)
data['date_first_activ'].replace(np.nan , data['date_activ'] , inplace=True)


## Generating the Suitable Data range for missing data
forcast_dis_en_av = data['forecast_discount_energy'].astype('float').mean(axis=0)
forcast_price_en_p1_av=data['forecast_price_energy_p1'].astype('float').mean(axis=0)
forcast_price_en_p2_av = data['forecast_price_energy_p2'].astype('float').mean(axis=0)
forcast_price_pow_p1_av = data['forecast_price_pow_p1'].astype('float').mean(axis=0)
margin_gross_pow_av= data['margin_gross_pow_ele'].mean(axis=0)
power_max_av = data['pow_max'].astype('float').mean(axis=0)
margin_net_pow_av=data['margin_net_pow_ele'].astype('float').mean(axis=0)
net_margin_av=data['net_margin'].astype('float').mean(axis=0)
price_p1_var_av=data['price_p1_var'].mean(axis=0)
price_p2_var_av=data['price_p2_var'].mean(axis=0)
price_p3_var_av=data['price_p3_var'].mean(axis=0)
price_p3_fix_av=data['price_p3_fix'].mean(axis=0)

price_p2_fix_av=data['price_p2_fix'].mean(axis=0)

price_p1_fix_av=data['price_p1_fix'].mean(axis=0)


## Fixing the Missing values

data['pow_max'].replace(np.nan , power_max_av , inplace=True)
data['margin_gross_pow_ele'].replace(np.nan , margin_gross_pow_av , inplace=True)
data['forecast_discount_energy'].replace(np.nan , forcast_dis_en_av , inplace=True)
data['forecast_price_energy_p1'].replace(np.nan , forcast_price_en_p1_av, inplace=True)
data['forecast_price_energy_p2'].replace(np.nan , forcast_price_en_p2_av, inplace=True)
data['forecast_price_pow_p1'].replace(np.nan, forcast_price_pow_p1_av, inplace=True)
data['margin_net_pow_ele'].replace(np.nan , margin_net_pow_av , inplace=True)
data['net_margin'].replace(np.nan , net_margin_av , inplace=True)
data['price_p1_var'].replace(np.nan , price_p1_var_av, inplace=True)

data['price_p2_var'].replace(np.nan , price_p2_var_av, inplace=True)


data['price_p3_var'].replace(np.nan , price_p3_var_av, inplace=True)


data['price_p1_fix'].replace(np.nan , price_p1_fix_av, inplace=True)
data['price_p2_fix'].replace(np.nan , price_p2_fix_av, inplace=True)
data['price_p3_fix'].replace(np.nan , price_p3_fix_av, inplace=True)




orig=pd.get_dummies(data['origin_up'])
data['origin_up'].replace(np.nan ,data['origin_up'].value_counts().idxmax() , inplace=True)


## Finding the correlation between Different Columns

corrrel= data.corr(method='pearson')



##Decluding the Less Required Column Vectors

data.drop(columns=['forecast_base_bill_ele','forecast_base_bill_year','forecast_bill_12m','forecast_cons','origin_up'] , inplace=True)




dataframe1=pd.concat([data,orig,channels], axis=1)

dataframe1['num_days']=dataframe1['num_days'].dt.days



correlation=dataframe1.corr()



dataframe1.drop(columns=['num_days'], inplace=True)

##Exporting Clean Dataset

dataframe1.to_csv('DataFrame_cleaned.csv')


## for the unsupervised feature data..

## Cleaning the Dataset

datastream1=dataset.set_index("id")
data1 = pd.merge(datastream1,datastream2 ,left_index=True ,right_index=True, how='outer')
data1= pd.merge(data1,price_ind ,left_index=True ,right_index=True, how='outer')


channels=pd.get_dummies(data1['channel_sales'])
chanel=data1['channel_sales'].value_counts().idxmax()
data1['channel_sales'].replace(np.nan , chanel , inplace=True)

activity_max=data1['activity_new'].value_counts().idxmax()
data1['activity_new'].replace(np.nan , activity_max , inplace=True)

data1['has_gas'].replace('f' , 0 , inplace=True)
data1['has_gas'].replace('t' , 1, inplace=True)
data1['date_first_activ'].replace(np.nan , data1['date_activ'] , inplace=True)

## Generating the Suitable Values for missing data

forcast_dis_en_av = data1['forecast_discount_energy'].astype('float').mean(axis=0)
forcast_price_en_p1_av=data1['forecast_price_energy_p1'].astype('float').mean(axis=0)
forcast_price_en_p2_av = data1['forecast_price_energy_p2'].astype('float').mean(axis=0)
forcast_price_pow_p1_av = data1['forecast_price_pow_p1'].astype('float').mean(axis=0)
margin_gross_pow_av= data1['margin_gross_pow_ele'].mean(axis=0)
power_max_av = data1['pow_max'].astype('float').mean(axis=0)
margin_net_pow_av=data1['margin_net_pow_ele'].astype('float').mean(axis=0)
net_margin_av=data1['net_margin'].astype('float').mean(axis=0)
price_p1_var_av=data1['price_p1_var'].mean(axis=0)
price_p2_var_av=data1['price_p2_var'].mean(axis=0)
price_p3_var_av=data1['price_p3_var'].mean(axis=0)
price_p3_fix_av=data1['price_p3_fix'].mean(axis=0)

price_p2_fix_av=data1['price_p2_fix'].mean(axis=0)

price_p1_fix_av=data1['price_p1_fix'].mean(axis=0)


## Fixing the Missing value

data1['pow_max'].replace(np.nan , power_max_av , inplace=True)
data1['margin_gross_pow_ele'].replace(np.nan , margin_gross_pow_av , inplace=True)
data1['forecast_discount_energy'].replace(np.nan , forcast_dis_en_av , inplace=True)
data1['forecast_price_energy_p1'].replace(np.nan , forcast_price_en_p1_av, inplace=True)
data1['forecast_price_energy_p2'].replace(np.nan , forcast_price_en_p2_av, inplace=True)
data1['forecast_price_pow_p1'].replace(np.nan, forcast_price_pow_p1_av, inplace=True)
data1['margin_net_pow_ele'].replace(np.nan , margin_net_pow_av , inplace=True)
data1['net_margin'].replace(np.nan , net_margin_av , inplace=True)
data1['price_p1_var'].replace(np.nan , price_p1_var_av, inplace=True)

data1['price_p2_var'].replace(np.nan , price_p2_var_av, inplace=True)


data1['price_p3_var'].replace(np.nan , price_p3_var_av, inplace=True)


data1['price_p1_fix'].replace(np.nan , price_p1_fix_av, inplace=True)
data1['price_p2_fix'].replace(np.nan , price_p2_fix_av, inplace=True)
data1['price_p3_fix'].replace(np.nan , price_p3_fix_av, inplace=True)


orig=pd.get_dummies(data1['origin_up'])
data1['origin_up'].replace(np.nan ,data1['origin_up'].value_counts().idxmax() , inplace=True)


## Correlation For the dataframe columns
corrrel= data1.corr(method='pearson')




## Dropping the Less required Column Vectors

data1.drop(columns=['forecast_base_bill_ele','forecast_base_bill_year','forecast_bill_12m','forecast_cons','origin_up'] , inplace=True)

data1.drop(columns=['num_days'], inplace=True)
dataframe=pd.concat([data1,orig,channels], axis=1)


correlation=dataframe.corr()

datast=dataframe[dataframe['churn'].isnull()]

## Exporting the Final data for Prediction

datast.to_csv('Check_new.csv')