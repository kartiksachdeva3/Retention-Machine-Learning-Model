# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:55:41 2019

@author: Kartik
"""

import pandas as pd
import numpy as np 

data_hist= pd.read_csv("Hackathon_case_training_hist_data.csv")
data_hist=data_hist.set_index("id")


data_finalised=data_hist.groupby(['id']).mean()

data_finalised.to_csv("Price_index.csv")