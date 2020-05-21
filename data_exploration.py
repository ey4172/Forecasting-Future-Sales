# Load the required libraries for the project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from datetime import datetime
from sklearn import preprocessing
from itertools import product
import gc
import altair as alt 
import os
import xgboost as xgb
from xgboost import XGBRegressor
import time
import sys
import gc
import pickle

# Load the datasets utilized in the project
# Path where the files are located 
path ='../input/competitive-data-science-predict-future-sales/'

# Item Categories Dataset: This contains the item category id as well as the name of the item category
item_categories = pd.read_csv(path+'item_categories.csv')

# Items Dataset: This item ID, the name of the item as well as the item category ID
items = pd.read_csv(path+'items.csv')

# Shops Dataset: This contains the shop Id and the shop name
shops = pd.read_csv(path+'shops.csv')

# Test Dataset: forecast the item sales per month for each unique shop and unique item
test = pd.read_csv(path+'test.csv')

# Sample submission dataset: This dataset contains the ID along with the amount of items sold per month
sample_submission = pd.read_csv(path+'sample_submission.csv')

# Training Dataset:
# Columns: 
# date- contains the date values between january 2013 and october 2015
# date_block_num - contains the consecutive month numbers; jan 2013 is 0; feb 2013 is 1; march 2013 is 2 ...
# shop_id - unique identifier of a shop
# item_id - unique identifier of an item
# item_price - current price of an item
# item_cnt_per_day - number of items sold in a day

sales_train = pd.read_csv(path+'sales_train.csv')

# Data Exploration

#------------------------------------------------------------------------------------------------------------------------------------------
# Item Categories Dataset
# Check for duplicate, NULL and incorrect datatypes columns 
item_categories.info()
item_categories['item_category_id'] = pd.Categorical(item_categories.item_category_id)
duplicated_item_categories = item_categories[item_categories.duplicated()]
print(duplicated_item_categories)
# Closer inspection of this dataset tells us that the item category name contains a main type and a subtype. For example the 
# main type can be Playstation and subtype can be PS3, PS4 etc. 
# I spilt the item category name into maintype and subtype by a hyphen. For items that do not have subtype names,the type names are used

# Split the category_item_name by '-'
main_sub_categories =  item_categories['item_category_name'].str.split('-')
# A column for main_category
item_categories['main_category'] = main_sub_categories.map(lambda row: row[0].strip())
# A column for sub_category
item_categories['sub_category'] = main_sub_categories.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())
# Creating columns for the label encoded values
le = preprocessing.LabelEncoder()
item_categories['main_category_id'] = le.fit_transform(item_categories['main_category'])
item_categories['sub_category_id'] = le.fit_transform(item_categories['sub_category'])
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# Items Dataset
# Conduct checks for NULL, missing and duplicate values 
items.info()
items['item_id'] = pd.Categorical(items.item_id)
items['item_category_id'] = pd.Categorical(items.item_category_id)
items[items.duplicated()]

#-----------------------------------------------------------------------------------------------------------------------------------------------------

# Shops Dataset
# Conducting the same checks
shops.info()
shops['shop_id'] = pd.Categorical(shops.shop_id)
shops['shop_name'] = shops['shop_name'].astype('str')
shops[shops.duplicated()]
# Closer inspection of the dataset, indicates some problems with duplicate values:
# Shop Ids 0 (!Якутск Орджоникидзе, 56 фран) and Shops 57(Якутск Орджоникидзе, 56) - Presence of an extra 'фран'
# Shop Id 1 (!Якутск ТЦ "Центральный" фран) and 58 (Якутск ТЦ "Центральный")-  Extra 'фран'
# Shop Id 10 ( Жуковский ул. Чкалова 39м?) and 11 (Жуковский ул. Чкалова 39м²) - Presence of special characters in the end

# Make changes in the shop id values in the shop_id in the shops, train and test dataset
def change_shop_id (dataset):
    dataset.loc[dataset['shop_id'] == 0, 'shop_id'] = 57
    dataset.loc[dataset['shop_id'] == 1, 'shop_id'] = 58
    dataset.loc[dataset['shop_id'] == 10, 'shop_id'] = 11
    
change_shop_id(shops)
change_shop_id(sales_train)
change_shop_id(test)

# Each shop name starts with the city name. Extract it and label encode for model
# Extract the city name
shops['city_name'] = shops['shop_name'].str.split(' ').map(lambda x : x[0])

# Correct the name of an incorrect shop name
shops.loc[shops['city_name'] == '!Якутск', 'city_name'] = 'Якутск'

# Label Encode the city name column for analysis
shops['city_name_code'] = le.fit_transform(shops['city_name'])

#-------------------------------------------------------------------------------------------------------------------------------------












