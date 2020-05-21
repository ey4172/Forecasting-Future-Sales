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
# Training Dataset
# Conduct basic checks and coerce certain columns to suitable datatypes
sales_train.info()
sales_train[sales_train.duplicated()]
sales_train = sales_train.drop_duplicates()

# Coerce the date column of the dataset into a suitable format, shop_id and item_id into categorical variables 
sales_train['date'] = pd.to_datetime(sales_train['date'],format = '%d.%m.%Y')
sales_train['shop_id'] = pd.Categorical(sales_train.shop_id)
sales_train['item_id'] = pd.Categorical(sales_train.item_id)

# Exploring the item_cnt_day column in the dataset
sns.boxplot(x = sales_train['item_cnt_day'])

# Inspected boxplot shows abnormally high values and thus, rows of these observations are inspected 
# Information of the items that have the top 5 highest item_cnt_day values
sales_train.sort_values(['item_cnt_day'],ascending = [False]).head(5)

# Calculate the median sales of item_id 11373 and 20949
item_11373_median = sales_train[sales_train['item_id'] == 11373]['item_cnt_day'].median()
print(item_11373_median)
item_20949_median = sales_train[sales_train['item_id'] == 20949]['item_cnt_day'].median()
print(item_20949_median)
# Thus the two values present in the dataset are clear outliers and should be removed 
sales_train = sales_train[sales_train['item_cnt_day'] < 1000]
# Negative values in the item_cnt_day column indicates that an item has been returned.

# Item price column
sns.boxplot(x = sales_train['item_price'])
# Information for the top 5 outliers present in the dataset
sales_train.sort_values(['item_price'],ascending = [False]).head(5)
# Calculate the median price of item_id 6066 and 11365
item_6066_median = sales_train[sales_train['item_id'] == 6066]['item_price'].median()
print(item_6066_median)
item_11365_median = sales_train[sales_train['item_id'] == 11365]['item_price'].median()
print(item_11365_median)
# Item ID 6066 is a clear outlier and should be removed 
sales_train = sales_train[sales_train['item_price'] < 100000]
# Observations of rows where the item price is negative
sales_train[sales_train['item_price'] < 0]
# Find the median price for item 2973 and impute the negative value with the median price
price_correction = train[(train['shop_id'] == 32) & (train['item_id'] == 2973) & (train['date_block_num'] == 4) & (train['item_price'] > 0)].item_price.median()
sales_train.loc[sales_train['item_price'] < 0 , 'item_price'] = price_correction
sales_train.head()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Generate the feature columns for revenue (item_price * item_cnt_day) , month, year , day of the week, weekday or weekend

# Revenue
sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']

# Month
sales_train['Month'] = sales_train['date'].dt.strftime('%m')

# Year
sales_train['Year'] = sales_train['date'].dt.strftime('%Y')

# Day of the week
sales_train['day_of_week'] = sales_train['date'].dt.dayofweek

# Is weekend or not
sales_train['is_weekend'] = 0
sales_train.loc[sales_train['day_of_week'].isin([5,6]),'is_weekend'] = 1

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Merging all datasets together for exploratory data analysis

# Merge datasets with the training dataset
# Merge the sales_train dataset with the shops dataset on the shop_id variable to get the shop_name
# Merge the sales_train dataset with the items dataset on the item_id variable to get the item_category_id
# Merge the sales_train dataset with the item_categories dataset to obtain the item category name. Merge on the item_category_id

sales_train = pd.merge(sales_train, shops , on ='shop_id', how = 'left')
sales_train = pd.merge(sales_train,items, on = 'item_id', how = 'left')
sales_train = pd.merge(sales_train, item_categories, on = 'item_category_id', how = 'left')
sales_train.columns

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



