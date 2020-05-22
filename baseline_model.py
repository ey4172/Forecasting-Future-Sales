# Before making any fancy models on the provided dataset, I create a simple basline model that will use the previous month's predictions
# for a given shop_id and item_id pair found in both the training and test dataset. If a pair isn't found in the train set, then
# the sales count for that item-shop pair is predicted as zero.

# Loading the libraries for the baseline model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time

from math import sqrt
from numpy import loadtxt
from itertools import product
from tqdm import tqdm
from sklearn import preprocessing
from xgboost import plot_tree
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the given datasets
sales_train = pd.read_csv('../data/sales_train.csv')
items = pd.read_csv('../data/items.csv')
shops = pd.read_csv('../data/shops.csv')
item_categories = pd.read_csv('../data/item_categories.csv')
test = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')

# Create a grid for all Shop IDs and Product IDs for a given month 
grid = []
for block_num in sales_train['date_block_num'].unique():
    cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales_train[sales_train['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Clip the values of number of items sold per day to be between (0,20) and do the necessary aggregations
sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20)
groups = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])
trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})
trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)
trainset = pd.merge(grid,trainset,how='left',on=index_cols)
trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)

# Merge the trainset with items to get the item category Id
trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')
trainset.to_csv('trainset_with_grid.csv')

# Creating the benchmark model
train_subset = trainset[trainset['date_block_num'] == 33]
groups = train_subset[['shop_id', 'item_id', 'item_cnt_month']].groupby(by = ['shop_id', 'item_id'])
train_subset = groups.agg({'item_cnt_month':'sum'}).reset_index()

# Using the previous months observations to predict the outcomes for the coming months
merged = test.merge(train_subset, on=["shop_id", "item_id"], how="left")[["ID", "item_cnt_month"]]
merged.head()
merged['item_cnt_month'] = merged.item_cnt_month.fillna(0).clip(0,20)
submission = merged.set_index('ID')

# Submit the file to the Kaggle platform
submission.to_csv('benchmark.csv')

