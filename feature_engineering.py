# Load some additional libraries for analysis
import gc
from tqdm import tqdm_notebook

# Loading the dataframe on which I want to do feature generation
new_train = pd.read_csv('../data/train_modified_features.csv')
new_train.head()

# Generate Lag Features
def generate_lag(train, months, lag_column):
    for month in months:
        # Speed up by grabbing only the useful bits
        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]
        train_shift['date_block_num'] += month
        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return train
  
# Change the dtypes of the columns of the dataset from 'float64' to 'float32' and 'int64' to 'int32'

def downcast_dtypes(df):
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols]   = df[int_cols].astype(np.int16)
    
    return df
 
new_train = downcast_dtypes(new_train)
gc.collect()

# Create a lag for the target variable (item_cnt_month) for the last 1,2,3,4,6,12 months 
new_train = generate_lag(new_train, [1,2,3,4,6,12], 'item_cnt_month')

# Generate mean encoded feature for monthly target-item mean 
# Only consider a lag of one month 
group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')
new_train = generate_lag(new_train, [1], 'item_month_mean')
new_train.drop(['item_month_mean'], axis=1, inplace=True)

# Generate mean encoded feature for monthly shop-target mean
# Take lag for one month only
group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')
new_train = generate_lag(new_train, [1], 'shop_month_mean')
new_train.drop(['shop_month_mean'], axis=1, inplace=True)


# Generate the mean encoded feature for monthly item_cnt_month
# Take lag for one month only
group = new_train.groupby(['date_block_num'])['item_cnt_month'].mean().rename('count_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num'], how='left')
new_train = generate_lag(new_train, [1], 'count_month_mean')
new_train.drop(['count_month_mean'], axis=1, inplace=True)

# Generate the mean encoded feature for monthly item_cnt_month by category_id
# Take lag for one month only
group = new_train.groupby(['date_block_num','item_category_id'])['item_cnt_month'].mean().rename('item_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num','item_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'item_category_month_mean')
new_train.drop(['item_category_month_mean'], axis=1, inplace=True)

# Generate the mean encoded feature for item_cnt_month by shop and item_category_id
# Take lag for one month only
group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'shop_category_month_mean')
new_train.drop(['shop_category_month_mean'], axis=1, inplace=True)

# Generate the mean encoded feature for item_cnt_month by main_category
# Take lag for one month only
group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'main_category_month_mean')
new_train.drop(['main_category_month_mean'], axis=1, inplace=True)

# Generate mean encoded feature for item_cnt_month by sub_type_category
# Take a lag of one month
group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'sub_category_month_mean')
new_train.drop(['sub_category_month_mean'], axis=1, inplace=True)

# Generate the mean encoded feature for item_cnt_month by shop_id and main_category
group = new_train.groupby(['date_block_num','shop_id', 'main_category_id'])['item_cnt_month'].mean().rename('shop_main_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num','shop_id', 'main_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'shop_main_category_month_mean')
new_train.drop(['shop_main_category_month_mean'], axis=1, inplace=True)

# Generate the mean encoded feature for item_cnt_month by shop_id and sub_category 
# Take lag of one month

group = new_train.groupby(['date_block_num','shop_id', 'sub_category_id'])['item_cnt_month'].mean().rename('shop_sub_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num','shop_id', 'sub_category_id'], how='left')
new_train = generate_lag(new_train, [1], 'shop_sub_category_month_mean')
new_train.drop(['shop_sub_category_month_mean'], axis=1, inplace=True)

# Generate the mean encoded feature for item_cnt_month by city_label
# Take lag of one month
group = new_train.groupby(['date_block_num', 'city_label'])['item_cnt_month'].mean().rename('city_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'city_label'], how='left')
new_train = generate_lag(new_train, [1], 'city_month_mean')
new_train.drop(['city_month_mean'], axis=1, inplace=True)

# Generate mean encoded feature for item_cnt_month by city_label and item_id
# Take lag of one month
group = new_train.groupby(['date_block_num','item_id', 'city_label'])['item_cnt_month'].mean().rename('city_item_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num','item_id', 'city_label'], how='left')
new_train = generate_lag(new_train, [1], 'city_item_month_mean')
new_train.drop(['city_item_month_mean'], axis=1, inplace=True)

# Generate mean encoded feature for item_price by item
# Take a lag of one month
group = train.groupby(['item_id'])['item_price'].mean().rename('item_price_mean').reset_index()
new_train = pd.merge(new_train, group, on=['item_id'], how='left')

# Generate mean encoded feature for item_price by item_id and month
# Take a lag of 1,2,3,4 and 6 months
group = train.groupby(['date_block_num','item_id'])['item_price'].mean().rename('item_price_mean_month').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num','item_id'], how='left')
new_train = generate_lag(new_train, [1,2,3,6], 'item_price_mean_month')
new_train.drop(['item_price_mean_month'], axis=1, inplace=True)

# Generate a feature for month
new_train['month'] = new_train['date_block_num'] % 12

# Add features for the number of holidays in a month
holiday_dict = {
    0: 6,
    1: 3,
    2: 2,
    3: 8,
    4: 3,
    5: 3,
    6: 2,
    7: 8,
    8: 4,
    9: 8,
    10: 5,
    11: 4,
}

new_train['holidays_in_month'] = new_train['month'].map(holiday_dict)

# Generate a feature for months since first sale for each shop/item pair and for item only
new_train['item_shop_first_sale'] = new_train['date_block_num'] - new_train.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
new_train['item_first_sale'] = new_train['date_block_num'] - new_train.groupby('item_id')['date_block_num'].transform('min')

# Prepare the dataset for further analysis

# Filter out values for the first year
new_train = new_train[new_train['date_block_num']>11]

# Fill NULL values with zeros
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            df[col].fillna(0, inplace=True)         
    return df

new_train = fill_na(new_train)

# Save the generated dataset to a pickle file
new_train.to_pickle('data.pkl')
del new_train
gc.collect()




