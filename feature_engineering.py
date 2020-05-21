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

# 



































