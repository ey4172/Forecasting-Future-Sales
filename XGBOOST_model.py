# Load the pickle file
data = pd.read_pickle('data.pkl')
gc.collect()

# Split the data into train, validation and test

# Train data for months < 33
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']

# Validation data for month = 33
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']

# Test data for month = 34
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
