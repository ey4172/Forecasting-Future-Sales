# Load the created pickle file 
data = pd.read_pickle('data.pkl')

# Creating train, validation and test datasets
# Train
X_train = data[data.date_block_num < 34].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 34]['item_cnt_month']
# Validation
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
# Test
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

# Loading this data into DMatrices
dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_valid, label=Y_valid)

# Initialize parameters for the XGBoost Model 
# Range of values hint take from discussion boards

params = {
    # Parameters that we are going to tune.
    'max_depth':10,
    'min_child_weight': 250,
    'eta':0.1,
    'subsample': 0.6,
    'colsample_bytree': 0.5,
    # Other parameters
    'objective':'reg:linear',
    'eval_metric' : 'mse'
}

# Number of boosting rounds
num_boost_rounds = 900

# The model 
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
