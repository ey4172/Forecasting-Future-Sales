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
    'min_child_weight': 300,
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

# Tuning the hyperparameter max_depth and min_child_weight 
# Max depth is the depth of the constructed tree, deep trees are more complex and may tend to overfit
# Min_child_weight is the minimum weight required in order to create a new node in the tree. Small min_child_weight 
# creates children with fewer samples and complex trees that may overfit

# Define the range of the grid search parameters - max_depth and min_child_weight 

gridsearch_params = [(max_depth,min_child_weight) for max_depth in range(9,12) for min_child_weight in range(250,310,10)]

# Run cross validation on the parameters
min_mse = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mse'},
        early_stopping_rounds=10
    )
    
    # Update best MSE
    mean_mse = cv_results['test-mse-mean'].min()
    boost_rounds = cv_results['test-mse-mean'].argmin()
    print("\tMSE {} for {} rounds".format(mean_mse, boost_rounds))
    if mean_mse < min_mse:
        min_mse = mean_mse
        best_params = (max_depth,min_child_weight)
        
print("Best params: {}, {}, MSE: {}".format(best_params[0], best_params[1], min_mse))
# The best parameter values for max_depth and min_child_weight was 10 and 300 respectively

# Tune parameters subsample and colsample_bytree 
# Subsample : number of rows to sample from the dataset. By default the value is set to one, indicating that all rows will be used
# Colsample_bytree : number of features to sample from dataset. Default value = 1, indicating that all columns will be used

# Define gridsearch parameters
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_mse = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mse'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_mse = cv_results['test-mse-mean'].min()
    boost_rounds = cv_results['test-mse-mean'].argmin()
    print("\tMSE {} for {} rounds".format(mean_mse, boost_rounds))
    if mean_mse < min_mse:
        min_mse = mean_mse
        best_params = (subsample,colsample)
print("Best params: {}, {}, MSE: {}".format(best_params[0], best_params[1], min_mse))

# Best parameter value for subsample and colsample was 0.8. 
# These tuned parameters are used in the construction of the final XGBOOST model.  
