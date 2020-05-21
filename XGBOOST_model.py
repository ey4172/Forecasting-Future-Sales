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

del data
gc.collect()

# Define the model 
model = XGBRegressor(max_depth=8,n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, subsample=0.8, eta=0.3,seed=42)

# Fit the model and evaluate it on the validation and test dataset
model.fit(X_train, Y_train, eval_metric="rmse", eval_set=[(X_train, Y_train), (X_valid, Y_valid)],verbose=True,early_stopping_rounds = 10)

# Plot the feature importance graph
plt.figure(figsize=(20,15))
xgb.plot_importance(model, ax=plt.gca())

# Plot tree built by the XGBOOST algorithm
plt.figure(figsize=(20,15))
xgb.plot_tree(model, ax=plt.gca())

# Save the model 
model.save_model('XGBoost_sales_1.model')

# Predict on provided test dataset
test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
Y_test = model.predict(X_test).clip(0, 20)

# Prepare file for submission
prediction_submission = pd.DataFrame({"ID": test.index, "item_cnt_month": Y_test})
prediction_submission.to_csv('xgb_submission.csv',index = False)
