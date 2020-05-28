# Import additional libraries required for analysis

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

# Tuning the Random Forest's Hyperparameters
seed = 123
rf = RandomForestRegressor(random_state = seed)

# Inspect the random forest's hyperparameters
rf.get_params()

# Define the grid for hyperparameters
params_rf = {
    'n_estimators':[20,25,50],
    'max_depth':[5,10,15],
    'min_samples_leaf': [0.1,0.2],
    'min_samples_leaf':[10,20,40]
}

# Instantiate grid search for the random forest

grid_rf = GridSearchCV(estimator = rf, param_grid = params_rf, cv=3,verbose = 1,n_jobs = -1 )
grid_rf.fit(X_train,Y_train)

# Extract best hyperparameters from grid_rf

best_hyperparams = grid_rf.best_params_
print('Best parameters for the RF model is :\n', best_hyperparams)
# estimators : 50 , max_depth = 15 , min_samples_leaf = 10

# Extract best model from the grid_rf
best_model = grid_rf.best_estimator_
# Predict the model on the validation set
y_val = best_model.predict(X_valid)
# Evaluate the validation RMSE
rmse_validation = MSE(Y_valid,y_val)**(1/2)
# The validation RMSE is :
print( 'Validation set RMSE for the RF model is: {:.2f}'.format(rmse_validation))

# Prepare file for submission
Y_test_pred = best_model.predict(X_test).clip(0, 20)

submission_rf = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test_pred
})
submission_rf.to_csv('rf_submission.csv', index=False)

# Score on leaderboard : 0.93
