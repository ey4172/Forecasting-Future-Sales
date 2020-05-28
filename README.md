# Forecasting Future Sales 
A Russian Software Company wishes to forecast its future sales for effective revenue and inventory management. In this case study, I first forecast the total sales of the company for the next five months using traditional statistical time series models. Next, I predict the sales for every product and store for the next one month using tree ensemble methods such as the Random Forest and XGBOOST. The overall goal of the project is to minimize the root mean squared error loss for the models I apply on the provided data.

### Data
1. **Training Data**: 'sales_train.csv'
   - Columns: Date , cummulative date , shop_id , item_id , item_price , item_cnt_day (items sold per day)

2. **Sample Submission Data**: 'sample_submission.csv'

3. **Test Data**: 'test.csv'

4. **Shops Data**: 'shops.csv' 
   - Columns: shop_id, shop_name

5. **Items Data**: 'items.csv'
   - Columns: item_id, item_category_id, item_name
  
6. **Item Categories Data**: 'item_categories.csv'
   - Columns: item_category_id, item_category_name 

### Files
- **traditional_time_series.py** : contains code for time series visualization, transformation and modelling. 
- **data_exploration.py** : contains code for data cleaning and pre-processing
- **data_visualization.py** : contains code for data visualization
- **feature_engineering.py** : contains code for all the created lag features and mean encodings 
- **baseline_model.py** : contains the simple baseline model for the dataset
- **hyperparameter_tuning_xgboost.py** : contains code for tuning hyperparameters of the XGBOOST model 
- **XGBOOST_model.py**: contains code for building the XGBOOST model with tuned parameters and creation of submission files for evaluation
- **RandomForest.py**: contains code for tuning RF model parameters and the final RF model applied to the test dataset 

### Tools & Packages 
Numpy, pandas, scipy, itertools, statsmodel, matplotlib, seaborn, sklearn, xgboost, altair, pickle, pmdarima 
