# Forecasting Future Sales 
A Russian Software Company wishes to forecast its future sales for effective revenue and inventory management. In this case study, I first forecast the total sales of the company for the next five months using traditional statistical time series models. Next, I predict the sales for every product and store for the next one month using tree ensemble methods such as the Random Forest and XGBOOST. The overall goal of the project is to minimize the root mean squared error loss for the models I apply on the provided data.

### DATA
1. Training Data: 'sales_train.csv'
  * Columns: Date , cummulative date , shop_id , item_id , item_price , item_cnt_day (items sold per day)

2. Sample Submission Data: 'sample_submission.csv'

3. Test Dataset: 'test.csv'

4. Shops Dataset: 'shops.csv' 
  * Columns: shop_id, shop_name

5. Items Dataset: 'items.csv'
  * Columns: item_id, item_category_id, item_name
  
6. Item Categories Dataset: 'item_categories.csv'
  * Columns: item_category_id, item_category_name 


