# Visualize the generated revenue by month and year

sales_m_y = pd.DataFrame(sales_train['revenue'].groupby([sales_train['date_block_num']]).agg('sum'))
sales_m_y = sales_m_y.reset_index()
#sales_m_y.head()
sales_my_plot = sns.relplot(x = 'date_block_num', y = 'revenue', data = sales_m_y, kind='line')
sales_my_plot.set(xlabel = 'Month & Year', ylabel ='Total Revenue')
plt.show()

# Sales peaks were observed during the 11th and 23rd month

# Visualize generated revenue for different months each year
sales_m = pd.DataFrame(sales_train['revenue'].groupby([sales_train.Year,sales_train.Month]).agg('sum'))
sales_m = sales_m.reset_index()
df2013 = sales_m[sales_m['Year'] == '2013']
df2014 = sales_m[sales_m['Year'] == '2014']
df2015 = sales_m[sales_m['Year'] == '2015']
c = sns.catplot(x='Month', y='revenue', data= df2013,kind = 'bar', ci= None)
c.set(xlabel = 'Month', ylabel= 'Total Revenue for 2013')
plt.show()

c2 = sns.catplot(x='Month', y='revenue', data= df2014,kind = 'bar', ci= None)
c2.set(xlabel = 'Month', ylabel= 'Total Revenue for 2014')
plt.show()

c3 = sns.catplot(x='Month', y='revenue', data= df2015,kind = 'bar', ci= None)
c3.set(xlabel = 'Month', ylabel= 'Total Revenue for 2015')
plt.show()

# Both for 2013 and 2014, maximum sales occur during the months of November and December because this time frame coincides with the 
# holidays

# Visualize the generated revenue during the weekdays and weekends by different months and years
sales_weekend = pd.DataFrame(sales_train['revenue'].groupby([sales_train.Year,sales_train.Month,sales_train.is_weekend]).agg('sum'))
sales_weekend = sales_weekend.reset_index()
sns.catplot(x='Month', y='revenue', data= sales_weekend[sales_weekend['Year'] == '2013'],kind = 'bar', ci= None,hue = 'is_weekend')
sns.catplot(x='Month',y = 'revenue', data = sales_weekend[sales_weekend['Year'] == '2014'],kind ='bar',ci = None,hue = 'is_weekend')
sns.catplot(x='Month',y= 'revenue', data = sales_weekend[sales_weekend['Year'] == '2015'],kind ='bar',ci=None, hue = 'is_weekend')

# Visualize the generated revenue during the different days of the week by year
# Change the aggregate function to check the trends 
sales_weekdays = pd.DataFrame(sales_train['revenue'].groupby([sales_train.Year,sales_train.day_of_week]).agg('sum'))
sales_weekdays = sales_weekdays.reset_index()
weekday_2013 = sns.catplot(x = 'day_of_week', y = 'revenue', data = sales_weekdays[sales_weekdays['Year'] == '2013'],kind ='bar', ci = None)
weekday_2013.set(xlabel = 'Day of the Week',ylabel = 'Total Revenue (2013)')
weekday_2014 = sns.catplot(x = 'day_of_week', y = 'revenue', data = sales_weekdays[sales_weekdays['Year'] == '2014'],kind ='bar', ci = None)
weekday_2014.set(xlabel = 'Day of the week',ylabel = 'Total Revenue (2014)')
weekday_2015 = sns.catplot(x = 'day_of_week', y = 'revenue', data = sales_weekdays[sales_weekdays['Year'] == '2015'],kind ='bar', ci = None)
weekday_2015.set(xlabel = 'Day of the week',ylabel = 'Total Revenue (2015)')

# Shop Ids and names with the maximum generated revenues
temp = sales_train[['revenue','item_cnt_day','shop_id','shop_name']]
temp['shop_id'] = temp['shop_id'].astype(str)
temp['shop_name_id'] = temp[['shop_id','shop_name']].apply(lambda x: ' '.join(x), axis = 1)
shopids = pd.DataFrame(temp['revenue'].groupby([temp.shop_name_id]).agg('sum'))
shopids = shopids.sort_values('revenue', ascending = False)
shopids = shopids.reset_index()
shopids['shop_name_id'] = pd.Categorical(shopids['shop_name_id'])
shop_id_name_plot = sns.catplot(x = 'shop_name_id' , y = 'revenue', data = shopids, kind = 'bar',height = 3, aspect = 4 , order= shopids['shop_name_id'])
shop_id_name_plot.set_xticklabels(rotation=90)

# Shop Ids and names with total sales
shopids = pd.DataFrame(temp['item_cnt_day'].groupby([temp.shop_name_id]).agg('sum'))
shopids = shopids.sort_values('item_cnt_day', ascending = False)
shopids = shopids.reset_index()
shopids['shop_name_id'] = pd.Categorical(shopids['shop_name_id'])
shop_id_name_plot = sns.catplot(x = 'shop_name_id' , y = 'item_cnt_day', data = shopids, kind = 'bar',height = 3, aspect = 4 , order= shopids['shop_name_id'])
shop_id_name_plot.set_xticklabels(rotation=90)

# From this we can see that the shop_ids 31, 25, 28, 42, 54 have the maximum revenue& sales and a lot of these shops
# are located in the city MOCKBA

# Item Ids and names with maximum revenue
temp = sales_train[['revenue','item_cnt_day','item_id','item_name']]
temp['item_id'] = temp['item_id'].astype(str)
temp['item_name_id'] = temp[['item_id','item_name']].apply(lambda x: ' '.join(x), axis = 1)
itemids = pd.DataFrame(temp['revenue'].groupby([temp.item_name_id]).agg('sum'))
itemids = itemids.sort_values('revenue', ascending = False)
itemids = itemids.reset_index()
itemids = itemids.head(50)
itemids['item_name_id'] = pd.Categorical(itemids['item_name_id'])
item_id_name_plot = sns.catplot(x = 'item_name_id' , y = 'revenue', data = itemids, kind = 'bar',height = 3, aspect = 4 , order= itemids['item_name_id'])
item_id_name_plot.set_xticklabels(rotation=90)

# Item Ids and names with highest sales
itemids = pd.DataFrame(temp['item_cnt_day'].groupby([temp.item_name_id]).agg('sum'))
itemids = itemids.sort_values('item_cnt_day', ascending = False)
itemids = itemids.reset_index()
itemids = itemids.head(50)
itemids['item_name_id'] = pd.Categorical(itemids['item_name_id'])
item_id_name_plot = sns.catplot(x = 'item_name_id' , y = 'item_cnt_day', data = itemids, kind = 'bar',height = 3, aspect = 4 , order= itemids['item_name_id'])
item_id_name_plot.set_xticklabels(rotation=90)

# From the two charts above, we observe that the maximum revenue is generated from PS4's, games that generate the maximum 
# sales was however was a bit different

# Item categories that generate the maximum revenue 
temp = sales_train[['revenue','item_cnt_day','item_category_id','item_category_name']]
temp['item_category_id'] = temp['item_category_id'].astype(str)
temp['item_cat_name_id'] = temp[['item_category_id','item_category_name']].apply(lambda x: ' '.join(x), axis = 1)
item_cat_ids = pd.DataFrame(temp['revenue'].groupby([temp.item_cat_name_id]).agg('sum'))
item_cat_ids = item_cat_ids.sort_values('revenue', ascending = False)
item_cat_ids = item_cat_ids.reset_index()
item_cat_ids = item_cat_ids.head(50)
item_cat_ids['item_cat_name_id'] = pd.Categorical(item_cat_ids['item_cat_name_id'])
item_cat_id_name_plot = sns.catplot(x = 'item_cat_name_id' , y = 'revenue', data = item_cat_ids, kind = 'bar',height = 3, aspect = 4 , order= item_cat_ids['item_cat_name_id'])
item_cat_id_name_plot.set_xticklabels(rotation=90)

# Item category names and ids that were sold the most
item_cat_ids = pd.DataFrame(temp['item_cnt_day'].groupby([temp.item_cat_name_id]).agg('sum'))
item_cat_ids = item_cat_ids.sort_values('item_cnt_day', ascending = False)
item_cat_ids = item_cat_ids.reset_index()
item_cat_ids = item_cat_ids.head(50)
item_cat_ids['item_cat_name_id'] = pd.Categorical(item_cat_ids['item_cat_name_id'])
item_cat_id_name_plot = sns.catplot(x = 'item_cat_name_id' , y = 'item_cnt_day', data = item_cat_ids, kind = 'bar',height = 3, aspect = 4 , order= item_cat_ids['item_cat_name_id'])
item_cat_id_name_plot.set_xticklabels(rotation=90)

# Price categorization
def price_categorization(x):
    if  0 < x <= 500:
        return 'cheap'
    elif  500 < x <= 1000:
        return 'moderate'
    elif 1000 < x <= 3000:
        return 'high'
    else:
        return 'luxury'
      
      
# Create a column for price categories
sales_train['price_categories'] = sales_train['item_price'].apply(price_categorization)

# What type of items and under which category are sold the most?
cat_type_most_bought  = pd.DataFrame(sales_train['item_cnt_day'].groupby([sales_train['price_categories']]).agg('sum'))
cat_type_most_bought = cat_type_most_bought.reset_index()
cat_type_most_bought.columns
sns.catplot(x = 'price_categories', y = 'item_cnt_day', data = cat_type_most_bought, kind = 'bar')

# Cheap items were bought the most by customers and expensive items the least

# What type of items and under which category generated the most revenue?
cat_type_most_sales  = pd.DataFrame(sales_train['revenue'].groupby([sales_train['price_categories']]).agg('sum'))
cat_type_most_sales = cat_type_most_sales.reset_index()
cat_type_most_sales.columns
sns.catplot(x = 'price_categories', y = 'revenue', data = cat_type_most_sales, kind = 'bar')

# High priced and luxury items generated the most amount of revenue

# How does the demand for different types of items vary during different times of the year?
sales_m_y_cat = pd.DataFrame(sales_train['item_cnt_day'].groupby([sales_train['date_block_num'],sales_train['price_categories']]).agg('sum'))
sales_m_y_cat = sales_m_y_cat.reset_index()
sales_m_y_cat
sales_my_cat_plot = sns.relplot(x='date_block_num', y = 'item_cnt_day', data = sales_m_y_cat, kind = 'line', style = 'price_categories', hue = 'price_categories')
plt.show()

# How does the generated revenue from different types of goods vary during different times of the year
sales_m_y_cat = pd.DataFrame(sales_train['revenue'].groupby([sales_train['date_block_num'],sales_train['price_categories']]).agg('sum'))
sales_m_y_cat = sales_m_y_cat.reset_index()
sales_m_y_cat
sales_my_cat_plot = sns.relplot(x='date_block_num', y = 'revenue', data = sales_m_y_cat, kind = 'line', style = 'price_categories', hue = 'price_categories')
plt.show()

# Create a matrix as a product of item/shop pairs within each month in the train dataset
grid = []
months = sales_train['date_block_num'].unique()
for month in months:
    shops_in_month = sales_train.loc[sales_train['date_block_num']==month, 'shop_id'].unique()
    items_in_month = sales_train.loc[sales_train['date_block_num']==month, 'item_id'].unique()
    grid.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))
    
cartesian_df = pd.DataFrame(np.vstack(grid), columns = ['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
# Aggregate sales to a monthly level and clip the target variable
x = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()

# Merge the cartesian dataframe with the aggregated sales dataframe for analysis
new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)
new_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)

# Append the test dataset to the training set
test.head()

# Add the column for cummulative month for the given dataset
test['date_block_num'] = 34
test['item_cnt_month'] = 0
# Delete the ID column from the dataset
del test['ID']
new_train = new_train.append(test)

# Add the city_name code to the new training dataset for further analysis
# Join the two dataframes by the shop_id and then proceed
new_train = pd.merge(new_train, shops.drop(['shop_name','city_name'],axis = 1), on = ['shop_id'], how = 'left')

# Add the item_category_id to the dataset from the items dataframe and joining on the item_id column
new_train = pd.merge(new_train, items.drop(['item_name'],axis = 1), on = ['item_id'], how = 'left')

# Add the main_category_id and sub_category_id to the dataset by merging on the item_category_id
new_train = pd.merge(new_train, item_categories.drop(['item_category_name','main_category','sub_category'], axis = 1), on = ['item_category_id'], how = 'left')

# Generate the csv file for this data
new_train.to_csv('train_data_features.csv')

# This new file will be loaded and feature extraction / engineering will be done to derive insights from the data 
