# Import all the libraries required for analysis

# Basic packages
import numpy as np
import pandas as pd 
import random as rd 
import datetime 

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns 

# Time series modules
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima.utils import ndiffs

# Load the dataset
path ='../input/competitive-data-science-predict-future-sales/'
sales_train = pd.read_csv(path+'sales_train.csv')

# Creating the univariate time series 
time_series  = pd.DataFrame(sales_train.groupby(['date_block_num'])['item_cnt_day'].sum())
time_series.index= pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
time_series = time_series.reset_index()
time_series.columns = ['cummulative_month','item_cnt_month']

# Visually diagnosing stationarity
# A time series is stationary when: 
# It isn't a function of time and doesn't have increasing / decreasing trends over time.
# It is homoskedastic and the variance of the time series isn't a function of time.
# The covariance of the ith and (i+m)th term should not be a function of time

# Plotting the total sales across the months 
sns.lineplot(x='cummulative_month',y='item_cnt_month',data = time_series)

# Decompose the trends, seasonalities and residuals given in the time series
ts = time_series['item_cnt_month']
decomposition = sm.tsa.seasonal_decompose(ts, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

# Plot the moving average and standard deviation of time series
roll_mean = ts.rolling(window = 12, center = False).mean()
roll_std = ts.rolling(window = 12, center = False).std()
plt.plot(ts, color = 'blue',label = 'Original Data')
plt.plot(roll_mean, color = 'red', label = 'Rolling Mean')
plt.plot(roll_std, color ='black', label = 'Rolling Std')
plt.xlabel('Time in Months')
plt.ylabel('Total Sales')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

# The rolling mean of the series shows a decreasing trend and the standard deviation also peaks around the times where the sales peak

# Conduct the Augumented Dickey Fuller Test 
# The Null hypothesis of this test assumes that the given time series isn't stationary.
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput) 

# For the given time series, the p-value (0.14) shows weak evidence against the NULL hypothesis.  The test statistic obtained
# also doesn't lie within the critical regions. Thus, we cannot reject the NULL and the time series is not stationary.
# Since the given time series isn't stationary, we need to transform it so that it becomes amenable for modelling purposes

# TIME SERIES TRANSFORMATION
# Aggregation - Taking average for a time period like monthly / weekly / daily
# Smoothing - Taking rolling averages
# Differencing 

# Moving Average Technique
# The average of k consecutive values is taken depending on the frequency of the time series. 
# For the given dataset, consider k = 4 and take the rolling mean for each quarter

# Construct the plot of rolling mean and overlay it with the actual time series
moving_average = ts.rolling(4).mean()
plt.plot(ts)
plt.plot(moving_average, color = 'red')
plt.xlabel('Cummulative Months')
plt.ylabel('Total Sales')
plt.title('Total Sales for the 1C Company')
plt.show()

# Substract the moving average value from the original series. Since the average of the last 4 months has been taken,
# rolling mean is not defined for the first 3 values

ts_moving_average_diff = ts - moving_average
ts_moving_average_diff.head(15)
ts_moving_average_diff.dropna(inplace = True)









