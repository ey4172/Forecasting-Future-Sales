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

# Test to check if the given time series is stationary or not
dftest = adfuller(ts_moving_average_diff)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput) 

# Dickey Fuller Test shows strong evidence against the NULL hypothesis. P-value obtained (0.0013) indicates that the time series is
# stationary.

# Exponentially Weighted Moving Average
# The more recent previous values are given more weight compared to the other values
# Consider a span of 12 months for the exponential weighted moving average 

mte_exp_wighted_avg = ts.ewm(span=12,adjust = False).mean()
plt.plot(ts,label = 'Original Time Series')
plt.plot(mte_exp_wighted_avg, color='red', label = 'Exponential Moving Average')
plt.xlabel('Time (months)')
plt.ylabel('Total Sales')
plt.title('Total Sales for the 1C Company')
plt.legend(loc = 'best')
plt.show()

# Test stationarity using the Augumented Dickey Fuller Test
dftest = adfuller(ts_moving_average_diff)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput) 

# The test statistic and p-values indicate that the transformed test series is stationary.

# Differencing 
# It is a common transformation technique, the original observation substracted from the previous k instants.
# This helps in making the series stationary

# Take the first order difference and plot the differenced series with the partial autocorrelation plot
ts_first_diff = ts.diff()
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(ts_first_diff); axes[0].set_title('First Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(ts_first_diff.dropna(), ax=axes[1])
plt.show()

# Dickey Fuller Test
adfuller(ts_first_diff.dropna())
# Confirms stationarity 

# Take seasonal differencing and plot the original series with PACF plot
ts_seasonal_diff = ts.diff(12)
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(ts_seasonal_diff); axes[0].set_title('Seasonal Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(ts_seasonal_diff.dropna(), ax=axes[1])
plt.show()

# Plotting the original and transformed time series with first order and seasonal differencing
fig, axes = plt.subplots(2, 1, figsize=(10,5),sharex=True)

# Usual Differencing
axes[0].plot(ts, label='Original Series')
axes[0].plot(ts.diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper right')


# Seasonal Differencing 
axes[1].plot(ts, label='Original Series')
axes[1].plot(ts.diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper right')
plt.suptitle('1-C Company Monthly Sales')
plt.show()

# Optimal Hyperparameters for the Seasonal ARIMA model
# For our given time series, we will fit a SARIMA(p,d,q)(P,D,Q) model. 
# If a time series is stationary then points in a given point of time would not depend on the previous values.
# It's residuals will resemble white noise
# If a time series isn't stationary then its values will depend on the previous values that have occurred.
# For our given model we have to choose the optimal parameters - AR(p) , MA(q) and d .
# p: It is the number of lags for the dependent variable. If p= 2 then the current value t would depend on the values t-1 and t-2.
# Moving average term (q): These are the lagged errors in the forecast function.
# This allows us to model the errors of our current observation as a linear combination of error values observed at previous values. 
# For eg if q = 2. The given observation t depends on (e-1) and (e-2)th error.
# where e  is the difference in the actual and predicted values 
# d: number of past points to substract from the current value 

# Plotting the series, ACF and PACF plot
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(ts.diff(12)); axes[0].set_title('Seasonal Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(ts.diff(12).dropna(), ax=axes[1])
axes[2].set(ylim =(0,5))
plot_acf(ts.diff(12).dropna(), ax=axes[2])
plt.show()

# Graphically diagnosing and finding optimal parameters is time consuming. Thus, we find the optimal parameters
# using the AUTOARIMA method. This function will fit the SARIMA model over a different combination of parameters and choose the one which minimizes the AIC value. 
# The AIC value measures the goodness of fit of a model relative to its complexity.
# It is maximized for models with low complexity but that generalize to the given data well 

import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(time_series['item_cnt_month'], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()
 
# The coefficients of each of the variables is significant with p-values close to zero.
# The AIC value is 437.635. We plot the model diagnostics to check for unusual behaviour.

smodel.plot_diagnostics(figsize=(7,7))
plt.show()

# The standardized residual plot doesn't show any obvious trends and patterns in the produced diagnostic plots.
# The histogram plus estimated density plot shows us that the kernel density estimation line follows the N(0,1) line. This indicates that the residuals of the model follow the normal distribution.
# The qq-plot also confirms the normality of residuals 
# The produced correlogram plot shows that the time series residuals have low correlation with the lagged versions of itself.

























