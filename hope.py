import glob, os
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import argrelextrema
from datetime import datetime

from statsmodels.tsa.stattools import coint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


from collections import defaultdict


def get_max_min(prices, smoothing, window_range):
    smooth_prices = prices.rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(prices.iloc[i - window_range:i + window_range].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(prices.iloc[i - window_range:i + window_range].idxmin())
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.date.duplicated()]
    p = prices.reset_index()

    max_min['day_num'] = p[p['Date'].isin(max_min.date)].index.values

    max_min = max_min.set_index('day_num')

    return max_min


def find_maximums(data,increment):
    start = 0
    end = increment
    maximums = pd.Series([])
    for i in range(int(len(data)/increment)):
        maximums = maximums.append(pd.Series(int(data[start:end].max())))
        start += increment
        end += increment
    maximums = list(maximums)
    maximums.sort()
    return maximums

def find_minimums(data,increment):
    start = 0
    end = increment
    minimums = pd.Series([])
    for i in range(int(len(data)/increment)):
        minimums = minimums.append(pd.Series(int(data[start:end].min())))
        start += increment
        end += increment
    minimums = list(minimums)
    #minimums.sort(reverse=True)
    return minimums


# find the data directory and extract each CSV file
path = "dataa/"
allFiles = glob.glob(os.path.join(path, "*.csv"))





np_array_list = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col=None, header=0)
    # get symbol name from file
    df['Symbol'] = (file_.split('/')[0]).split(".")[0]
    # pull only needed fields
    df = df[['Symbol', 'Date', 'Adj Close']]
    np_array_list.append(df.as_matrix())
# stack all arrays and tranfer it into a data frame
comb_np_array = np.vstack(np_array_list)
# simplify column names
stock_data_raw = pd.DataFrame(comb_np_array, columns = ['Symbol','Date', 'Close'])
# fix datetime data
stock_data_raw['Date'] = pd.to_datetime(stock_data_raw['Date'], infer_datetime_format=True)
stock_data_raw['Date'] = stock_data_raw['Date'].dt.date

# check for NAs
stock_data_raw = stock_data_raw.dropna(axis=1, how='any')

# quick hack to get the column names (i.e. whatever stocks you loaded)
stock_data_tmp = stock_data_raw.copy()

# make symbol column header
stock_data_raw = stock_data_raw.pivot('Date','Symbol')
stock_data_raw.columns = stock_data_raw.columns.droplevel()
# collect correct header names (actual stocks)
column_names = list(stock_data_raw)

#print(stock_data_raw.tail())


# hack to remove mult-index stuff
stock_data_raw = stock_data_tmp[['Symbol', 'Date', 'Close']]
stock_data_raw = stock_data_raw.pivot('Date','Symbol')
stock_data_raw.columns = stock_data_raw.columns.droplevel(-1)
stock_data_raw.columns = column_names

# replace NaNs with previous value
stock_data_raw.fillna(method='bfill', inplace=True)

print(stock_data_raw.tail())
stock_data = stock_data_raw.copy()


fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['dataa\FDX (1)'], color='green', label='FDX')
plt.plot(stock_data['dataa\AMZN'], color='purple', label='AMAZON')
ax.grid(True)
plt.legend(loc=2)
plt.show()

def corr(data1, data2):
    "data1 & data2 should be numpy arrays."
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()
    corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr

stock_name_1 = 'dataa\AMZN'
stock_name_2 = 'dataa\FDX (1)'

score, pvalue, _ = coint(stock_data[stock_name_1], stock_data[stock_name_2])
correlation = corr(stock_data[stock_name_1], stock_data[stock_name_2])

print('Correlation between %s and %s is %f' % (stock_name_1, stock_name_2, correlation))
print('Cointegration between %s and %s is %f' % (stock_name_1, stock_name_2, pvalue))

print(stock_data[stock_name_2].std())


smoothing = 3
window = 10
"""
minmax = get_max_min(stock_data['dataa\AMZN'], smoothing, window)
print(minmax)
ragnar=minmax['dataa\AMZN'].tolist()
print(sorted(ragnar))"""

minmax_f = find_minimums(stock_data['dataa\AMZN'], 10)
print(minmax_f)
minmvax_f = find_maximums(stock_data['dataa\AMZN'], 10)
print(minmvax_f)





#LSTM
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(minmax_f, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
for i in range(len(X)):
    print(X[i], y[i])
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)

# demonstrate prediction
x_input = array([minmax_f[0], minmax_f[1],minmax_f[2]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)

print(yhat)
print(len(minmax_f))

fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['dataa\AMZN'] , color='purple', label='amazon')
#plt.scatter(minmax['date'],minmax['dataa\AMZN'],color='blue',alpha=.5)
ax.grid(True)
ax.axhline(y=0, color='black', linestyle='-')
ax.axhline(y=yhat, color='green', linestyle='-')
"""ax.axhline(y=193.985185, color='green', linestyle='-')
ax.axhline(y=308.835716, color='green', linestyle='-')
ax.axhline(y=276.14268380487806, color='red', linestyle='-')
ax.axhline(y=457.11200880000007, color='red', linestyle='-')"""
plt.legend(loc=2)
plt.show()

def find_support(data):
    cutoff = 2
    increment = 10

    minimums = find_minimums(data=data,increment=increment)

    histogram = np.histogram(minimums,bins=(int(len(minimums)/increment*increment)))
    histogram_occurences = pd.DataFrame(histogram[0])
    histogram_occurences.columns = ['occurence']
    histogram_splits = pd.DataFrame(histogram[1])
    histogram_splits.columns = ['bins']

    histogram_bins = []
    for x in histogram_splits.index:
        element = []
        if x < len(histogram_splits.index)-1:
            element.append(int(histogram_splits.iloc[x]))
            element.append(int(histogram_splits.iloc[x+1]))
            histogram_bins.append(element)

    histogram_bins = pd.DataFrame(histogram_bins)
    histogram_bins['occurence'] = histogram_occurences
    histogram_bins.columns = ['start','end','occurence']

    histogram_bins = histogram_bins[histogram_bins['occurence'] >= cutoff]
    histogram_bins.index = range(len(histogram_bins))

    data = list(data)
    data.sort()
    data = pd.Series(data)

    lst_minser = []
    for i in histogram_bins.index:
        lst_minser.append(data[(data > histogram_bins['start'][i]) & (data < histogram_bins['end'][i])])

    lst_minser = pd.Series(lst_minser)

    lst_support=[]

    for i in lst_minser.index:
        lst_support.append(lst_minser[i].mean())

    support_df = pd.DataFrame(lst_support)
    support_df.columns = ['support']
    support_df.dropna(inplace=True)
    support_df.index = range(len(support_df))
    support_ser = pd.Series(support_df['support'])
    support_ser = list(support_ser)
    return support_ser



def find_resistance(data):
    cutoff = 2
    increment = 10
    maximums = find_maximums(data=data,increment=increment)

    histogram = np.histogram(maximums,bins=(int(len(maximums)/increment*increment)))
    histogram_occurences = pd.DataFrame(histogram[0])
    histogram_occurences.columns = ['occurence']
    histogram_splits = pd.DataFrame(histogram[1])
    histogram_splits.columns = ['bins']

    histogram_bins = []
    for x in histogram_splits.index:
        element = []
        if x < len(histogram_splits.index)-1:
            element.append(int(histogram_splits.iloc[x]))
            element.append(int(histogram_splits.iloc[x+1]))
            histogram_bins.append(element)

    histogram_bins = pd.DataFrame(histogram_bins)
    histogram_bins['occurence'] = histogram_occurences
    histogram_bins.columns = ['start','end','occurence']

    histogram_bins = histogram_bins[histogram_bins['occurence'] >= cutoff]
    histogram_bins.index = range(len(histogram_bins))

    data = list(data)
    data.sort()
    data = pd.Series(data)

    lst_maxser = []
    for i in histogram_bins.index:
        lst_maxser.append(data[(data > histogram_bins['start'][i]) & (data < histogram_bins['end'][i])])

    lst_maxser = pd.Series(lst_maxser)

    lst_resistance=[]

    for i in lst_maxser.index:
        lst_resistance.append(lst_maxser[i].mean())

    resistance_df = pd.DataFrame(lst_resistance)
    resistance_df.columns = ['resistance']
    resistance_df.dropna(inplace=True)
    resistance_df.index = range(len(resistance_df))
    resistance_ser = pd.Series(resistance_df['resistance'])
    resistance_ser = list(resistance_ser)

    return resistance_ser

crsp =stock_data['dataa\AMZN']
crsp.columns = ['price']

print(find_support(crsp))
print(find_resistance(crsp))