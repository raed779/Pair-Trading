import glob, os
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import argrelextrema
from datetime import datetime
from numpy import hstack
from statsmodels.genmod.families.links import probit
from statsmodels.tsa.stattools import coint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed



#*************************get min max*************************************************************
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
    #maximums.sort()
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

#*********************************************************************************************

#*********************************get date****************************************************

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

#******************************************************************************************


#************************show the difference between 2 company*****************************
fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['dataa\FDX (1)'], color='green', label='FDX')
plt.plot(stock_data['dataa\AMZN'], color='purple', label='AMAZON')
ax.grid(True)
plt.legend(loc=2)
plt.show()
#*******************************************************************************************



#******************************some types of finance analytics*****************************

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

#*******************************************************************************************

smoothing = 3
window = 10
print("eeeeeeeeeeeeeeee")
print(len(stock_data['dataa\AMZN']))
minimums = find_minimums(stock_data['dataa\AMZN'], 6)
print(minimums)
maximums = find_maximums(stock_data['dataa\AMZN'], 6)
print(maximums)


#**********************************SPLIT FUNCTION*******************************************
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


#******************************************LSTM*********************************************

# define input sequence
in_seq1 = array(minimums)
in_seq2 = array(maximums)
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
print(in_seq1)
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
# choose a number of time steps
print(dataset)
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=5000, verbose=0)
# demonstrate prediction
#x_input = array([[minimums[len(minimums)-6], maximums[len(maximums)-4]], [minimums[len(minimums)-5], maximums[len(maximums)-3]], [minimums[len(minimums)-4], maximums[len(maximums)-2]]])
x_input = array([[minimums[3], maximums[3]], [minimums[4],  maximums[4]], [minimums[6],  maximums[6]]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat[0][0])
cc = yhat[0][0].reshape((2, 1))
print(cc[0])


x_input2 = array([[minimums[9],  maximums[9]], [minimums[10],  maximums[10]], [minimums[11],  maximums[11]]])
x_input2 = x_input2.reshape((1, n_steps_in, n_features))
yhat2 = model.predict(x_input2, verbose=0)
print(yhat2[0][0])
cc2 = yhat2[0][0].reshape((2, 1))

x_input3 = array([[minimums[12],  maximums[12]], [minimums[13],  maximums[13]], [minimums[14],  maximums[14]]])
x_input3 = x_input3.reshape((1, n_steps_in, n_features))
yhat3 = model.predict(x_input3, verbose=0)
print(yhat3[0][0])
cc3 = yhat3[0][0].reshape((2, 1))



fig, ax = plt.subplots(figsize=(12,5))
plt.plot(stock_data['dataa\AMZN'] , color='purple', label='AMZN')
#plt.scatter(minmax['date'],minmax['dataa\AMZN'],color='blue',alpha=.5)
ax.grid(True)
ax.axhline(y=0, color='black', linestyle='-')
ax.axhline(y=cc[0], color='green', linestyle='-')
ax.axhline(y=cc[1], color='red', linestyle='-')
ax.axhline(y=cc2[0], color='green', linestyle='-')
ax.axhline(y=cc2[1], color='red', linestyle='-')
ax.axhline(y=cc3[0], color='green', linestyle='-')
ax.axhline(y=cc3[1], color='red', linestyle='-')
"""ax.axhline(y=193.985185, color='green', linestyle='-')
ax.axhline(y=308.835716, color='green', linestyle='-')
ax.axhline(y=276.14268380487806, color='red', linestyle='-')
ax.axhline(y=457.11200880000007, color='red', linestyle='-')"""
plt.legend(loc=2)
plt.show()