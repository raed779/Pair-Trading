# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

"""plt.figure()
plt.plot(allFiles["Open"])
plt.plot(allFiles["High"])
plt.plot(allFiles["Low"])
plt.plot(allFiles["Close"])
plt .title('Historique des cours GE')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open', 'High', 'Low', 'Close'] , loc = 'en haut à gauche')
plt.show()"""

"""
plt.figure(figsize=(12,5))
ax1 = stock_data['dataa\FDX (1)'].plot(color='green', grid=True, label='FDX')
ax2 = stock_data['dataa\AMZN'].plot(color='purple', grid=True, secondary_y=True, label='AMAZON')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

plt.legend(h1+h2, l1+l2, loc=2)
plt.show()
"""

"""
def find_patterns(max_min):
    patterns = defaultdict(list)

    # Window range is 5 units
    for i in range(5, len(max_min)):
        window = max_min.iloc[i - 5:i]

        # Pattern must play out in less than n units
        if window.index[-1] - window.index[0] > 100:
            continue

        a, b, c, d, e = window.iloc[0:5]

        # IHS
        if a < b and c < a and c < e and c < d and e < d and abs(b - d) <= np.mean([b, d]) * 0.02:
            patterns['IHS'].append((window.index[0], window.index[-1]))

    return patterns

patterns = find_patterns(minmax)
print(patterns)"""

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


# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([61, 73, 85])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)