# encoding=utf-8
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
data_path = 'data/international-airline-passengers.csv'
footer = 3
seed = 7
look_back = 3
epochs = 100
batch_size = 1


def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i: i+look_back, 0]
        dataX.append(x)
        y = dataset[i+look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)


def build_model():
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(1, look_back)))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    np.random.seed(seed)
    data = read_csv(data_path, usecols=[1], engine='python', skipfooter=footer)
    dataset = data.values.astype('float32')
    # 标准化数据
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.67)
    valid_size = len(dataset) - train_size
    train, valid = dataset[0: train_size, :], dataset[train_size: len(dataset), :]
    # 创建dataset, 让数据产生相关性
    X_train, y_train = create_dataset(train)
    X_valid, y_valid = create_dataset(valid)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1]))

    model = build_model()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    predict_train = model.predict(X_train)
    predict_valid = model.predict(X_valid)
    predict_train = scaler.inverse_transform(predict_train)
    y_train = scaler.inverse_transform([y_train])
    predict_valid = scaler.inverse_transform(predict_valid)
    y_valid = scaler.inverse_transform([y_valid])

    train_score = math.sqrt(mean_squared_error(y_train[0], predict_train[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    valid_score = math.sqrt(mean_squared_error(y_valid[0], predict_valid[:, 0]))
    print('Validation Score: %.2f RMSE' % valid_score)

    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train

    predict_valid_plot = np.empty_like(dataset)
    predict_valid_plot[:, :] = np.nan
    predict_valid_plot[len(predict_train) + look_back * 2 + 1:len(dataset) - 1, :] = predict_valid

    dataset = scaler.inverse_transform(dataset)
    plt.plot(dataset, color='blue')
    plt.plot(predict_train_plot, color='green')
    plt.plot(predict_valid_plot, color='red')
    plt.show()
