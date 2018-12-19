# encoding=utf-8
# 用DNN做回归预测，训练X:t-3, t-2, t-1; y: t
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

data_path = 'data/international-airline-passengers.csv'
footer = 3
seed = 7
look_back = 3
epochs = 400
batch_size = 2


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
    model.add(Dense(units=12, input_dim=look_back, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    np.random.seed(seed)
    data = read_csv(data_path, usecols=[1], engine='python', skipfooter=footer)
    dataset = data.values.astype('float32')
    train_size = int(len(dataset) * 0.67)
    valid_size = len(dataset) - train_size
    train, valid = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

    X_train, y_train = create_dataset(train)
    X_valid, y_valid = create_dataset(valid)

    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    train_score = model.evaluate(X_train, y_train, verbose=1)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
    valid_score = model.evaluate(X_valid, y_valid, verbose=1)
    print('Validation Score: %.2f MSE (%.2f RMSE)' % (valid_score, math.sqrt(valid_score)))

    predict_train = model.predict(X_train)
    predict_valid = model.predict(X_valid)

    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train

    predict_valid_plot = np.empty_like(dataset)
    predict_valid_plot[:, :] = np.nan
    predict_valid_plot[len(predict_train) + look_back * 2 + 1:len(dataset) - 1, :] = predict_valid

    plt.plot(dataset, color='blue')
    plt.plot(predict_train_plot, color='green')
    plt.plot(predict_valid_plot, color='red')
    plt.show()


