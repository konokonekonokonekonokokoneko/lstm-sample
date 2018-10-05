import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(10)

# データ読み込み　Yは最初の列に配置する
dataframe = pandas.read_csv('icecream_sales_2003_2012.csv',
                            usecols=[0, 3, 4, 5, 6], engine='python', skipfooter=1)
# plt.plot(dataframe)
# plt.show()
# print(dataframe.head())

dataset = dataframe.values
dataset = dataset.astype('float32')
# print(dataset)
# print("====================")
# print("====================")
# データセットを正規化します。scikit learnの関数を使ってやってしまいます
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# print(dataset)
# print("====================")
# print("====================")

# これは訓練用のデータtpテスト用のデータを分けているだけ。2/3と1/3にします。
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# print(train)
# print("====================")
# print(test)
# print("====================")
# print("====================")

# これが少し面倒臭いデータの変換になります


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            xset.append(a)
        dataY.append(dataset[i + look_back, 0])
        dataX.append(xset)
    # print(dataX)
    # print("====================")
    # print(dataY)
    # print("====================")
    # print("====================")
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# こんなデータがあったら
# A     B     C     D
# 100   200   300   400
# 101   201   301   401
# ~
# 111   211   311   411
# 112   212   312   412
# 113   213   313   413
# 112の値を、111,211,311,411以下4*12（look_back）のデータを使って予想します
# そんな感じのデータを作ります

# reshape input to be [samples, time steps(number of variables), features] *convert time series into column
# また変換かよ。普通にちゃんと並んだ多次元配列に変換します
trainX = numpy.reshape(
    trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
print(trainX.shape[0])
print(trainX.shape[1])
print(trainX.shape[2])
print(testX.shape[0])
print(testX.shape[1])
print(testX.shape[2])
print("====================")
print("====================")
print("====================")
print(trainX)
print("====================")
print(testX)
print("====================")
print("====================")
print(trainX[0])
print("====================")
print(testX[0])
