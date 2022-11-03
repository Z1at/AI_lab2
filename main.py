import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow import keras as ks
from keras.layers import Dense


def missing_value_checker(data):
    list = []
    for feature, content in data.items():
        if data[feature].isnull().values.any():
            sum = data[feature].isna().sum()
            type = data[feature].dtype
            print(f'{feature}: {sum}, type: {type}')
            list.append(feature)
    print(list)
    print(len(list))


def nan_filler(data):
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content):
            data[label] = content.fillna(content.median())
        else:
            data[label] = content.astype("category").cat.as_ordered()
            data[label] = pd.Categorical(content).codes + 1


def read_data(path, name):
    return pd.read_csv(os.path.join(path, name))


train_data = read_data("", 'train.csv')
test_data = read_data("", 'test.csv')

# print(train_data.head())
# print(test_data.head())

missing_value_checker(test_data)

test_edited = test_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
train_edited = train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

nan_filler(test_edited)
nan_filler(train_edited)

missing_value_checker(test_edited)
missing_value_checker(train_edited)

test_edited.info()
train_edited.info()

X = train_edited.drop('SalePrice', axis=1)
y = train_edited['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# first - кол-во полносвязных слоёв, кол-во нейронов, кол-во выходов
model = ks.Sequential(
    [Dense(X_train.shape[1]),
     Dense(100),
     Dense(1)]
)

model1 = ks.Sequential(
    [Dense(X_train.shape[1]),
     Dense(100),
     Dense(1)]
)

model2 = ks.Sequential(
    [Dense(X_train.shape[1]),
     Dense(100),
     Dense(1)]
)

model3 = ks.Sequential(
    [Dense(X_train.shape[1]),
     Dense(200),
     Dense(1)]
)

model4 = ks.Sequential(
    [Dense(X_train.shape[1]),
     Dense(200),
     Dense(1)]
)

model5 = ks.Sequential(
    [Dense(X_train.shape[1]),
     Dense(200),
     Dense(1)]
)

tf.random.set_seed(40)

# loss - mse, mae
# optimizer - sgd, adam
# metrics - mae, mse

model.compile(loss=ks.losses.MSE, optimizer='adam', metrics=['mse'])
model1.compile(loss='mse', optimizer='adam', metrics=['mae'])
model2.compile(loss='huber', optimizer='adam', metrics=['mae'])
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae'])
model4.compile(loss='mae', optimizer='sgd', metrics=['mae', 'accuracy'])
model5.compile(loss='mse', optimizer='adam', metrics=['mae'])
models = [model, model1, model2, model3, model4, model5]

history = [model.fit(X_train, y_train, batch_size=5, epochs=50), model1.fit(X_train, y_train, batch_size=20, epochs=30),
           model2.fit(X_train, y_train, batch_size=6, epochs=30), model3.fit(X_train, y_train, batch_size=5, epochs=50),
           model4.fit(X_train, y_train, batch_size=5, epochs=30),
           model5.fit(X_train, y_train, batch_size=10, epochs=50)]

# model1.compile(loss="mse", optimizer="adam", metrics=["mae"])
# model.compile(loss="mse", optimizer="adam", metrics=["mae"])
# model2.compile(loss="mse", optimizer="adam", metrics=["mae"])
# models = [model2, model, model1]
#
# history = [model2.fit(X_train, y_train, batch_size=5, epochs=10), model1.fit(X_train, y_train, batch_size=5, epochs=30),
#            model.fit(X_train, y_train, batch_size=5, epochs=50)]

for i in history:
    pd.DataFrame(i.history).plot()
    plt.ylabel('Meaning')
    plt.xlabel('Epochs')
    print(i.history)
    plt.show()

for i in models:
    scores = i.evaluate(X_val, y_val, verbose=1)

preds = []
for i in models:
    preds.append(i.predict(test_edited))

for i in preds:
    output = pd.DataFrame(
        {
            'Id': test_data['Id'],
            'SalePrice': np.squeeze(i)
        })
    print(output)
