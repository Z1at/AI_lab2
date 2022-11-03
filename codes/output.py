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

tf.random.set_seed(40)

# loss - mse, mae
# optimizer - sgd, adam
# metrics - mae, mse

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mae"])
model1.compile(loss="mse", optimizer="adam", metrics=["mae"])
models = [model, model1]

history = [model.fit(X_train, y_train, batch_size=10, epochs=30), model1.fit(X_train, y_train, batch_size=10, epochs=30)]

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