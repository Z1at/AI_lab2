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

tf.random.set_seed(40)

models = []
history = []
rem = []
loss = ["mse", "mae"]
optimizer = ["adam"]
metric = ["mae", "mse"]
for neurons in range(100, 301, 50):
    model = ks.Sequential(
        [Dense(X_train.shape[1]),
         Dense(neurons),
         Dense(1)]
    )
    for l in loss:
        for o in optimizer:
            for m in metric:
                model.compile(loss=l, optimizer=o, metrics=[m])
                models.append(model)
                for epoch in range(10, 51, 20):
                    for batch in range(10, 51, 10):
                        history.append(model.fit(X_train, y_train, batch_size=batch, epochs=epoch))
                        rem.append([l, o, m, epoch, batch, neurons])

for i in models:
    scores = i.evaluate(X_val, y_val, verbose=1)

preds = []
for i in models:
    preds.append(i.predict(test_edited))

for i in range(len(preds)):
    output = pd.DataFrame(
        {
            'Id': test_data['Id'],
            "loss": rem[i][0],
            "optimizer": rem[i][1],
            "metric": rem[i][2],
            "epoch": rem[i][3],
            "batch": rem[i][4],
            "neurons": rem[i][5],
            'SalePrice': np.squeeze(preds[i])
        })
    pd.DataFrame(history[i].history).plot()
    plt.ylabel('Meaning')
    plt.xlabel('Epochs')
    print(history[i].history)
    plt.show()
    print()

    print(output)