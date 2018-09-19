import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import csv
from matplotlib import style
import string
from collections import Counter
import sys
import pickle
from sklearn.model_selection import train_test_split
import random

import keras
from keras.layers import Input, Dense,Dropout, Flatten, Reshape, LSTM
from keras.models import Model

from keras.models import load_model
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

with open('normal_features', 'rb') as fp:
    normal_features=pickle.load(fp)

with open('normal_new_labels', 'rb') as fp:
    normal_labels=pickle.load(fp)


with open('ill_features', 'rb') as fp:
    ill_features=pickle.load(fp)

with open('ill_new_labels', 'rb') as fp:
    ill_labels=pickle.load(fp)


X=[]
y=[]

for f in normal_features:
    X.append(f)

for f in ill_features:
    X.append(f)

X=np.array(X)
print(X.shape)


for l in normal_labels:
    y.append(l)

for l in ill_labels:
    y.append(l)

y=np.array(y)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

inputs = Input(shape=(15,7680))

lstm=LSTM(128)(inputs)
resh=Reshape((32,4))(lstm)

lstm=LSTM(256)(resh)
resh=Reshape((64,4))(lstm)


flat=Flatten()(resh)

dense=Dense(64,activation='linear')(flat)
dense=Dense(128,activation='linear')(dense)
dense=Dense(264,activation='linear')(dense)
dense=Dense(512,activation='linear')(dense)
dense=Dense(1024,activation='linear')(dense)
dense=Dense(2048,activation='linear')(dense)
dense=Dense(4096,activation='linear')(dense)


output = Dense(2,activation='softmax')(dense)

model = Model(inputs=inputs,outputs=output)
model.summary()
adam=keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam ,loss='binary_crossentropy',metrics=['acc'])
model.fit(X_train,y_train, batch_size=8, epochs=50,
shuffle=False,validation_data=(X_test,y_test))
model.save('lstm.h5')