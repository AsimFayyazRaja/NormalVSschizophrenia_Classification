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
from keras.layers import Input, Dense, Flatten,Activation, Reshape,BatchNormalization,SeparableConv2D, MaxPooling2D
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
flat=Flatten()(inputs)

resh=Reshape((64,120,15))(flat)

conv=SeparableConv2D(filters=32,kernel_size=(3,3))(resh)
acti=Activation('relu')(conv)
conv=MaxPooling2D()(acti)

conv=SeparableConv2D(filters=64,kernel_size=(3,3))(conv)
acti=Activation('relu')(conv)
conv=MaxPooling2D()(acti)

conv=SeparableConv2D(filters=128,kernel_size=(3,3))(conv)
acti=Activation('relu')(conv)
conv=MaxPooling2D()(acti)

conv=SeparableConv2D(filters=256,kernel_size=(3,3))(conv)
acti=Activation('relu')(conv)
conv=MaxPooling2D()(acti)

flat=Flatten()(conv)

dense=Dense(64,activation='relu')(flat)
dense=Dense(128,activation='relu')(dense)
dense=Dense(264,activation='relu')(dense)
dense=Dense(512,activation='relu')(dense)
dense=Dense(1024,activation='relu')(dense)
dense=Dense(2048,activation='relu')(dense)
dense=Dense(4096,activation='relu')(dense)


output = Dense(2,activation='softmax')(dense)

model = Model(inputs=inputs,outputs=output)
model.summary()
adam=keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam ,loss='binary_crossentropy',metrics=['acc'])
model.fit(X_train,y_train, batch_size=8, epochs=20,
shuffle=False,validation_data=(X_test,y_test))
model.save('ANN.h5')