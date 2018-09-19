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


with open('normal_labels', 'rb') as fp:
    normal_labels=pickle.load(fp)

with open('ill_labels', 'rb') as fp:
    ill_labels=pickle.load(fp)


normal_new_labels=[]
for l in normal_labels:
    normal_new_labels.append(np.array([1,0]))
normal_new_labels=np.array(normal_new_labels)


ill_new_labels=[]
for l in ill_labels:
    ill_new_labels.append(np.array([0,1]))

ill_new_labels=np.array(ill_new_labels)

print(normal_new_labels.shape)
print(ill_new_labels.shape)

with open('ill_new_labels', 'wb') as fp:
    pickle.dump(ill_new_labels, fp)


with open('normal_new_labels', 'wb') as fp:
    pickle.dump(normal_new_labels, fp)
