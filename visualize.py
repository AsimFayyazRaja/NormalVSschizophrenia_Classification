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


with open('normal_features', 'rb') as fp:
    normal_features=pickle.load(fp)


with open('ill_features', 'rb') as fp:
    ill_features=pickle.load(fp)


plt.plot(ill_features[0])
plt.show()