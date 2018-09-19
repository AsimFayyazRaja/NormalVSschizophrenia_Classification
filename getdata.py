from tqdm import tqdm
import glob
import numpy as np
import pickle

import os
import os.path as path

normalfiles=os.listdir('normal/')
features=[]
labels=[]
r=0
with tqdm(total=len(normalfiles)) as pbar:
    for file in normalfiles:
        feats=[]
        filename="normal/"+file
        f = open(filename, 'r')     #opening file
        x = f.readline()
        sublist=[]
        i=0
        while x:
            if i<=7679:                     #getting features per channels
                sublist.append(float(x))
            else:
                i=0
                sublist=np.array(sublist)
                feats.append(sublist)
                sublist=[]
                sublist.append(float(x))
            i+=1
            x=f.readline()
        feats=np.array(feats)
        features.append(feats)
        labels.append(1)      #1,0 for nomal
        pbar.update(1)

features=np.array(features)
print(features.shape)

with open('normal_features', 'wb') as fp:
    pickle.dump(features, fp)

labels=np.array(labels)
print(labels.shape)

with open('normal_labels', 'wb') as fp:
    pickle.dump(labels, fp)



illfiles=os.listdir('ill/')
features=[]
labels=[]
r=0
with tqdm(total=len(illfiles)) as pbar:
    for file in illfiles:
        feats=[]
        filename="ill/"+file
        f = open(filename, 'r')     #opening file
        x = f.readline()
        sublist=[]
        i=0
        while x:
            if i<=7679:                     #getting features per channels
                sublist.append(float(x))
            else:
                i=0
                sublist=np.array(sublist)
                feats.append(sublist)
                sublist=[]
                sublist.append(float(x))
            i+=1
            x=f.readline()
        feats=np.array(feats)
        features.append(feats)
        labels.append(0)      #1,0 for nomal
        pbar.update(1)

features=np.array(features)
print(features.shape)

with open('ill_features', 'wb') as fp:
    pickle.dump(features, fp)

labels=np.array(labels)
print(labels.shape)

with open('ill_labels', 'wb') as fp:
    pickle.dump(labels, fp)
