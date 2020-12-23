import pandas as pd
import numpy as np


def iris_data(file):
    iris = pd.DataFrame(data=pd.read_csv(file , engine='python'))
    iris = iris.iloc[:]

    iris.columns = ['c1', 'c2', 'c3', 'c4', 'label']
    iris = iris[iris['label'] != 'Iris-setosa']
    iris.loc[iris['label'] == 'Iris-versicolor', 'label'] = 1
    iris.loc[iris['label'] == 'Iris-virginica', 'label'] = -1

    return iris


def hbt_data(file):

    hbt = pd.DataFrame(data=pd.read_csv(file, delimiter=r"\s+", engine='python'))
    # hbt = hbt.iloc[:, [0, 2]] //if we want first and third columns
    hbt.columns = ['c1', 'label', 'c2']
    #values of 1, -1
# reorder columns so that label is the third column
    hbt = hbt[hbt.columns[[0,2,1]]]

    # change label 2 --> -1
    hbt.loc[hbt['label'] == 2, 'label'] = -1
    return hbt
