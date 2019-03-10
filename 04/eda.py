import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns
#%matplotlib inline
DATA_IN_PATH='./data_in/'
train_data = pd.read_csv(DATA_IN_PATH+'labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
print(train_data.head())