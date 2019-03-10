import os
import re
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras import utils
import matplotlib.pyplot as plt

def dataset():
    return tf.keras.utils.get_file(fname='imdb.tar.gz', origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', extract=True)

def directory_data(directory):
    data={}
    data['review'] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path),'r') as file:
            data['review'].append(file.read())
    return pd.DataFrame.from_dict(data)

def data(directory):
    pos_df = directory_data(os.path.join(directory, 'pos'))
    neg_df = directory_data(os.path.join(directory, 'neg'))
    pos_df['sentiment'] = 1
    neg_df['sentiment'] = 0

    return pd.concat([pos_df, neg_df])

def plot(train_df):
    reviews = list(train_df['review'])
    tokenized_reviews = [r.split() for r in reviews]
    review_len_by_token = [len(t) for t in tokenized_reviews]
    review_len_by_eumjeol = [len(s.replace(' ','')) for s in reviews]

    plt.figure(figsize=(12,5))
    plt.hist(review_len_by_token, bins=50, alpha=0.5, color='r',label='word')
    plt.hist(review_len_by_eumjeol, bins=50, alpha=0.5, color='b', label='alphabet')
    plt.yscale('log', nonposy='clip')
    plt.title('review length histogram')
    plt.xlabel('review length')
    plt.ylabel('# of reviews')
    plt.show()
    print('done')

if __name__ == '__main__':
    #download()
    #directory_data()
    dataset = '/home/geekslife/.keras/datasets/'
    train_df = data(os.path.join(os.path.dirname(dataset), 'aclImdb', 'train'))
    test_df = data(os.path.join(os.path.dirname(dataset), 'aclImdb', 'test'))
    
    plot(train_df)
