import datasets
import pandas as pd
import numpy as np

def load_dataset(threshold = 0.5, train_split='test', test_split='train', valid_split='validation'):
    ds = datasets.load_dataset('civil_comments')
    
    # train
    ds_ = ds['train'].shuffle(seed=0)
    small_train_ds = ds_.select(range(0,100000))
    large_train_ds = ds_.select(range(100000, len(ds_)))                                 
    
    # labels = [ 1 if i else 0 for i in np.array(ds[train_split]['toxicity']) > threshold ]
    labels = [ 1 if i else 0 for i in np.array(small_train_ds['toxicity']) > threshold ]
    train_ds = small_train_ds.add_column('label', labels)
    
    # test
    labels = [ 1 if i else 0 for i in np.array(large_train_ds['toxicity']) > threshold ] 
    test_ds = large_train_ds.add_column('label', labels)
    
    # validation
    labels = [ 1 if i else 0 for i in np.array(ds[valid_split]['toxicity']) > threshold ] 
    valid_ds = ds[valid_split].add_column('label', labels)
    valid_ds = valid_ds.select(range(0, 5000))
    
    return train_ds, valid_ds, test_ds