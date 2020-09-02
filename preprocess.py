import numpy as np
import pandas as po

def standardize_dataframe(df, train_fracn=0.6):
    """
    Standardize each column separately and return the mean and std of each col so that the transform can be reversed. 
    To ensure there is no data leakage and at the same time ensure that all sets are scaled similarly, all thre sets are scaled using the mean and std of the train set.
    """

    df_train    = df.iloc[:int(train_fracn*len(df))]

    mean_std = {col: {'mean': 0, 'std': 0} for col in df_train.columns}
    for col in df.columns:
        mean    = df_train[col].mean() 
        std     = df_train[col].std()

        df[col] = (df[col] - mean)/std

        mean_std[col]['mean'] = mean
        mean_std[col]['std'] = std

    return df, mean_std

def train_val_test_split(df, train_fracn=0.6, val_fracn=0.2):
    """
    Split dataset into train, val and test sets
    """

    df_train     = df.iloc[:int(train_fracn*len(df)/96)*96]
    df_val       = df.iloc[int(train_fracn*len(df)/96)*96:int((train_fracn+val_fracn)*len(df)/96)*96]
    df_test      = df.iloc[int((train_fracn+val_fracn)*len(df)/96)*96:]

    return df_train, df_val, df_test

def make_windows(data, train_seq_len = 4*24*7, val_seq_len = 4*24):
    X = np.zeros((len(data) - val_seq_len + 1 - train_seq_len, train_seq_len, len(data.columns)))
    y = np.zeros((len(data) - val_seq_len + 1 - train_seq_len, val_seq_len, 1))

    for i in tqdm(range(train_seq_len, len(data) - val_seq_len + 1)):        
        X[i-train_seq_len, :, :] = data.values[i - train_seq_len: i]
        y[i-train_seq_len, :, :] = data['consumption'].values[i: i + val_seq_len].reshape(-1, 1)
        
    return X, y