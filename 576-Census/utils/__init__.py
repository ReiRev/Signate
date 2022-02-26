import pandas as pd
import time

def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    dfs = [pd.read_feather(f'features/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train

def save_submission(y_pred, description ,score=0.):
    pd.read_csv('./data/input/sample_submit.csv', names=['id', 'Y'])
    sample_submit['Y'] = y_pred
    sample_submit.to_csv('./data/output/{}-{}-{}.csv'.format(int(time.time()), description, score), header=False, index=False)