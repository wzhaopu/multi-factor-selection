import numpy as np
import pandas as pd
from tqdm import tqdm

def predict_on_batch(sess, predict_func, test_x, 
                     dropmode='Bypass', rep_times=3,
                     batchsize=5000, verbose=0):
    n_samples_test = test_x.shape[0]
    n_batch_test = n_samples_test//batchsize
    if dropmode!='Bayes':
        drop = True if dropmode=='Dropout' else False
        test_pred = np.zeros(n_samples_test)
        test_logit = np.zeros(n_samples_test)
        for i_batch in range(n_batch_test):
            batch_x = test_x.iloc[i_batch*batchsize:(i_batch+1)*batchsize]
            _pred = predict_func(sess, batch_x, drop=drop)
            if type(_pred) is list:
                _pred, _logit = _pred
                test_logit[i_batch*batchsize:(i_batch+1)*batchsize] = _logit.reshape(-1)
            test_pred[i_batch*batchsize:(i_batch+1)*batchsize] = _pred.reshape(-1)
        if n_batch_test*batchsize<n_samples_test:
            batch_x = test_x.iloc[n_batch_test*batchsize:]
            _pred = predict_func(sess, batch_x, drop=drop)
            if type(_pred) is list:
                _pred, _logit = _pred[0], _pred[1]
                test_logit[n_batch_test*batchsize:] = _logit.reshape(-1)
            test_pred[n_batch_test*batchsize:] = _pred.reshape(-1)
    else:
        test_preds = np.zeros((n_samples_test, rep_times))
        test_logit = np.zeros((n_samples_test, rep_times))
        for j in range(rep_times):
            test_preds[:,j], test_logit[:,j] = predict_on_batch(
                sess, predict_func, test_x, dropmode='Dropout', batchsize=batchsize)
        test_pred = test_preds.mean(1)
    return test_pred, test_logit
def train_on_batch(sess, train_func, 
                   train_x, train_y, 
                   batchsize=5000, 
                   lr=1e-3,
                   shuffle=False,
                   verbose=0):
    n_samples = train_x.shape[0]
    n_batch = n_samples//batchsize
    inds = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(inds)
    for i_batch in tqdm(range(n_batch), disable=verbose==0):
        batch_x = train_x.iloc[inds[i_batch*batchsize:(i_batch+1)*batchsize]]
        batch_y = train_y.iloc[inds[i_batch*batchsize:(i_batch+1)*batchsize]].values
        loss = train_func(sess, batch_x, batch_y, lr=lr)
    return

def neg_sample(X, y, neg_sample_rate=0.05, keep_order=False, which_batch=0):
    if neg_sample_rate>0.99:
        return X, y
    else:
        print("Negative sampling, neg_sample_rate is {}...".format(neg_sample_rate))
        idx_pos = np.array(y[y==1].index)
        idx_neg = np.array(y[y==0].index)
        np.random.shuffle(idx_neg)
        batch_len = int(neg_sample_rate*len(idx_neg))
        idx_neg = idx_neg[which_batch*batch_len:min(len(idx_neg),(which_batch+1)*batch_len)]
        idx = np.concatenate([idx_pos, idx_neg])
        if keep_order:
            idx = np.sort(idx)
        else:
            np.random.shuffle(idx)
        return X.loc[idx], y.loc[idx]