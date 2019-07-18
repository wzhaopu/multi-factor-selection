import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
def get_scores(y, pred, n_bins=100, bins=None, bins_name='Ad', name='',
               verbose=1):
    y = np.array(y)
    assert len(y)==len(pred)
    n = len(y)
    mse = mean_squared_error(y, pred)
    #loss = log_loss(y, pred)
    #auc = roc_auc_score(y, pred)
    ans =  {
            'MSE': np.round(mse, 8)}
    if bins is not None:
        pred_in_bins = np.array(pd.Series(bins).astype('category').cat.codes)
        df = pd.DataFrame(data={'b': pred_in_bins, 'p': pred, 'y': y, 'cnt': 1.})
        gp = df.groupby('b')
        sums = gp.sum()
        sums['err'] = np.abs(sums['p'] - sums['y'])
        ece = sums['err'].sum() / n
        smooth = lambda x: x + 0.01
        rce = np.sum(sums['err']/smooth(sums['y']/sums['cnt'])) / n
        ans['ECE-{}'.format(bins_name)] = np.round(ece, 8)
        ans['RCE-{}'.format(bins_name)] = np.round(rce, 8)
    if verbose:
        print(name,ans, sep='\t')
    return ans
