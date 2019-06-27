import numpy as np
import pandas as pd
import feather
import os
from training_utils import *
from scoring_func import get_scores
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from base_models import Model


# specify useful columns of features
num_words_dict = {
    'C1': 7,
    'banner_pos': 7,
    'site_id': 4737,
    'site_domain': 7745,
    'site_category': 26,
    'app_id': 8552,
    'app_domain': 559,
    'app_category': 36,
    'device_model': 8251,
    'device_type': 5,
    'device_conn_type': 4,
    'C14': 2626,
    'C15': 8,
    'C16': 9,
    'C17': 435,
    'C18': 4,
    'C19': 68,
    'C20': 172,
    'C21': 60,
    'day': 10,
    'hour': 24
}
feat_cols = list(num_words_dict.keys())
