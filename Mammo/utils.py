import os
import math
import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def find_th(pred, n):
    pred = pd.Series(pred)
    pred = pred.sort_values(0, ascending=False)
    pred[:n] = 1; pred[n:] = 0
    pred = pred.sort_index().values
    return pred