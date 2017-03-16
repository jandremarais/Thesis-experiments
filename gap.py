import os
import sys
sys.path.append('youtube-8m')
#os.chdir('/home/maraisjandre9/youtube-8m')
import eval_util

import numpy as np
#labels_val = np.genfromtxt ('y.csv', delimiter=",")
#predictions_val = np.genfromtxt ('yhat.csv', delimiter=",")

import feather
labels_val = feather.read_dataframe('y.feather').as_matrix()
predictions_val = feather.read_dataframe('yhat.feather').as_matrix()

value = eval_util.calculate_gap(predictions_val, labels_val)
