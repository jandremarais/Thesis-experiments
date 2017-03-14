import os
import sys
sys.path.append('youtube-8m')
#os.chdir('/home/maraisjandre9/youtube-8m')
import eval_util

import numpy as np
labels_val = np.genfromtxt ('y.csv', delimiter=",")
predictions_val = np.genfromtxt ('yhat.csv', delimiter=",")

value = eval_util.calculate_gap(predictions_val, labels_val)
