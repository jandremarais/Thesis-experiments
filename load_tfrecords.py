import numpy as np
import tensorflow as tf
import string

rng = list(string.ascii_lowercase)
filenames = ["gs://youtube8m-ml/1/video_level/train/trainb{}.tfrecord".format(i) for i in rng]

vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for filename in filenames:
    for example in tf.python_io.tf_record_iterator(filename):
        tf_example = tf.train.Example.FromString(example)
        vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
        labels.append(tf_example.features.feature['labels'].int64_list.value)
        mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
        mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
    
from numpy import array
mean_audio = array(mean_audio)
mean_rgb = array(mean_rgb)

X = np.hstack((mean_rgb, mean_audio))

from sklearn import preprocessing
lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(labels)
