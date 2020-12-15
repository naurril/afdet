import tensorflow as tf
import model as M
import dataset
import datetime
import numpy as np
import os

model_file = "afnet.h5"
weights_file = "afnet_weights.h5"

from config import *


tf.random.set_seed(0)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()


model = M.get_model(1, [H, W, P, D], True)
model.load_weights("epoch_0_afnet_weights.h5")

datapath = "./data/kitti"
frame = "000035"

pillars = np.fromfile(os.path.join(datapath, "pillars", frame))
gt = np.fromfile(os.path.join(datapath, "gt", frame))

pillars = pillars.reshape((H,W,P,D))
pillars = pillars[np.newaxis, :,:,:,3:]
print(pillars.shape)
gt = gt.reshape((H,W,-1))
gt = gt[:,:,[1]]

heatmap_pred = model.predict(pillars)

print(heatmap_pred, gt)