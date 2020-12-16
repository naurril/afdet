import tensorflow as tf
import numpy as np
import os
from config import *


data_path = "./data/kitti"

def get_dataset():

    frames = os.listdir(os.path.join(data_path,"pillars"))
    frames.sort() # sort to avoid randomness
    
    train_len = int(len(frames)*70/100)
    
    train_data  = tf.data.Dataset.from_tensor_slices(list(range(0, train_len)))
    eval_data  = tf.data.Dataset.from_tensor_slices(list(range(train_len, len(frames))))
    
    train_data = train_data.shuffle(buffer_size=8000)

    def load_frame(idx):
        f = frames[idx]
        pillar_image = np.fromfile(os.path.join(data_path, "pillars", f))
        pillar_image = np.reshape(pillar_image, [H,W,10,9])
        pillar_image = pillar_image[:,:,:,3:]
        #print(np.mean(pillar_image), np.max(pillar_image), np.min(pillar_image))
        gt = np.fromfile(os.path.join(data_path, "gt", f))
        gt = np.reshape(gt, [H,W,CLASS_NUM+9]) #ind, heatmap, offset, z, size, angle

        return pillar_image, gt

    def tf_load_frame(idx):
        [pillar_image, gt] = tf.py_function(load_frame, [idx], [tf.float32, tf.float32])
        
        pillar_image.set_shape([H,W,P,D-3])
        gt.set_shape([H,W,CLASS_NUM+9])

        return pillar_image, gt

    train_data = train_data.map(tf_load_frame)
    eval_data = eval_data.map(tf_load_frame)

    return train_data, eval_data


if __name__ == "__main__":
    train,eval = get_dataset()

