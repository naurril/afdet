import tensorflow as tf
import numpy as np
import os
from config import *
import pickle

data_path = "./data/kitti-afdet"

def get_dataset():

    frames = os.listdir(os.path.join(data_path,"pillars"))
    frames.sort() # sort to avoid randomness
    
    train_len = int(len(frames)*70/100)
    
    train_data  = tf.data.Dataset.from_tensor_slices(list(range(0, train_len)))
    eval_data  = tf.data.Dataset.from_tensor_slices(list(range(train_len, len(frames))))
    
    train_data = train_data.shuffle(buffer_size=8000)

    def load_frame(idx):
        f = frames[idx]
        
        # pillar_image = np.fromfile(os.path.join(data_path, "pillars", f))
        # pillar_image = np.reshape(pillar_image, [H,W,10,9])
        # pillar_image = pillar_image[:,:,:,3:]

        with open(os.path.join(data_path,"pillars",f),"rb") as fin:
            pillars, coord = pickle.load(fin)  #1600,10,9; 1600,2
            
            pillars = pillars[:,:,3:]
            pillar_image = np.zeros([PILLAR_IMAGE_HEIGHT,PILLAR_IMAGE_WIDTH,10,6],dtype=np.float64)

            coord = coord.astype(np.int64)
            for i in range(coord.shape[0]):
                x,y = coord[i]
                if x>=0 and x < PILLAR_IMAGE_HEIGHT and y>=0 and y<PILLAR_IMAGE_WIDTH:
                    pillar_image[coord[i,0],coord[i,1]] = pillars[i]
                else:
                    print("out of range", x, y)


        #print(np.mean(pillar_image), np.max(pillar_image), np.min(pillar_image))
        gt = np.fromfile(os.path.join(data_path, "gt", f))
        gt = np.reshape(gt, [PILLAR_IMAGE_HEIGHT,PILLAR_IMAGE_WIDTH,CLASS_NUM+9]) #ind, heatmap, offset, z, size, angle

        return pillar_image, gt

    def tf_load_frame(idx):
        [pillars, gt] = tf.py_function(load_frame, [idx], [tf.float32,tf.float32])
        
        pillars.set_shape([PILLAR_IMAGE_HEIGHT,PILLAR_IMAGE_WIDTH,10,6])
        #coord.set_shape([1600,2])
        gt.set_shape([PILLAR_IMAGE_HEIGHT,PILLAR_IMAGE_WIDTH,CLASS_NUM+9])

        return pillars, gt

    train_data = train_data.map(tf_load_frame)
    eval_data = eval_data.map(tf_load_frame)

    return train_data, eval_data


if __name__ == "__main__":
    train,eval = get_dataset()

