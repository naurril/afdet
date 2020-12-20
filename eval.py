import tensorflow as tf
import model as M
import dataset
import datetime
import numpy as np
import os
import math
import json


model_file = "afnet.h5"
weights_file = "afnet_weights.h5"

from config import *

tf.random.set_seed(0)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()


model = M.get_model(CLASS_NUM, [H, W, P, D-3], True)
model.load_weights("models/epoch_2_afnet_weights.h5")
# model_file = "models/feature_no_point_coord/afnet.h5"
# model = tf.keras.models.load_model(model_file)
model.summary()

datapath = "./data/kitti"

kitti_cls_name=[
        "Car",
        "Pedestrian",
        "Cyclist",
        ]


def predict_one_frame(frame):
    pillars = np.fromfile(os.path.join(datapath, "pillars", frame))
    gt = np.fromfile(os.path.join(datapath, "gt", frame))

    pillars = pillars.reshape((H,W,P,D))
    pillars = pillars[np.newaxis, :,:,:,3:]
    print("pillars shape", pillars.shape)
    #gt = gt.reshape((1,H,W,-1))
    #print("gt shape", gt.shape)
    #gt = gt[:,:,[1]]

    pred = model.predict(pillars)
    pred = pred.astype(np.float64)

    
    heatmap = pred[:,:,:,            0:CLASS_NUM]
    offset =  pred[:,:,:,    CLASS_NUM:(CLASS_NUM+2)]
    z =       pred[:,:,:,(CLASS_NUM+2):(CLASS_NUM+3)]
    size =       pred[:,:,:,(CLASS_NUM+3):(CLASS_NUM+6)]
    angle =       pred[:,:,:,(CLASS_NUM+6):(CLASS_NUM+8)]

    heatmap_highlights = tf.nn.max_pool2d(heatmap, 3, 1, "SAME").numpy()

    keep = (heatmap_highlights == heatmap)

    #confidential = heatmap[keep].reshape(keep.shape)

    boxes = []
    shape = keep.shape
    for n in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                for c in range(shape[3]):
                    if keep[n,h,w,c]:
                        prob = heatmap[n, h, w, c]
                        if (prob > 0.6):
                            print(n,h,w,c,d0,d1, offset[n,h,w])
                            box = [prob, c, (h - offset[n,h,w,0])*d0, (w-offset[n,h,w,1]-shape[2]/2)*d1, z[n,h,w,0],
                                    size[n,h,w,0], size[n,h,w,1], size[n,h,w,2],
                                    math.atan2(angle[n,h,w,1],angle[n,h,w,0]),
                                ]
                            print(box)
                            boxes.append(box)

    print(boxes)


    def box_to_label(box):
            return {
                "probability": box[0],
                "obj_type": kitti_cls_name[int(box[1])],
                "obj_id":"",
                "psr":{
                    "position": {
                        "x": box[2],
                        "y": box[3],
                        "z": box[4],
                    },
                    "scale": {
                        "x": box[5],
                        "y": box[6],
                        "z": box[7],
                    },
                    "rotation": {
                        "x": 0,
                        "y": 0,
                        "z": box[8],
                    },
                }
            }
            

    labels = list(map(box_to_label, boxes))

    fpath="/home/lie/fast/code/SUSTechPoints-be/data/afdet-pred/label/{}.json".format(frame)
    print("writing", fpath)
    with open(fpath, "w") as fout:
        json.dump(labels, fout)


frames = os.listdir(os.path.join(datapath, "pillars"))
frames.sort()
for f in frames[6000:6100]:
    predict_one_frame(f)



# {"x": 24.17015399652417, "y": -0.6803653287203262, "z": -0.7849088723448446}
# 0 48 79 0 0.5 0.5 [-0.34030799  0.36073066]
# [1.0, 0, 23.82984600347583, -0.3196346712796725, -0.7849088723448446, 3.2, 1.66, 1.61, -0.09079632679489658]