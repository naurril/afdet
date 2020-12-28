import tensorflow as tf
import model as M
import dataset
import datetime
import numpy as np
import os
import math
import json
import pickle


from config import *

model = M.get_model(CLASS_NUM, [MAX_PILLAR_NUM, PILLAR_IMAGE_HEIGHT, PILLAR_IMAGE_WIDTH, MIN_POINTS, POINT_FEATURE_LENGTH-2], True)
model.load_weights("models/afnet_weights.h5")


kitti_cls_name=[
        "Car",
        "Pedestrian",
        "Cyclist",
        ]

# labelpath= "/home/lie/fast/code/SUSTechPoints-be/data/2020-06-28-14-21-56/label"
# datapath = "./data/2020-06-28-14-21-56-afdet"

labelpath= "/home/lie/fast/code/SUSTechPoints-be/data/sustechscapes-mini-dataset-test/label"
datapath = "./data/sustechscapes-mini-dataset-afdet"


def predict_one_frame(frame):
    with open(os.path.join(datapath,"pillars",f),"rb") as fin:
        pillars, coord = pickle.load(fin)  #1600,10,9; 1600,2
        
        pillars = pillars[:,:,2:]
        pillars[:,:,0] = pillars[:,:,0] + 0.1
        pillars[:,:,1] = pillars[:,:,1]/200.0
        pillar_image = np.zeros([PILLAR_IMAGE_HEIGHT,PILLAR_IMAGE_WIDTH,MIN_POINTS_PER_PILLAR,POINT_FEATURE_LENGTH-2],dtype=np.float64)

        coord = coord.astype(np.int64)
        for i in range(coord.shape[0]):
            x,y = coord[i]
            if x>=0 and x < PILLAR_IMAGE_HEIGHT and y>=0 and y<PILLAR_IMAGE_WIDTH:
                pillar_image[coord[i,0],coord[i,1]] = pillars[i]
            else:
                print("out of range", x, y)
                
    
    pillar_image = np.expand_dims(pillar_image, 0)
    pred = model.predict(pillar_image)
    pred = pred.astype(np.float64)

    # gt = np.fromfile(os.path.join(datapath, "gt", f))
    # gt = np.reshape(gt, [PILLAR_IMAGE_HEIGHT,PILLAR_IMAGE_WIDTH,CLASS_NUM+9]) #ind, heatmap, offset, z, size, angle
    # pred = np.expand_dims(gt[:,:,1:],0)  #gt contains indicator, where pred doesn't
    
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
                        if (prob > 0.3):
                            print(n,h,w,c,PILLAR_SIZE_X,PILLAR_SIZE_Y, offset[n,h,w])
                            box = [prob, c, (h - offset[n,h,w,0]-shape[1]/2)*PILLAR_SIZE_X, (w-offset[n,h,w,1]-shape[2]/2)*PILLAR_SIZE_Y, z[n,h,w,0],
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
                "obj_id":box[0],
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

    fpath=f"{labelpath}/{frame}.json"
    print("writing", fpath)
    with open(fpath, "w") as fout:
        json.dump(labels, fout)


frames = os.listdir(os.path.join(datapath, "pillars"))
frames.sort()
for f in frames:
    predict_one_frame(f)



# {"x": 24.17015399652417, "y": -0.6803653287203262, "z": -0.7849088723448446}
# 0 48 79 0 0.5 0.5 [-0.34030799  0.36073066]
# [1.0, 0, 23.82984600347583, -0.3196346712796725, -0.7849088723448446, 3.2, 1.66, 1.61, -0.09079632679489658]