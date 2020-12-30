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

labelpath= "/home/lie/fast/code/SUSTechPoints-be/data/afdet-pred/label"
# datapath = "./data/2020-06-28-14-21-56-afdet"

#labelpath= "./data/kitti/output"
datapath = "./data/kitti/kitti-afdet"
calib_path = "./data/kitti/training/calib"

# P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
# P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
# P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
# P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
# R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
# Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
# Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01



def read_kitti_extrinsic_matrix(frame):
    with open(os.path.join(calib_path, frame+".txt")) as f:
        lines = f.readlines()
        trans = [x for x in filter(lambda s: s.startswith("Tr_velo_to_cam"), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.split(" ")[1:])]
        matrix = matrix + [0,0,0,1]
        m = np.array(matrix)
        velo_to_cam  = m.reshape([4,4])


        trans = [x for x in filter(lambda s: s.startswith("R0_rect"), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.split(" ")[1:])]        
        m = np.array(matrix).reshape(3,3)
        
        m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)
        
        rect = np.concatenate((m, np.expand_dims(np.array([0,0,0,1]), 0)), axis=0)        
        m = np.matmul(rect, velo_to_cam)
        #m = np.linalg.inv(m)
        return m



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
    #pred = pred.astype(np.float64)

    # gt = np.fromfile(os.path.join(datapath, "gt", f))
    # gt = np.reshape(gt, [PILLAR_IMAGE_HEIGHT,PILLAR_IMAGE_WIDTH,CLASS_NUM+9]) #ind, heatmap, offset, z, size, angle
    # pred = np.expand_dims(gt[:,:,1:],0)  #gt contains indicator, where pred doesn't
        
    pred = pred.astype(np.float64)
    heatmap = pred[:,:,:,            0:CLASS_NUM]
    offset =  pred[:,:,:,    CLASS_NUM:(CLASS_NUM+2)]
    z =       pred[:,:,:,(CLASS_NUM+2):(CLASS_NUM+3)]
    size =       pred[:,:,:,(CLASS_NUM+3):(CLASS_NUM+6)]
    angle =       pred[:,:,:,(CLASS_NUM+6):(CLASS_NUM+8)]
    


    
    heatmap_highlights = tf.nn.max_pool2d(heatmap, (3,3), (1,1), "SAME")
    heatmap_highlights = tf.math.reduce_max(heatmap_highlights, axis=-1, keepdims=True)    
    #heatmap_highlights = heatmap_highlights.numpy()

    ## float32 is not json serializable
    keep = tf.math.equal(heatmap_highlights, heatmap)
    keep = keep.numpy()

    #confidential = heatmap[keep].reshape(keep.shape)



    boxes = []
    shape = keep.shape
    for n in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                for c in range(shape[3]):
                    if keep[n,h,w,c]:
                        prob = heatmap[n, h, w, c]
                        if (prob > 0.5):
                            #print(n,h,w,c,PILLAR_SIZE_X,PILLAR_SIZE_Y, offset[n,h,w])
                            box = [prob, c, (h - offset[n,h,w,0]-IAMGE_COORD_OFFSET[0])*PILLAR_SIZE_X, (w-offset[n,h,w,1]-IAMGE_COORD_OFFSET[1])*PILLAR_SIZE_Y, z[n,h,w,0],
                                    size[n,h,w,0], size[n,h,w,1], size[n,h,w,2],
                                    math.atan2(angle[n,h,w,1],angle[n,h,w,0]),
                                ]
                            #print(box)
                            boxes.append(box)

    #print(boxes)


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

    if True:
        fpath=f"{labelpath}/{frame}.json"
        print("writing", fpath)
        with open(fpath, "w") as fout:
            json.dump(labels, fout)
    else:
        trans = read_kitti_extrinsic_matrix(frame)


        fpath=f"{labelpath}/{frame}.txt"
        print("writing", fpath)
        with open(fpath, "w") as fout:
            for l in labels:
                p = l["psr"]["position"]
                pos = np.matmul(trans, np.array([p["x"], p["y"], p["z"], 1]))
                #The reference point for the 3D bounding box for each object is centered on the bottom face of the box.
                fout.write(f'{l["obj_type"]} 0 0 0 0 0 0 0 {l["psr"]["scale"]["z"]} {l["psr"]["scale"]["y"]} {l["psr"]["scale"]["x"]} {pos[0]} {pos[1] + l["psr"]["scale"]["z"]/2} {pos[2]} {-l["psr"]["rotation"]["z"]-math.pi/2} {l["probability"]}\n')


# ----------------------------------------------------------------------------
#    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                      'Misc' or 'DontCare'
#    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                      truncated refers to the object leaving image boundaries
#    1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                      0 = fully visible, 1 = partly occluded
#                      2 = largely occluded, 3 = unknown
#    1    alpha        Observation angle of object, ranging [-pi..pi]
#    4    bbox         2D bounding box of object in the image (0-based index):
#                      contains left, top, right, bottom pixel coordinates
#    3    dimensions   3D object dimensions: height, width, length (in meters)
#    3    location     3D object location x,y,z in camera coordinates (in meters)
#    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1    score        Only for results: Float, indicating confidence in
#                      detection, needed for p/r curves, higher is better.

frames = os.listdir(os.path.join(datapath, "pillars"))
frames.sort()
for f in frames:
    predict_one_frame(f)



# {"x": 24.17015399652417, "y": -0.6803653287203262, "z": -0.7849088723448446}
# 0 48 79 0 0.5 0.5 [-0.34030799  0.36073066]
# [1.0, 0, 23.82984600347583, -0.3196346712796725, -0.7849088723448446, 3.2, 1.66, 1.61, -0.09079632679489658]