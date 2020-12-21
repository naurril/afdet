
import numpy as np
import os
import math
from config import *
from timeit import timeit
from multiprocessing import Pool
import pickle

from sustechscapes_dataset  import SustechScapesDataset

def cloud_to_pillars(points):
    #PILLAR_SIZE_X=0.6
    #PILLAR_SIZE_Y=0.6
    #MIN_POINTS_PER_PILLAR = 10

    #front_points  = d.crop_points_in_camera_view("kitti","000001", "front")
    #points = front_points

    # grid coordinates
    grid_coord_raw = points[:,0:2]/np.array([PILLAR_SIZE_X,PILLAR_SIZE_Y]) #.reshape([-1,2])
    grid_coord = np.round(grid_coord_raw)  # np.round, the grid_coord are in the center of the pillar.

    # attach grid coord to all points
    grid_data = np.concatenate([points,grid_coord], axis=-1)
    #grid_data

    def extract_feature(points, idx):
        idx = np.reshape(idx, (1,2))
        idx_pos = idx*np.array([PILLAR_SIZE_X,PILLAR_SIZE_Y])
        points_coord = points[:,0:3]
        dist2center = points_coord-points_coord.mean(axis=0)
        dist2pillarcenter = points[:,0:2] - idx_pos  # 4:6 are pillor coordinates, and the center of the pillar
        pt_features = np.concatenate([points[:,0:4], dist2center, dist2pillarcenter], axis=-1)

        return pt_features

    def resample_pillar_points(pt_features):
        #print(pt_features.shape[0])
        if pt_features.shape[0]>MIN_POINTS_PER_PILLAR:
            idx = np.arange(pt_features.shape[0])
            np.random.shuffle(idx)
            ret_features =  pt_features[idx[0:MIN_POINTS_PER_PILLAR]]
        else:
            padding = np.zeros([MIN_POINTS_PER_PILLAR-pt_features.shape[0], pt_features.shape[1]])
            ret_features = np.concatenate([pt_features, padding], axis=0)
        return ret_features

    def make_one_pillar(pts,idx):
        pts = extract_feature(pts, idx)
        return pts

    index = np.unique(grid_coord, axis=0)
    #index = kitti_pillar_coord_to_image_pos(index, PILLAR_SIZE_X, PILLAR_SIZE_Y, IMAGE_DIMENSION)
    index = index[valid_pillar_coord(index, PILLAR_SIZE_X, PILLAR_SIZE_Y, IMAGE_DIMENSION)]

    # df = pd.DataFrame(data=grid_data)
    # df_groupby = df.groupby([4,5], as_index=False)
    # df_features = df_groupby.apply(make_one_pillar)
    pillars = []
    for idx in index:
        keep = ((grid_data[:,4] == idx[0]) & (grid_data[:,5]==idx[1]))
        group = grid_data[keep][:,:4]
        pillar = make_one_pillar(group, idx)
        pillars.append(pillar)

    #points_number = list(map(lambda p:p.shape[0], pillars))
    #print(len(points_number))
    #sort_indices = np.argsort(points_number)

    #if len(points_number) > MAX_PILLAR_NUM:
    #    sort_indices = sort_indices[(len(points_number) - MAX_PILLAR_NUM): len(points_number)]

    #pillars = np.array(pillars)[sort_indices]
    #index = np.array(index)[sort_indices]
    
    # translate index should be after pillar feature extracting
    index = kitti_pillar_coord_to_image_pos(index)

    pillars = map(resample_pillar_points, pillars)
    pillars = np.stack(pillars)
    return pillars,index


# for kitti, the front camera view covers x \in [0,80], y \in [-40,40] area
# x goes forward in kitti lidar coordinate system

def kitti_pillar_coord_to_image_pos(coord):
    return (coord + np.array(IAMGE_COORD_OFFSET)).astype(np.int)

def valid_pillar_coord(pillar_coord):
    img_pos = kitti_pillar_coord_to_image_pos(pillar_coord)
    return (img_pos[:,0] >= 0) &  (img_pos[:,1] >= 0) & (img_pos[:,0] < IMAGE_DIMENSION[0]) & (img_pos[:,1] < IMAGE_DIMENSION[1])

def build_pillar_image(d,f,save_path):    
    "dataset, frame, savepath"
    print(f)
    front_points  = d.crop_points_in_camera_view("kitti",f, "front")
    pillars,coord = cloud_to_pillars(front_points)

        
    with open(os.path.join(save_path, "pillars", f), "wb") as f:
        pickle.dump((pillars, coord), f)
    #img is too large, so we save indices and pillars only
    #img = pillars_to_image(pillars, coord, PILLAR_SIZE_X, PILLAR_SIZE_Y, IMAGE_DIMENSION)        
    #img.tofile(os.path.join(save_path, "pillars", f))



kitti_cls_num = 3
kitti_cls_index_map={
        "Car":0,
        "Pedestrian":1,
        "Cyclist":2,
}


def build_gt(labels):
    H,W = IMAGE_DIMENSION

    if labels is None:
        return np.zeros([H,W,1+kitti_cls_num+2])

   
    # input_pointcloud = tf.keras.Input(shape=(H, W, P, D)) #
    
    # input_obj_ind = tf.keras.Input(dtype=tf.bool, shape=(H, W, 1)) #
    
    box_positions = labels[:, 0:2]
    box_pillar_coord = np.round(box_positions[:, 0:2]/np.array([PILLAR_SIZE_X, PILLAR_SIZE_Y]))
    box_img_coord = kitti_pillar_coord_to_image_pos(box_pillar_coord)


    # obj img index
    ##
    ## heatmap, [H, W, Cls] in paper, but let's try [H,W,1]
    obj_ind = np.zeros((H,W,1))
    obj_heatmap = np.zeros((H,W,kitti_cls_num))
    obj_z = np.zeros((H,W,1))
    obj_size =  np.zeros((H,W,3))
    obj_angle = np.zeros((H,W,2))
    
   

    for idx in range (box_img_coord.shape[0]):
        
        x,y = box_img_coord[idx,:] # this is int

        if x >=0 and x < Ｈ and y >=0 and y < W:
            obj_ind[x,y]=1.0
            obj_z[x,y] = labels[idx, 2]
            obj_size[x,y] =  labels[idx, 3:6] # shall we use unit PILLAR_SIZE_X/PILLAR_SIZE_Y
            obj_angle[x,y] = [math.cos(labels[idx, 8]), math.sin(labels[idx, 8])]

        channel_ind = int(labels[idx, 9])
        # note: in centernet, the \sigma is adaptive according to actual object size
        # todo:
        for i in range(x-OBJECT_OFFSET_RADIUS, x+OBJECT_OFFSET_RADIUS):
            for j in range(y-OBJECT_OFFSET_RADIUS, y+OBJECT_OFFSET_RADIUS):
                if i >=0 and i < Ｈ and j >=0 and j < W:
                    obj_heatmap[i,j, channel_ind] = max(obj_heatmap[i,j, channel_ind], math.exp(-((x-i)*(x-i) + (y-j)*(y-j))/2.0))  # gaussian kernel

    ## offset
    ## we need to set offset for all pixels around the object position
    ## with radius r
    offset = box_pillar_coord * np.array([PILLAR_SIZE_X, PILLAR_SIZE_Y]) - box_positions  # use pillar coord rather than image coord
    offset /= PILLAR_SIZE_X  # offset is in units of the pillar size
    obj_offset = np.zeros((H, W, 2))

    for idx in range (box_img_coord.shape[0]):
        x,y = box_img_coord[idx,:] # this is int
        if x >=0 and x < Ｈ and y >=0 and y < W:
            obj_offset[x,y] = offset[idx,:]
        else:
            print(x,y, "out of range")
        
        # the neighbors may still  be in range
        for i in range (2*OBJECT_OFFSET_RADIUS+1):
            for j in range(2*OBJECT_OFFSET_RADIUS+1):
                px = x+i-OBJECT_OFFSET_RADIUS
                py = y+j-OBJECT_OFFSET_RADIUS

                if px >=0 and px < Ｈ and py >=0 and py < W:
                    obj_offset[px, py] = offset[idx,:] + np.array([px-x, py-y])
        
    # z coord

    # grid_coord_raw = points[:,0:2]/np.array([PILLAR_SIZE_X,PILLAR_SIZE_Y]).reshape([-1,2])
    # grid_coord = np.round(grid_coord_raw)

    # #ground truth
    # input_heatmap = tf.keras.Input(shape=(H, W, num_classes))
    # input_offset =  tf.keras.Input(shape=(H, W, 2))
    # input_z_value =  tf.keras.Input(shape=(H, W, num_classes*1))
    # input_dim =  tf.keras.Input(shape=(H, W, num_classes*3))
    # input_orientation = tf.keras.Input(shape=(H, W, num_classes*8))

    return np.concatenate([obj_ind, obj_heatmap, obj_offset, obj_z, obj_size, obj_angle], axis=-1)
    #return np.concatenate([obj_heatmap, obj_offset], axis=-1)



def build_gt_file(d,f,save_path):
    print(f)
    # save it to file
    #np.save(save_path+"/"+f, img)
    label = d.load_label("kitti", f)
    #print("lable num", len(label), list(map(lambda l: l["obj_type"], label)))

    label_nparray = d.label_to_nparray(label, kitti_cls_index_map)

    
    #labels = label_nparray.reshape((-1,10))  # last ele is typeindex
    gt = build_gt(label_nparray)
    #print(gt.shape, np.mean(gt,axis=(0,1)), np.sum(gt[:,:,0]))

    # save img,heatmap,offset, for later training and testing
    #print(label)
    
    gt.tofile(os.path.join(save_path, "gt", f))
    return gt


def prepare_raw_data(func):
    sustechscapes_root_dir = "/home/lie/code/SUSTechPOINTS/data"
    save_path = "./data/kitti-afdet"

    d = SustechScapesDataset(sustechscapes_root_dir, ["kitti"])
    scene = d.get_scene("kitti")
    
    for f in scene["frames"]:
        func(d,f,save_path)

if __name__ == "__main__":
    
    prepare_raw_data(build_pillar_image)
    prepare_raw_data(build_gt_file)

    #test()


