

import numpy as np
import pandas as pd
import os
import math
from config import *

from sustechscapes_dataset  import SustechScapesDataset

def cloud_to_pillars(points, dx, dy, min_pts):
    #dx=0.6
    #dy=0.6
    #min_pts = 10

    #front_points  = d.crop_points_in_camera_view("kitti","000001", "front")
    #points = front_points

    # grid coordinates
    grid_coord_raw = points[:,0:2]/np.array([dx,dy]).reshape([-1,2])
    grid_coord = np.round(grid_coord_raw)  # np.round, the grid_coord are in the center of the pillar.

    # attach grid coord to all points
    grid_data = np.concatenate([points,grid_coord], axis=-1)
    grid_data

    def do_sample(points, min_pts):
        points_coord = points[:,0:3]
        dist2center = points_coord-points_coord.mean(axis=0)
        dist2pillarcenter = points[:,0:2] - points[:, 4:6]  # 4:6 are pillor coordinates, and the center of the pillar
        #print(points.shape, dist2center.shape, dist2pillarcenter.shape)
        pt_features = np.concatenate([points[:,0:4], dist2center, dist2pillarcenter], axis=-1)

        #print(pt_features)

        if pt_features.shape[0]>min_pts:
            idx = np.arange(pt_features.shape[0])
            np.random.shuffle(idx)
            ret_features =  pt_features[idx[0:min_pts]]
            return ret_features
        else:
            padding = np.zeros([min_pts-pt_features.shape[0], pt_features.shape[1]])
            return  np.concatenate([pt_features, padding], axis=0)


    def make_one_pillar(pts):
        np_pts = pts.to_numpy()
        pts = do_sample(np_pts, min_pts)
        pts = pts
        return pts

    df = pd.DataFrame(data=grid_data)
    df_features = df.groupby([4,5], as_index=False).apply(make_one_pillar)
    
    pilar_coord = np.apply_along_axis(lambda x:list(x), 0, df_features.index.to_numpy())
        
    return np.stack(df_features), pilar_coord


# for kitti, the front camera view covers x \in [0,80], y \in [-40,40] area
# x goes forward in kitti lidar coordinate system


def kitti_pillar_coord_to_image_pos(coord, dx, dy, img_dim):
    return (coord + np.array([0., img_dim[1]/2])).astype(np.int)

def pillars_to_image(pillars ,pillar_coord, dx, dy, img_dim):
    #img_dim = np.ceil([50.*2/dx, 80./dy]).astype(np.int)
    image = np.zeros([img_dim[0], img_dim[1], pillars.shape[1], pillars.shape[2]])
    #print(image.shape)

    img_pos = kitti_pillar_coord_to_image_pos(pillar_coord, dx, dy, img_dim)
    
    for i in range(pillars.shape[0]):
        px,py = img_pos[i]
        if px < 0 or py < 0 or px >= img_dim[0] or py >= img_dim[1]:
            continue
        image[px,py] = pillars[i]
        
    return image


kitti_cls_num = 3
kitti_cls_index_map={
        "Car":0,
        "Pedestrian":1,
        "Cyclist":2,
}


def build_gt(labels, dx, dy, img_dim):
    H,W = img_dim

    if labels is None:
        return np.zeros([H,W,1+kitti_cls_num+2])

   
    # input_pointcloud = tf.keras.Input(shape=(H, W, P, D)) #
    
    # input_obj_ind = tf.keras.Input(dtype=tf.bool, shape=(H, W, 1)) #
    
    box_positions = labels[:, 0:2]
    box_pillar_coord = np.round(box_positions[:, 0:2]/np.array([dx, dy]))
    box_img_coord = kitti_pillar_coord_to_image_pos(box_pillar_coord, dx, dy, (H,W))


    # obj img index
    ##
    ## heatmap, [H, W, Cls] in paper, but let's try [H,W,1]
    obj_ind = np.zeros((H,W,1))
    obj_heatmap = np.zeros((H,W,kitti_cls_num))
    
    for idx in range (box_img_coord.shape[0]):
        
        x,y = box_img_coord[idx,:] # this is int

        if x >=0 and x < Ｈ and y >=0 and y < W:
            obj_ind[x,y]=1.0

        channel_ind = int(labels[idx, 9])
        # note: in centernet, the \sigma is adaptive according to actual object size
        # todo:
        for i in range(obj_heatmap.shape[0]):
            for j in range(obj_heatmap.shape[1]):
                obj_heatmap[i,j, channel_ind] = max(obj_heatmap[i,j, channel_ind], math.exp(-((x-i)*(x-i) + (y-j)*(y-j))/2.0))  # gaussian kernel
        # if x >=0 and x < Ｈ and y >=0 and y < W:
        #     obj_ind[x,y]=1.0
        #     obj_heatmap[x,y]=1.0
            
        #     non_center=0.8
        #     if y+1 < W: 
        #         obj_heatmap[x,y+1]=non_center
        #         obj_heatmap[x,y-1]=non_center

        #     if x+1 < H:
        #         if y+1 < W:
        #             obj_heatmap[x+1,y+1]=non_center
                
        #         obj_heatmap[x+1,y  ]=non_center

        #         if y-1 > 0:
        #             obj_heatmap[x+1,y-1]=non_center

        #     if x-1 > 0:
        #         obj_heatmap[x-1,y  ]=non_center

        #         if y+1 < W:
        #             obj_heatmap[x-1,y+1]=non_center
        #         if y-1 > 0:
        #             obj_heatmap[x-1,y-1]=non_center

        # else:
        #     print(x,y, "out of range")

    
    

    ## offset
    ## we need to set offset for all pixels around the object position
    ## with radius r
    offset = box_pillar_coord * np.array([dx, dy]) - box_positions  # use pillar coord rather than image coord
    offset /= dx  # offset is in units of the pillar size
    obj_offset = np.zeros((H, W, 2))

    for idx in range (box_img_coord.shape[0]):
        x,y = box_img_coord[idx,:] # this is int
        if x >=0 and x < Ｈ and y >=0 and y < W:
            obj_offset[x,y] = offset[idx,:]
        else:
            print(x,y, "out of range")
        
        # the neighbors may still  be in range
        for i in range (2*offset_radius+1):
            for j in range(2*offset_radius+1):
                px = x+i-offset_radius
                py = y+j-offset_radius

                if px >=0 and px < Ｈ and py >=0 and py < W:
                    obj_offset[px, py] = offset[idx,:] + np.array([px-x, py-y])
        

    # grid_coord_raw = points[:,0:2]/np.array([dx,dy]).reshape([-1,2])
    # grid_coord = np.round(grid_coord_raw)

    # #ground truth
    # input_heatmap = tf.keras.Input(shape=(H, W, num_classes))
    # input_offset =  tf.keras.Input(shape=(H, W, 2))
    # input_z_value =  tf.keras.Input(shape=(H, W, num_classes*1))
    # input_dim =  tf.keras.Input(shape=(H, W, num_classes*3))
    # input_orientation = tf.keras.Input(shape=(H, W, num_classes*8))

    return np.concatenate([obj_ind, obj_heatmap, obj_offset], axis=-1)
    #return np.concatenate([obj_heatmap, obj_offset], axis=-1)



def prepare_raw_data():
    sustechscapes_root_dir = "/home/lie/fast/code/SUSTechPoints-be/data"
    save_path = "./data/kitti"

    d = SustechScapesDataset(sustechscapes_root_dir, ["kitti"])
    scene = d.get_scene("kitti")
    
    def proc_one_frame(f):    
        front_points  = d.crop_points_in_camera_view("kitti",f, "front")    
    
        pillars,coord = cloud_to_pillars(front_points, dx, dy, min_pts)
        
        img = pillars_to_image(pillars, coord, dx, dy, img_dim)
        print(img.mean())
        # save it to file
        #np.save(save_path+"/"+f, img)
        label = d.load_label("kitti", f)
        print("lable num", len(label), list(map(lambda l: l["obj_type"], label)))

        label_nparray = d.label_to_nparray(label, kitti_cls_index_map)

        
        #labels = label_nparray.reshape((-1,10))  # last ele is typeindex
        gt = build_gt(label_nparray, dx, dy, img_dim)
        print(gt.shape, np.mean(gt,axis=(0,1)), np.sum(gt[:,:,0]))

        # save img,heatmap,offset, for later training and testing
        #print(label)
        img.tofile(os.path.join(save_path, "pillars", f))
        gt.tofile(os.path.join(save_path, "gt", f))
        return img, gt

    imgs = []
    gts = []
    for f in scene["frames"]:
        print(f)
        img,gt=proc_one_frame(f)
        imgs.append(img)
        gts.append(gt)
        
    return imgs, gts

def test():
    points = np.array([
        [-0.1,0.4,1,3],
        [0.1,0.5,2,4],
        [1,1,0,5],
        [0,-1,0,6],
        ])
    features, coord = cloud_to_pillars(points, 1, 1, 3)
    assert(features.shape[0]==3)
    print(features)
    print(coord)

    img = pillars_to_image(features, coord, 1, 1, [3,3])
    print(img)


if __name__ == "__main__":
    prepare_raw_data()

    #test()


