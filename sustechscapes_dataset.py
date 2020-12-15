import numpy as np
import os
import json

import math
import cv2
import pypcd.pypcd as pypcd
# scan data
# 


class SustechScapesDataset:
    def __init__(self, root_dir, spec_scenes=[]):
        self.root_dir = root_dir
        self.scenes = self._load_all_scenes(spec_scenes)
        #print("loaded scenes, ", list(map(lambda s:s["scene"], self.scenes)))

    def get_scene_list(self):
        return list(map(lambda s:s["scene"], self.scenes))

    def get_scene(self, scene):
        result = list(filter(lambda s: s["scene"]==scene, self.scenes))
        if result:
            return result[0]
        else:
            return None
    
    def get_radar_list(self, scene):
        return self.get_scene(scene)["radar"]
    def get_camera_list(self, scene):
        return self.get_scene(scene)["camera"]

    def get_image_dimension(self, scene, camera):        
        return self.get_scene(scene)["calib"]["camera"][camera]["image_dimension"]
        

    def _load_all_scenes(self, spec_scenes):
        if not spec_scenes:
            spec_scenes = self._read_all_scene_names()
        return list(map(self.read_one_scene, spec_scenes))

    def _read_all_scene_names(self):
        scenes = os.listdir(self.root_dir)
        scenes = filter(lambda s: not os.path.exists(os.path.join(self.root_dir, s, "disable")), scenes)
        scenes = list(scenes)
        scenes.sort()
        return scenes
    
    def load_label(self, scene, frame):
        sc = self.get_scene(scene)
        filepath = os.path.join(self.root_dir, scene, "label", frame+".json")
        with open(filepath) as f:
            label = json.load(f)
        
        return label
    
    
    def label_to_nparray(self, labels, class_index_map):
        labels = filter(lambda b: class_index_map.get(b["obj_type"]) is not None , labels)

        def box_to_nparray(box):
            psr = box["psr"]
            return np.array(
                [
                    psr["position"]["x"], psr["position"]["y"], psr["position"]["z"],
                    psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"],
                    psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"],
                    class_index_map[box["obj_type"]],
                ])

        datalist = list(map(box_to_nparray, labels))

        if len(datalist) > 0 :
            return np.stack(datalist, axis=0)
        else:
            return None

    def load_point_cloud(self, scene, frame):
        sc = self.get_scene(scene)
        filepath = os.path.join(self.root_dir, scene, "lidar", frame+sc["lidar_ext"])

        if sc["lidar_ext"] == ".bin":
            return np.fromfile(filepath, dtype=np.float32).reshape([-1,4])
        else:
            pc = pypcd.PointCloud.from_path(filepath)
            #raise Exception("file type '{}' not implemented yet".format(sc["lidar_ext"]))
            pts =  np.stack([pc.pc_data['x'], 
                             pc.pc_data['y'], 
                             pc.pc_data['z'], 
                             pc.pc_data['intensity']], axis=-1)
            return pts

    def _euler_angle_to_rotate_matrix(self, eu, t):
        theta = eu
        #Calculate rotation about x axis
        R_x = np.array([
            [1,       0,              0],
            [0,       math.cos(theta[0]),   -math.sin(theta[0])],
            [0,       math.sin(theta[0]),   math.cos(theta[0])]
        ])

        #Calculate rotation about y axis
        R_y = np.array([
            [math.cos(theta[1]),      0,      math.sin(theta[1])],
            [0,                       1,      0],
            [-math.sin(theta[1]),     0,      math.cos(theta[1])]
        ])

        #Calculate rotation about z axis
        R_z = np.array([
            [math.cos(theta[2]),    -math.sin(theta[2]),      0],
            [math.sin(theta[2]),    math.cos(theta[2]),       0],
            [0,               0,                  1]])

        R = np.matmul(R_x, np.matmul(R_y, R_z))

        t = t.reshape([-1,1])
        R = np.concatenate([R,t], axis=-1)
        R = np.concatenate([R, np.array([0,0,0,1]).reshape([1,-1])], axis=0)
        return R


    
        
    def crop_points_in_camera_view(self, scene, frame, camera):
        """
        get points in the view range of camera
        """
        sc = self.get_scene(scene)
        points = self.load_point_cloud(scene, frame) # xyzi
        #print(points.dtype, points.shape)
        calib = sc["calib"]["camera"][camera]

        extrinsic = np.array(calib["extrinsic"])
        intrinsic = np.array(calib["intrinsic"])
        extrinsic_matrix  = np.reshape(extrinsic, [4,4])
        # intrinsic 3*4, or 3*3 for kitti
        intrinsic_matrix  = np.reshape(intrinsic, [3,-1])

        #print(extrinsic_matrix)
        #print(intrinsic_matrix)

        #def proj_pts3d_to_img(pts_xyzi):

        pts = points[:,:3]  #xyz
        pts = np.concatenate([pts, np.ones([pts.shape[0],1])], axis=-1)
        imgpos = np.matmul(pts, np.transpose(extrinsic_matrix))

        # rect matrix shall be applied here, for kitti
        if calib.get("rect"):
            rect = np.array(calib["rect"]).reshape([-1,4])
            #print(rect)
            imgpos = np.matmul(imgpos, np.transpose(rect))


        imgpos3 = imgpos[:,:3]

        if intrinsic_matrix.shape[1] == 3:
            imgpos2 = np.matmul(imgpos3, np.transpose(intrinsic_matrix))
        else: # 4:
            imgpos2 = np.matmul(imgpos, np.transpose(intrinsic_matrix))

        imgfinal = imgpos2[:, 0:2]/imgpos2[:, 2:]

        dx,dy,dc = self.get_image_dimension(scene, camera)
        filter = (imgpos3[:,2] > 0) & (imgfinal[:,1] > 0) & (imgfinal[:,1]<dx) & (imgfinal[:,0]>0) & (imgfinal[:,0]<dy)
        
        # pts here is 4d but with last dim being 1.
        #print(filter.shape, pts.shape)
        ret = points[filter].astype(np.float32)
        #print(ret.shape)
        
        return ret

        #return proj_pts3d_to_img(points)


    def read_one_scene(self, s):
        scene = {
            "scene": s,  #scene name
            "frames": []
        }

        scene_dir = os.path.join(self.root_dir, s)

        frames = os.listdir(os.path.join(scene_dir, "lidar"))
        
        #print(s, frames)
        frames.sort()

        scene["lidar_ext"]="pcd"
        for f in frames:
            #if os.path.isfile("./data/"+s+"/lidar/"+f):
            filename, fileext = os.path.splitext(f)
            scene["frames"].append(filename)
            scene["lidar_ext"] = fileext

        point_transform_matrix=[]

        if os.path.isfile(os.path.join(scene_dir, "point_transform.txt")):
            with open(os.path.join(scene_dir, "point_transform.txt"))  as f:
                point_transform_matrix=f.read()
                point_transform_matrix = point_transform_matrix.split(",")

        def strip_str(x):
            return x.strip()

        calib = {}
        calib_camera={}
        calib_radar={}
        if os.path.exists(os.path.join(scene_dir, "calib")):
            if os.path.exists(os.path.join(scene_dir, "calib","camera")):
                calibs = os.listdir(os.path.join(scene_dir, "calib", "camera"))
                for c in calibs:
                    calib_file = os.path.join(scene_dir, "calib", "camera", c)
                    calib_name, _ = os.path.splitext(c)
                    if os.path.isfile(calib_file):
                        #print(calib_file)
                        with open(calib_file)  as f:
                            cal = json.load(f)
                            calib_camera[calib_name] = cal

        
            if os.path.exists(os.path.join(scene_dir, "calib", "radar")):
                calibs = os.listdir(os.path.join(scene_dir, "calib", "radar"))
                for c in calibs:
                    calib_file = os.path.join(scene_dir, "calib", "radar", c)
                    calib_name, _ = os.path.splitext(c)
                    if os.path.isfile(calib_file):
                        #print(calib_file)
                        with open(calib_file)  as f:
                            cal = json.load(f)
                            calib_radar[calib_name] = cal

        # camera names
        camera = []
        camera_ext = ""
        cam_path = os.path.join(scene_dir, "camera")
        if os.path.exists(cam_path):
            cams = os.listdir(cam_path)
            for c in cams:
                cam_file = os.path.join(scene_dir, "camera", c)
                if not os.path.isdir(cam_file):
                    continue
                
                camera.append(c)

                if camera_ext == "":
                    #detect camera file ext
                    files = os.listdir(cam_file)
                    if len(files)>=2:
                        _,camera_ext = os.path.splitext(files[0])
                        if calib_camera[c]:
                            tempimg = cv2.imread(os.path.join(scene_dir, "camera",c,files[0]))
                            calib_camera[c]["image_dimension"] = tempimg.shape

        if camera_ext == "":
            camera_ext = ".jpg"
        scene["camera_ext"] = camera_ext


        # radar names
        radar = []
        radar_ext = ""
        radar_path = os.path.join(scene_dir, "radar")
        if os.path.exists(radar_path):
            radars = os.listdir(radar_path)
            for r in radars:
                radar_file = os.path.join(scene_dir, "radar", r)
                if os.path.isdir(radar_file):
                    radar.append(r)
                    if radar_ext == "":
                        #detect camera file ext
                        files = os.listdir(radar_file)
                        if len(files)>=2:
                            _,radar_ext = os.path.splitext(files[0])

        if radar_ext == "":
            radar_ext = ".pcd"
        scene["radar_ext"] = radar_ext



        if not os.path.isdir(os.path.join(scene_dir, "bbox.xyz")):
            scene["boxtype"] = "psr"
            if point_transform_matrix:
                scene["point_transform_matrix"] = point_transform_matrix
            if camera:
                scene["camera"] = camera
            if radar:
                scene["radar"] = radar
            if calib_camera:
                calib["camera"] = calib_camera
            if calib_radar:
                calib["radar"] = calib_radar
        else:
            scene["boxtype"] = "xyz"
            if point_transform_matrix:
                scene["point_transform_matrix"] = point_transform_matrix
            if camera:
                scene["camera"] = camera
            if radar:
                scene["radar"] = radar
            if calib_camera:
                calib["camera"] = calib_camera
            if calib_radar:
                calib["radar"] = calib_radar

        scene["calib"] = calib
        return scene


if __name__ == "__main__":
    sustechscapes_root_dir = "/home/lie/fast/code/SUSTechPoints-be/data"

    d = SustechScapesDataset(sustechscapes_root_dir, ["kitti","sustechscapes-mini-dataset"])
    print(d.get_scene_list())
    print(d.get_radar_list("sustechscapes-mini-dataset"))
    print(d.get_camera_list("sustechscapes-mini-dataset"))

    #print(d.get_scene("kitti")["calib"]["camera"]["front"]["image_dimension"])
    #d.load_point_cloud("sustechscapes-mini-dataset","000000")
    #pts = d.load_point_cloud("kitti","000000")
    #

    for f in d.get_scene("kitti")['frames']:
        print(f)
        pts = d.crop_points_in_camera_view("kitti",f, "front")
        print(pts.shape)
        pts.tofile("./temp/lidar/{}.bin".format(f))