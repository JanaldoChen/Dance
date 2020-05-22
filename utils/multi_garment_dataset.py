import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .util import load_pickle_file, load_obj, get_f2vts

class Multi_Garment_Dataset(Dataset):
    def __init__(self, data_root, pose_cam_path='assets/pose_cam.pkl', people_IDs_list=None, num_frame=1, isHres=True):
        self.data_root = data_root
        self.num_frame = num_frame
        self.people_IDs = people_IDs_list
        self.isHres = isHres
        pose_cam_pkl = load_pickle_file(pose_cam_path)
        self.poses = pose_cam_pkl['poses']
        self.cams = pose_cam_pkl['cams']
        
        if self.people_IDs is None:
            self.people_IDs = [people_ID for people_ID in os.listdir(data_root)]
        
        self.ToTensor = transforms.ToTensor()
        
    def __getitem__(self, index):
        people_ID = self.people_IDs[index]
        smpl_registered_pkl = load_pickle_file(os.path.join(self.data_root, people_ID, 'smpl_registered.pkl'))
        
        shape = torch.from_numpy(smpl_registered_pkl['betas']).float()
        
        #poses = torch.from_numpy(smpl_registered_pkl['pose']).float().unsqueeze(0)
        #cam = torch.from_numpy(smpl_registered_pkl['camera']).float()
        IDs = np.random.choice(self.poses.shape[0], self.num_frame)
        poses = torch.from_numpy(self.poses[IDs]).float()
        cams = torch.from_numpy(self.cams[IDs]).float()
        
        v_personal = torch.from_numpy(smpl_registered_pkl['v_personal']).float()
        if not self.isHres:
            v_personal = v_personal[:6890, :]
        
        uv_img = Image.open(os.path.join(self.data_root, people_ID, 'registered_tex.jpg')).convert('RGB')
        uv_img = self.ToTensor(uv_img)
        if self.isHres:
            vt, ft = load_pickle_file('assets/smpl_vt_ft_hres.pkl')
        else:
            vt, ft = load_pickle_file('assets/smpl_vt_ft.pkl')
        f2vts = torch.from_numpy(get_f2vts(vt, ft)).float()

        output = {
            'shape': shape,
            'poses': poses,
            'cams': cams,
            'v_personal': v_personal,
            'uv_image': uv_img,
            'f2vts': f2vts
        }
        return output
        
    def __len__(self):
        return len(self.people_IDs)
            
        