import os
import h5py
import numpy as np
from PIL import Image
from .util import load_pickle_file, get_f2vts_from_obj, get_camera
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class People_Snapshot_Dataset(Dataset):
    def __init__(self, data_root, interval=10, people_IDs=None, image_size=256, get_uv=False):

        self.data_root = data_root
        self.image_size = image_size
        self.interval = interval
        self.get_uv = get_uv
        self.people_IDs = people_IDs

        if self.people_IDs is None:
            self.people_IDs = os.listdir(self.data_root)

        self.num_peoples = len(self.people_IDs)
        self.dataset_size = 0
        self.data_info = {}

        for people_ID in self.people_IDs:
            images_path = os.listdir(os.path.join(data_root, people_ID, 'images'))
            images_path = [img_path for img_path in images_path if (img_path[-4:] == '.jpg' or img_path[-4:] == '.png')]
            images_path.sort()
            images_path = [os.path.join(self.data_root, people_ID, 'images', x) for x in images_path]

            people_info = {
                'people_ID': people_ID,
                'images_path': images_path,
                'length': len(images_path),
                'people_masks_hdf5': h5py.File(os.path.join(self.data_root, people_ID, 'masks.hdf5'), 'r'),
                'people_reconstructed_poses_hdf5': h5py.File(os.path.join(self.data_root, people_ID, 'reconstructed_poses.hdf5'), 'r')
            }
            self.data_info[people_ID] = people_info
            self.dataset_size += people_info['length'] // self.interval
        
        self.Resize = transforms.Resize(image_size)
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, index):
        people_ID = self.people_IDs[index % self.num_peoples]
        people_info = self.data_info[people_ID]
        pair_ids = np.random.choice(people_info['length'], size=2, replace=False)

        masks_hdf5 = people_info['people_masks_hdf5']
        reconstructed_poses_hdf5 = people_info['people_reconstructed_poses_hdf5']
        consensus_pkl = load_pickle_file(os.path.join(self.data_root, people_ID, 'consensus.pkl'))
        camera_pkl = load_pickle_file(os.path.join(self.data_root, people_ID, 'camera.pkl'))

        src_id = pair_ids[0]
        img_src = self.ToTensor(self.Resize(Image.open(people_info['images_path'][src_id]).convert('RGB'))).float()
        mask_src = self.ToTensor(self.Resize(Image.fromarray(masks_hdf5['masks'][src_id]))).float()
        pose_src = torch.from_numpy(reconstructed_poses_hdf5['pose'][src_id]).float()
        trans_src = torch.from_numpy(reconstructed_poses_hdf5['trans'][src_id]).float()

        ref_id = pair_ids[1]
        img_ref = self.ToTensor(self.Resize(Image.open(people_info['images_path'][ref_id]).convert('RGB'))).float()
        mask_ref = self.ToTensor(self.Resize(Image.fromarray(masks_hdf5['masks'][ref_id]))).float()
        pose_ref = torch.from_numpy(reconstructed_poses_hdf5['pose'][ref_id]).float()
        trans_ref = torch.from_numpy(reconstructed_poses_hdf5['trans'][ref_id]).float()

        shape = torch.from_numpy(consensus_pkl['betas']).float()
        v_personal = torch.from_numpy(consensus_pkl['v_personal']).float()
        camera = get_camera(camera_pkl)

        output = {
                'image_src': img_src,
                'mask_src': mask_src,
                'pose_src': pose_src,
                'trans_src': trans_src,
                'image_ref': img_ref,
                'mask_ref': mask_ref,
                'pose_ref': pose_ref,
                'trans_ref': trans_ref,
                'shape': shape,
                'v_personal': v_personal,
                'camera_K': torch.from_numpy(camera['K']).float(),
                'camera_R': torch.from_numpy(camera['R']).float(),
                'camera_t': torch.from_numpy(camera['t']).float(),
                'camera_dist_coeffs': torch.from_numpy(camera['dist_coeffs']).float(),
                'camera_orig_size': torch.Tensor([camera['orig_size']]).float()
            }

        if self.get_uv:
            uv_img = Image.open(os.path.join(self.data_root, people_ID, 'tex-{}.jpg'.format(people_ID))).convert('RGB')
            uv_img = self.ToTensor(uv_img)
            f2vts = get_f2vts_from_obj(os.path.join(self.data_root, people_ID, 'consensus.obj'))
            f2vts = torch.from_numpy(f2vts).float()
            output['uv_image'] = uv_img
            output['f2vts'] = f2vts
        return output

    def __len__(self):
        return self.dataset_size
        