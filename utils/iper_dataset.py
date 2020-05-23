import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.util import load_obj, load_pickle_file

class iPER_Dataset(Dataset):
    def __init__(self, imgs_path, pose_shape_pkl_path, image_size=256):
        imgs_fn_list = os.listdir(imgs_path)
        imgs_fn_list.sort()
        self.imgs_path_list = [os.path.join(imgs_path, img_fn) for img_fn in imgs_fn_list]
        self.pose_shape_pkl = load_pickle_file(pose_shape_pkl_path)
        self.Resize = transforms.Resize(image_size)
        self.ToTensor = transforms.ToTensor()
        
        if len(self.imgs_path_list) != self.pose_shape_pkl['pose'].shape[0]:
            print('images: ', len(self.imgs_path_list))
            print('smpls: ', self.pose_shape_pkl['pose'].shape[0])
        
        
    def __getitem__(self, index):
        img_path = self.imgs_path_list[index]
        img = self.Resize(Image.open(img_path).convert('RGB'))
        pose = self.pose_shape_pkl['pose'][index]
        shape = self.pose_shape_pkl['shape'][index]
        cam = self.pose_shape_pkl['cams'][index]
        
        output = {
            'image': self.ToTensor(img).float(),
            'pose': torch.from_numpy(pose).float(),
            'shape': torch.from_numpy(shape).float(),
            'cam': torch.from_numpy(cam).float()
        }
        return output
        
    def __len__(self):
        return len(self.imgs_path_list)