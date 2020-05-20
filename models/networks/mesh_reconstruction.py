import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr

from .render import orthographic_proj_withz_idrot
from .hmr import HumanModelRecovery
from .graph_projection import GraphProjection
from .mesh_deformation import MeshDeformation

class MeshReconstruction(nn.Module):
    
    def __init__(self, num_frame, deformed=0.1, deformed_iterations=3, adj_mat_pkl_path='assets/adj_mat_info.pkl'):
        super(MeshReconstruction, self).__init__()
        
        self.num_frame = num_frame
        self.deformed = deformed
        self.deformed_iterations = deformed_iterations
        self.proj_func = orthographic_proj_withz_idrot
        
        # Human Mesh Recovery
        self.hmr = HumanModelRecovery()
        
        # Perceptual features pooling
        self.pool = GraphProjection()
        
        # Mesh Deformation
        self.mesh_deformation = MeshDeformation(feat_dim=768, hid_dim=256, out_dim=3, deformed=self.deformed, adj_mat_pkl_path=adj_mat_pkl_path)
        
            
    def forward(self, imgs, smpl):
        
        bs = int(imgs.shape[0] / self.num_frame)
        
        shapes, poses, cams, imgs_feats = self.get_shape_pose_cam(imgs)
        
        shapes = shapes.view(bs, self.num_frame, shapes.shape[-1])
        shape = shapes.mean(1)
        shapes = shape.unsqueeze(1).repeat(1, self.num_frame, 1)
        shapes = shapes.view(-1, shapes.shape[-1])
        # Get smpl 3D mesh
        verts = smpl(shapes, poses)
        verts_personal = verts
        
        for _ in range(self.deformed_iterations):
            # Perceptual features pooling
            proj_verts = self.project_to_image(verts_personal, cams, flip=False, withz=False)
            verts_feats = self.pool(imgs_feats, proj_verts)
        
            verts_feats = verts_feats.view(bs, self.num_frame, verts_feats.shape[-2], verts_feats.shape[-1])
            verts_feats = verts_feats.mean(1)
        
            v_personal = self.mesh_deformation(verts_feats)
            v_personals = v_personal.unsqueeze(1).repeat(1, self.num_frame, 1, 1).view(-1, v_personal.shape[-2], v_personal.shape[-1])
            verts_personal = smpl(shapes, poses, v_personals)
            
        
        outputs = {
            'shape': shape,
            'poses': poses,
            'cams': cams,
            'v_personal': v_personal,
            'verts': verts,
            'verts_personal': verts_personal
        }
        
        return outputs
    
    def get_shape_pose_cam(self, imgs, get_feats=True):
        # Human mesh recovery from images
        theta, imgs_feats = self.hmr(imgs)
        cam = theta[:, 0:3].contiguous()
        pose = theta[:, 3:75].contiguous()
        shape = theta[:, 75:].contiguous()
        if get_feats:
            return shape, pose, cam, imgs_feats
        return shape, pose, cam
    
    def get_pose(self, imgs):
        with torch.no_grad():
            theta, _ = self.hmr(imgs, get_feats=False)
        pose = theta[:, 3:75].contiguous()
        return pose
    
    def project_to_image(self, vertices, cam, offset_z=0., flip=False, withz=False):
        proj_verts = self.proj_func(vertices, cam, offset_z)
        
        # if flipping the y-axis here to make it align with the image coordinate system!
        if flip:
            proj_verts[:, :, 1] *= -1
        
        # if preserving the z
        if not withz:
            proj_verts = proj_verts[:, :, 0:2]

        return proj_verts
