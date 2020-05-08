import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr

from .smpl import SMPL
from .render import SMPLRenderer
from .hmr import HumanModelRecovery
from .graph_projection import GraphProjection
from .mesh_deformation import MeshDeformation

class MeshReconstruction(nn.Module):
    
    def __init__(self, image_size, tex_size=3, gen_tex=True, deformed=0.1, isHres=False, smpl_pkl_path='assets/smpl_model.pkl', adj_mat_pkl_path='assets/adj_mat_info.pkl'):
        super(MeshReconstruction, self).__init__()
        
        self.gen_tex = gen_tex
        self.deformed = deformed
        
        # Human Mesh Recovery
        self.hmr = HumanModelRecovery()
        
        # SMPL
        self.smpl = SMPL(pkl_path=smpl_pkl_path, isHres=isHres)
        
        # Perceptual features pooling
        self.pool = GraphProjection()
        
        # Mesh Deformation
        self.mesh_deformation = MeshDeformation(feat_dim=771, hid_dim=256, out_dim=3, deformed=self.deformed, adj_mat_pkl_path=adj_mat_pkl_path)
        
        # Neural Render
        if isHres:
            self.smpl_render = SMPLRenderer(faces=self.smpl.faces_hres, image_size=image_size, tex_size=tex_size)
        else:
            self.smpl_render = SMPLRenderer(faces=self.smpl.faces, image_size=image_size, tex_size=tex_size)
            
    def forward(self, imgs):
        
        bs, num_frame = imgs.shape[:2]
        imgs = imgs.view(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
        
        shapes, poses, cams, imgs_feats = self.get_shape_pose_cam(imgs)
        
        shapes = shapes.view(bs, num_frame, shapes.shape[-1])
        shape = shapes.mean(1)
        shapes = shape.unsqueeze(1).repeat(1, num_frame, 1)
        shapes = shapes.view(-1, shapes.shape[-1])
        # Get smpl 3D mesh
        verts = self.get_verts(shapes, poses)
            
        # Perceptual features pooling
        proj_verts = self.project_to_image(verts, cams, flip=False, withz=False)
        verts_feats = self.pool(imgs_feats, proj_verts)
        
        verts_feats = verts_feats.view(bs, num_frame, verts_feats.shape[-2], verts_feats.shape[-1])
        verts_feats = verts_feats.mean(1)
        
        if self.deformed > 0:
            v_personal = self.mesh_deformation(verts_feats)
            v_personals = v_personal.unsqueeze(1).repeat(1, num_frame, 1, 1).view(-1, v_personal.shape[-2], v_personal.shape[-1])
            verts_personal = self.get_verts(shapes, poses, v_personals)
            
        
        outputs = {
            'shape': shape,
            'poses': poses,
            'cams': cams,
            'v_personal': v_personal,
            'verts': verts,
            'verts_personal': verts_personal
        }
        
        return outputs
        
        #return shape, pose, cam, verts_personal, tex, img, img_sil
    
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
            theta, _ = self.hmr(imgs)
        pose = theta[:, 3:75].contiguous()
        return pose
    
    def get_verts(self, shape, pose, v_personal=None):
        verts = self.smpl(beta=shape, theta=pose, v_personal=v_personal)
        return verts
    
    def get_render(self, verts, tex):
        img_masked, mask = self.smpl_render(verts, tex)
        return img_masked, mask
    
    def project_to_image(self, vertices, cam, offset_z=0., flip=False, withz=False):
        proj_verts = self.smpl_render.project_to_image(vertices, cam, offset_z=offset_z, flip=flip, withz=withz)
        return proj_verts
