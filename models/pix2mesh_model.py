import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from .networks.mesh_reconstruction import MeshReconstruction
from .networks.discriminator import PatchDiscriminator
from .networks.loss import GANLoss

from .base_model import BaseModel

class Pix2Mesh(BaseModel):
    def __init__(self, opt):
        super(Pix2Mesh, self).__init__()
        
        self.opt = opt
        if opt.isHres:
            self.opt.adj_mat_path = opt.adj_mat_hres_path
        # Generator
        self.net_G = MeshReconstruction(image_size=opt.image_size, tex_size=opt.tex_size, deformed=opt.deformed, isHres=opt.isHres, hmr_state_path=opt.hmr_state_path, smpl_pkl_path=opt.smpl_path, adj_mat_pkl_path=opt.adj_mat_path, gen_tex=opt.gen_tex)
        self.model_names.append('G')
        
        if self.opt.use_loss_gan:
            # Discriminator
            self.net_D = PatchDiscriminator(input_nc=3)
            self.model_names.append('D')
            
        # Optimiziers
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opt.G_lr, betas=(opt.G_adam_b1, opt.G_adam_b2))
        self.optimizer_names.append('G')
        if self.opt.use_loss_gan:
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.D_lr, betas=(opt.D_adam_b1, opt.D_adam_b2))
            self.optimizer_names.append('D')
            
        # Loss Functions
        self.loss_names.append('G')
        if self.opt.use_loss_gan:
            self.criterion_gan = GANLoss(gan_mode=opt.gan_mode, tensor=torch.cuda.FloatTensor)
            self.loss_names.append('gan')
            self.loss_names.append('D_real')
            self.loss_names.append('D_fake')
            self.loss_names.append('D')
        if self.opt.use_loss_img_masked:
            self.criterion_img_masked = nn.MSELoss()
            self.loss_names.append('img_masked')
        if self.opt.use_loss_mask:
            self.criterion_mask = nn.BCELoss()
            self.loss_names.append('mask')
        if self.opt.use_loss_mask_personal:
            self.criterion_mask_personal = nn.BCELoss()
            self.loss_names.append('mask_personal')
        if self.opt.use_loss_shape:
            self.criterion_shape = nn.MSELoss()
            self.loss_names.append('shape')
        if self.opt.use_loss_pose:
            self.criterion_pose = nn.MSELoss()
            self.loss_names.append('pose')
        if self.opt.use_loss_verts:
            self.criterion_verts = nn.MSELoss()
            self.loss_names.append('verts')
        if self.opt.use_loss_verts_personal:
            self.criterion_verts_personal = nn.MSELoss()
            self.loss_names.append('verts_personal')
        if self.opt.use_loss_v_personal:
            self.criterion_v_personal = nn.MSELoss()
            self.loss_names.append('v_personal')
        
        # Scheduler
        self.scheduler_G = lr_scheduler.StepLR(self.optimizer_G, step_size=opt.step_size, gamma=opt.lr_gamma)
        self.scheduler_names.append('G')
        if self.opt.use_loss_gan:
            self.scheduler_D = lr_scheduler.StepLR(self.optimizer_D, step_size=opt.step_size, gamma=opt.lr_gamma)
            self.scheduler_names.append('D')
        
    def set_input(self, input):
        with torch.no_grad():
            
            self.shape_gt = input['shape'].cuda()
            self.poses_gt = input['poses'].cuda()
            self.v_personal_gt = input['v_personal'].cuda()
            self.cams_gt = input['cams'].cuda()

            uv_img, f2vts = input['uv_image'].cuda(), input['f2vts'].cuda()
            
            self.batch_size, self.num_frame = self.poses_gt.shape[:2]
            
            self.tex_gt = self.net_G.smpl_render.extract_tex(uv_img, self.net_G.smpl_render.points_to_sampler(f2vts))
            
            shape = self.shape_gt.unsqueeze(1).repeat(1, self.num_frame, 1).view(-1, self.shape_gt.shape[-1])
            pose = self.poses_gt.view(-1, self.poses_gt.shape[-1])
            v_personal = self.v_personal_gt.unsqueeze(1).repeat(1, self.num_frame, 1, 1).view(-1, self.v_personal_gt.shape[-2], self.v_personal_gt.shape[-1])
            cam = self.cams_gt.view(-1, self.cams_gt.shape[-1])
            tex = self.tex_gt.unsqueeze(1).repeat(1, self.num_frame, 1, 1, 1, 1, 1).view(-1, self.tex_gt.shape[-5], self.tex_gt.shape[-4], self.tex_gt.shape[-3], self.tex_gt.shape[-2], self.tex_gt.shape[-1])
            
            verts = self.net_G.smpl(shape, pose)
            verts = self.net_G.project_to_image(verts, cam, flip=True, withz=True)
            
            self.verts_gt = verts
            
            imgs_masked, masks = self.net_G.get_render(verts, tex)
            self.imgs_masked_gt = imgs_masked.view(self.batch_size, self.num_frame, imgs_masked.shape[-3], imgs_masked.shape[-2], imgs_masked.shape[-1])
            self.masks_gt = masks.view(self.batch_size, self.num_frame, masks.shape[-2], masks.shape[-1])
        
            verts_personal = self.net_G.smpl(shape, pose, v_personal)
            verts_personal = self.net_G.project_to_image(verts_personal, cam, flip=True, withz=True)
            
            self.verts_personal_gt = verts_personal
            
            imgs_masked_personal, masks_personal = self.net_G.get_render(verts_personal, tex)
            self.imgs_masked_personal_gt = imgs_masked_personal.view(self.batch_size, self.num_frame, imgs_masked_personal.shape[-3], imgs_masked_personal.shape[-2], imgs_masked_personal.shape[-1])
            self.masks_personal_gt = masks_personal.view(self.batch_size, self.num_frame, masks_personal.shape[-2], masks_personal.shape[-1])
        
        
    def forward(self):
        
        output = self.net_G(self.imgs_masked_gt)
            
        self.shape = output['shape']
        self.poses = output['poses']
        self.v_personal = output['v_personal']
        
        verts = output['verts']
        verts_personal = output['verts_personal']
        
        cam = output['cams']
        
        self.verts = self.net_G.project_to_image(verts, cam, flip=True, withz=True)
        self.verts_personal = self.net_G.project_to_image(verts_personal, cam, flip=True, withz=True)
        
        tex = self.tex_gt.unsqueeze(1).repeat(1, self.num_frame, 1, 1, 1, 1, 1).view(-1, self.tex_gt.shape[-5], self.tex_gt.shape[-4], self.tex_gt.shape[-3], self.tex_gt.shape[-2], self.tex_gt.shape[-1])
        
        imgs_masked, masks = self.net_G.get_render(self.verts, tex)
        self.imgs_masked = imgs_masked.view(self.batch_size, self.num_frame, imgs_masked.shape[-3], imgs_masked.shape[-2], imgs_masked.shape[-1])
        self.masks = masks.view(self.batch_size, self.num_frame, masks.shape[-2], masks.shape[-1])
        
        imgs_masked_personal, masks_personal = self.net_G.get_render(self.verts_personal, tex)
        self.imgs_masked_personal = imgs_masked_personal.view(self.batch_size, self.num_frame, imgs_masked_personal.shape[-3], imgs_masked_personal.shape[-2], imgs_masked_personal.shape[-1])
        self.masks_personal = masks_personal.view(self.batch_size, self.num_frame, masks_personal.shape[-2], masks_personal.shape[-1])
        
    def optimize_parameters(self):
        self.forward()
        
        if self.opt.use_loss_gan:
            #update D
            self._optimizer_D()
        
        #update G
        self._optimizer_G()
        
        
    def _optimizer_G(self):
        if self.opt.use_loss_gan:
            self.set_requires_grad(self.net_D, False)
            
        self.optimizer_G.zero_grad()
        
        self.loss_G = 0
        
        if self.opt.use_loss_gan:
            D_fake_out = self.net_D(self.imgs_masked)
            self.loss_gan = self.criterion_gan(D_fake_out, True)
            self.loss_G += self.loss_gan
        
        if self.opt.use_loss_img_masked:
            self.loss_img_masked = self.opt.lambda_img_masked * self.criterion_img_masked(self.imgs_masked, self.imgs_masked_tex_gt)
            self.loss_G += self.loss_img_masked
        
        if self.opt.use_loss_mask:
            self.loss_mask = self.opt.lambda_mask * self.criterion_mask(self.masks, self.masks_gt)
            self.loss_G += self.loss_mask
            
        if self.opt.use_loss_mask_personal:
            self.loss_mask_personal = self.opt.lambda_mask_personal * self.criterion_mask_personal(self.masks_personal, self.masks_personal_gt)
            self.loss_G += self.loss_mask_personal
        
        if self.opt.use_loss_shape:
            self.loss_shape = self.opt.lambda_shape * self.criterion_shape(self.shape, self.shape_gt)
            self.loss_G += self.loss_shape
            
        if self.opt.use_loss_pose:
            self.loss_pose = self.opt.lambda_pose * self.criterion_pose(self.poses, self.poses_gt)
            self.loss_G += self.loss_pose
            
        if self.opt.use_loss_v_personal:
            self.loss_v_personal = self.opt.lambda_v_personal * self.criterion_v_personal(self.v_personal, self.v_personal_gt)
            self.loss_G += self.loss_v_personal
        
        if self.opt.use_loss_verts:
            self.loss_verts = self.opt.lambda_verts * self.criterion_verts(self.verts, self.verts_gt)
            self.loss_G += self.loss_verts
            
        if self.opt.use_loss_verts_personal:
            self.loss_verts_personal = self.opt.lambda_verts_personal * self.criterion_verts_personal(self.verts_personal, self.verts_personal_gt)
            self.loss_G += self.loss_verts_personal
    
        self.loss_G.backward()
        self.optimizer_G.step()
        
    def _optimizer_D(self):
        self.set_requires_grad(self.net_D, True)
        self.optimizer_D.zero_grad()
        
        D_real_out = self.net_D(self.imgs_masked_gt)
        D_fake_out = self.net_D(self.imgs_masked)
        self.loss_D_real = self.criterion_gan(D_real_out, True)
        self.loss_D_fake = self.criterion_gan(D_fake_out, False)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2
        
        self.loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        
        
        
        