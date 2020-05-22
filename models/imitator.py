import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import itertools

from .networks.hmr import HumanModelRecovery
from .networks.smpl import SMPL
from .networks.render import SMPLRenderer
from .networks.inpainter import InpaintSANet
from .networks.generator import ResNetGenerator, ResUnetGenerator
from .networks.discriminator import PatchDiscriminator
from .networks.loss import GANLoss

from .base_model import BaseModel

class Imitator(BaseModel):
    def __init__(self, opt):
        super(Imitator, self).__init__(opt)

        if opt.isHres:
            self.opt.adj_mat_path = opt.adj_mat_hres_path
        # smpl
        self.smpl = SMPL(pkl_path=opt.smpl_path, isHres=opt.isHres).to(self.device)

        # Neural Render
        if opt.isHres:
            faces = self.smpl.faces_hres
        else:
            faces = self.smpl.faces
        self.smpl_render = SMPLRenderer(faces=faces, image_size=opt.image_size, tex_size=opt.tex_size).to(self.device)

        # Mesh Reconstruction
        self.hmr = HumanModelRecovery().to(self.device)
        self.model_names.append('hmr')

        # Inpainter
        self.bg_inpainter = InpaintSANet(c_dim=4)
        self.model_names.append('bg_inpainter')

        # RefineNet
        self.refine_net = ResUnetGenerator(c_dim=3).to(self.device)
        self.model_names.append('refine_net')

        if self.opt.use_loss_gan:
            # Discriminator
            self.net_D = PatchDiscriminator(input_nc=3).to(self.device)
            self.model_names.append('net_D')

        self.initialize()

        # Optimiziers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.hmr.parameters(), self.bg_inpainter.parameters(), self.refine_net.parameters()), lr=opt.G_lr, betas=(opt.G_adam_b1, opt.G_adam_b2))
        self.optimizer_names.append('optimizer_G')

        if opt.use_loss_gan:
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.D_lr, betas=(opt.D_adam_b1, opt.D_adam_b2))
            self.optimizer_names.append('optimizer_D')

        # Loss Functions
        self.loss_names.append('loss_G')
        if opt.use_loss_rec:
            self.criterion_rec = nn.MSELoss()
            self.loss_names.append('loss_rec')
        if opt.use_loss_mask:
            self.criterion_mask = nn.BCELoss()
            self.loss_names.append('loss_mask')
        if opt.use_loss_verts:
            self.criterion_verts = nn.MSELoss()
            self.loss_names.append('loss_verts')
        # if opt.use_loss_shape:
        #     self.criterion_shape = nn.MSELoss()
        #     self.loss_names.append('loss_shape')
        # if opt.use_loss_pose:
        #     self.criterion_pose = nn.MSELoss()
        #     self.loss_names.append('loss_pose')
        if opt.use_loss_gan:
            self.criterion_gan = GANLoss(gan_mode=opt.gan_mode, tensor=torch.cuda.FloatTensor)
            self.loss_names.append('loss_gan')
            self.loss_names.append('loss_D_real')
            self.loss_names.append('loss_D_fake')
            self.loss_names.append('loss_D')

        # Scheduler
        self.scheduler_G = lr_scheduler.StepLR(self.optimizer_G, step_size=opt.step_size, gamma=opt.lr_gamma)
        self.scheduler_names.append('scheduler_G')
        if opt.use_loss_gan:
            self.scheduler_D = lr_scheduler.StepLR(self.optimizer_D, step_size=opt.step_size, gamma=opt.lr_gamma)
            self.scheduler_names.append('scheduler_D')

    def initialize(self):
        BaseModel.initialize(self)
        self.load_state(self.hmr, self.opt.hmr_state_path)
        #self.set_requires_grad(self.hmr, requires_grad=False)
        self.load_state(self.bg_inpainter, self.opt.inpainter_state_path)
        #self.set_requires_grad(self.bg_inpainter, requires_grad=False)

    def set_input(self, input):
        with torch.no_grad():
            self.img_src_gt = input['image_src'].to(self.device)
            self.mask_src_gt = input['mask_src'].to(self.device)
            self.pose_src_gt = input['pose_src'].to(self.device)

            self.img_ref_gt = input['image_ref'].to(self.device)
            self.mask_ref_gt = input['mask_ref'].to(self.device)
            self.pose_ref_gt = input['pose_ref'].to(self.device)

            shape = input['shape'].to(self.device)
            self.pose_T = torch.zeros(self.pose_src_gt.size()).float().to(self.device)
            #self.v_personal = input['v_personal'].to(self.device)
            uv_img, f2vts = input['uv_image'].to(self.device), input['f2vts'].to(self.device)
            self.tex_gt = self.smpl_render.extract_tex(uv_img, self.smpl_render.points_to_sampler(f2vts))

            K, R, t, dist_coeffs, orig_size = input['camera_K'].to(self.device), input['camera_R'].to(self.device), input['camera_t'].to(self.device), input['camera_dist_coeffs'].to(self.device), input['camera_orig_size'].to(self.device)
            t_src = input['trans_src'].to(self.device)
            t_ref = input['trans_ref'].to(self.device)
            verts_src = self.smpl(shape, self.pose_src_gt)
            self.verts_src_gt = self.smpl_render.projection(verts_src, K, R, t_src, dist_coeffs, orig_size)
            verts_ref = self.smpl(shape, self.pose_ref_gt)
            self.verts_ref_gt = self.smpl_render.projection(verts_ref, K, R, t_ref, dist_coeffs, orig_size)

    def forward(self):
        # 3D Reconstruction
        src_info = self.hmr.get_details(self.hmr(self.img_src_gt, get_feats=False))
        self.shape_src = src_info['shape']
        self.pose_src = src_info['pose']
        self.cam_src = src_info['cam']

        ref_info = self.hmr.get_details(self.hmr(self.img_ref_gt, get_feats=False))
        self.shape_ref = ref_info['shape']
        self.pose_ref = ref_info['pose']
        self.cam_ref = ref_info['cam']

        # Motion transfer
        shape = (self.shape_src + self.shape_ref) / 2.0
        cam = (self.cam_src + self.cam_ref) / 2.0

        # verts_personal_src = self.smpl(shape, self.pose_src, self.v_personal)
        # verts_personal_src = self.smpl_render.project_to_image(self.verts_personal_src, self.cam_src, flip=True, withz=True)
        # self.img_masked_src, self.mask_src = self.smpl_render(verts_personal_src, self.tex_gt)

        # verts_personal_ref = self.smpl(shape, self.pose_ref, self.v_personal)
        # verts_personal_ref = self.mesh_rec.project_to_image(verts_personal_ref, self.cam, flip=True, withz=True)
        # self.img_masked_ref, self.mask_ref = self.smpl_render(verts_personal_ref, self.tex_gt)

        verts_src = self.smpl(shape, self.pose_src)
        self.verts_src = self.smpl_render.project_to_image(verts_src, self.cam_src, offset_z=5., flip=True, withz=True)
        self.img_masked_src, self.mask_src = self.smpl_render(self.verts_src, self.tex_gt)
        self.mask_src = self.mask_src.unsqueeze(1)

        verts_ref = self.smpl(shape, self.pose_ref)
        self.verts_ref = self.smpl_render.project_to_image(verts_ref, self.cam_ref, offset_z=5., flip=True, withz=True)
        self.img_masked_ref, self.mask_ref = self.smpl_render(self.verts_ref, self.tex_gt)
        self.mask_ref = self.mask_ref.unsqueeze(1)

        # Background inpainting
        self.bg = self.bg_inpainter(self.img_src_gt, self.mask_src_gt, only_out=True)

        # Refine
        self.img_src_rec = self.img_masked_src * self.mask_src + self.bg * (1 - self.mask_src)
        self.img_src_rec = self.refine_net(self.img_src_rec)

        self.img_ref_rec = self.img_masked_ref * self.mask_ref + self.bg * (1 - self.mask_ref)
        self.img_ref_rec = self.refine_net(self.img_ref_rec)

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
            D_fake_out_ref = self.net_D(self.img_ref_rec)
            D_fake_out_src = self.net_D(self.img_src_rec)
            self.loss_gan = self.criterion_gan(D_fake_out_ref, True) + self.criterion_gan(D_fake_out_src, True)
            self.loss_G += self.loss_gan

        if self.opt.use_loss_rec:
            self.loss_rec = self.criterion_rec(self.img_ref_rec, self.img_ref_gt) + self.criterion_rec(self.img_src_rec, self.img_src_gt)
            self.loss_G += self.opt.lambda_rec * self.loss_rec

        if self.opt.use_loss_mask:
            self.loss_mask = self.criterion_mask(self.mask_ref, self.mask_ref_gt) + self.criterion_mask(self.mask_src, self.mask_src_gt)
            self.loss_G += self.opt.lambda_mask * self.loss_mask

        # if self.opt.use_loss_shape:
        #     self.loss_shape = self.criterion_shape(self.shape, self.shape_gt)
        #     self.loss_G += self.opt.lambda_shape * self.loss_shape

        # if self.opt.use_loss_pose:
        #     self.loss_pose = self.criterion_pose(self.pose_ref, self.pose_ref_gt) + self.criterion_pose(self.pose_src, self.pose_src_gt)
        #     self.loss_G += self.opt.lambda_pose * self.loss_pose
        if self.opt.use_loss_verts:
            self.loss_verts = self.criterion_verts(self.verts_ref, self.verts_ref_gt) + self.criterion_verts(self.verts_src, self.verts_src_gt)
            self.loss_G += self.opt.lambda_verts * self.loss_verts

        self.loss_G.backward()
        self.optimizer_G.step()

    def _optimizer_D(self):
        self.set_requires_grad(self.net_D, True)
        self.optimizer_D.zero_grad()

        D_real_out = self.net_D(self.img_ref_gt)
        D_fake_out = self.net_D(self.img_ref_rec)
        self.loss_D_real = self.criterion_gan(D_real_out, True)
        self.loss_D_fake = self.criterion_gan(D_fake_out, False)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2

        self.loss_D.backward(retain_graph=True)
        self.optimizer_D.step()

    def visualize(self):
        imgs_vis = {
            'img_src_gt': self.img_src_gt,
            'img_ref_gt': self.img_ref_gt,
            'img_masked_src': self.img_masked_src,
            'img_masked_ref': self.img_masked_ref,
            'background': self.bg,
            'img_src_rec': self.img_src_rec,
            'img_ref_rec': self.img_ref_rec
        }
        return imgs_vis

    
