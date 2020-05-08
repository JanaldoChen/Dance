import os
import time
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from utils.multi_garment_dataset import Multi_Garment_Dataset
from utils.util import AverageMeter
from options.train_options import TrainOptions
from models.pix2mesh_model import Pix2Mesh

def main():
    opt = TrainOptions().parse()
    
    train_dataset = Multi_Garment_Dataset(data_root=opt.data_root, pose_cam_path=opt.pose_cam_path, num_frame=opt.num_frame)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    
    model = Pix2Mesh(opt)
    
    for fn in os.listdir(opt.log_dir):
        os.remove(os.path.join(opt.log_dir, fn))
        
    model.load_state(model.net_G.hmr, opt.hmr_state_path)
    if opt.hmr_no_grad:
        model.set_requires_grad(model.net_G.hmr, requires_grad=False)
        
    if opt.resume:
        model.load_checkpoint(opt.start_epoch)
    
    model.train()
    
    total = len(train_loader.dataset)
    for epoch in range(opt.start_epoch, opt.epochs):
        train_num = 0
        for i, data in enumerate(train_loader):
            
            model.set_input(data)
            model.optimize_parameters()
            
            N = model.batch_size
            train_num += N
            
            if (i + 1) % opt.print_freq == 0 or train_num == total:
                loss_vis = model.get_loss_vis()
                mess = "Epoch %d: [%d / %d]"%(epoch, train_num, total)
                for name in loss_vis:
                    mess += " | " + "loss_%s: %.7f"%(name, loss_vis[name])
                print(mess)
                c, h, w = model.imgs_masked_personal_gt.shape[-3:]
                img_masked_personal_gt = model.imgs_masked_personal_gt.detach().view(-1, c, h, w).cpu()
                img_masked_personal_gt = make_grid(img_masked_personal_gt, nrow=img_masked_personal_gt.size(0), padding=0)
                
                img_masked_gt = model.imgs_masked_gt.detach().view(-1, c, h, w).cpu()
                img_masked_gt = make_grid(img_masked_gt, nrow=img_masked_gt.size(0), padding=0)
                
                img_masked = model.imgs_masked.detach().view(-1, c, h, w).cpu()
                img_masked = make_grid(img_masked, nrow=img_masked.size(0), padding=0)
                
                img_masked_personal = model.imgs_masked_personal.detach().view(-1, c, h, w).cpu()
                img_masked_personal = make_grid(img_masked_personal.view(-1, c, h, w).cpu(), nrow=img_masked_personal.size(0), padding=0)
                
                save_image([img_masked_personal_gt ,img_masked_gt, img_masked, img_masked_personal], os.path.join(opt.log_dir, "epoch_{:0>2d}_iter_{:0>5d}.png".format(epoch, i)), nrow=1, padding=0)
        
        model.save_checkpoint(epoch)
    
if __name__ == '__main__':
    main()
