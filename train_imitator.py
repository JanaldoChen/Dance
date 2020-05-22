import os
import time
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from utils.people_snapshot_dataset import People_Snapshot_Dataset
from options.train_options import TrainOptions
from models.imitator import Imitator

def main():
    opt = TrainOptions().parse()

    opt.data_root = 'data/people_snapshot_public'
    opt.checkpoints_dir = 'outputs/checkpoints/imitator'
    opt.log_dir = 'outputs/logs/imitator'
    opt.use_loss_gan = True
    opt.use_loss_rec = True
    opt.use_loss_mask = True
    opt.use_loss_verts = False
    opt.lambda_gan = 1
    opt.lambda_rec = 10
    opt.lambda_mask = 0.1
    opt.lambda_verts = 10
    opt.isHres = False
    opt.batch_size = 4
    opt.epochs = 20
    opt.step_size = 2
    opt.lr_gamma = 0.5

    train_dataset = People_Snapshot_Dataset(data_root=opt.data_root, image_size=opt.image_size, get_uv=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    
    model = Imitator(opt)
    
    for fn in os.listdir(opt.log_dir):
        os.remove(os.path.join(opt.log_dir, fn))
        
    if opt.resume:
        model.load_checkpoint(opt.start_epoch)
    
    model.train()
    
    total = len(train_loader.dataset)
    for epoch in range(opt.start_epoch, opt.epochs):
        train_num = 0
        for i, data in enumerate(train_loader):
            
            model.set_input(data)
            model.optimize_parameters()
            
            N = data['shape'].size(0)
            train_num += N
            
            if (i + 1) % opt.print_freq == 0 or train_num == total:
                loss_vis = model.get_loss_vis()
                mess = "Epoch %d: [%d / %d]"%(epoch, train_num, total)
                for name in loss_vis:
                    mess += " | " + "%s: %.7f"%(name, loss_vis[name])
                print(mess)

                if (i + 1) % (opt.print_freq * 10) == 0 or train_num == total:
                    imgs_vis = model.visualize()

                    img_src_gt = imgs_vis['img_src_gt'].detach().cpu()
                    img_src_gt = make_grid(img_src_gt, nrow=img_src_gt.size(0), padding=0)

                    img_ref_gt = imgs_vis['img_ref_gt'].detach().cpu()
                    img_ref_gt = make_grid(img_ref_gt, nrow=img_ref_gt.size(0), padding=0)

                    img_masked_src = imgs_vis['img_masked_src'].detach().cpu()
                    img_masked_src = make_grid(img_masked_src, nrow=img_masked_src.size(0), padding=0)

                    img_masked_ref = imgs_vis['img_masked_ref'].detach().cpu()
                    img_masked_ref = make_grid(img_masked_ref, nrow=img_masked_ref.size(0), padding=0)

                    bg = imgs_vis['background'].detach().cpu()
                    bg = make_grid(bg, nrow=bg.size(0), padding=0)

                    img_src_rec = imgs_vis['img_src_rec'].detach().cpu()
                    img_src_rec = make_grid(img_src_rec, nrow=img_src_rec.size(0), padding=0)

                    img_ref_rec = imgs_vis['img_ref_rec'].detach().cpu()
                    img_ref_rec = make_grid(img_ref_rec, nrow=img_ref_rec.size(0), padding=0)
                    
                    save_image([img_src_gt, img_ref_gt, img_masked_src, img_masked_ref, bg, img_src_rec, img_ref_rec], os.path.join(opt.log_dir, "epoch_{:0>2d}_iter_{:0>5d}.png".format(epoch, i)), nrow=1, padding=0)
        
        model.save_checkpoint(epoch)
        model.update_learning_rate()
    
if __name__ == '__main__':
    main()
