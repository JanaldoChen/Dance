import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.checkpoints_dir = opt.checkpoints_dir
        self.model_names = []
        self.loss_names = []
        self.optimizer_names = []
        self.scheduler_names = []
        self.metric = 0

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass
    
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass
                
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for name in self.scheduler_names:
            if isinstance(name, str):
                scheduler = getattr(self, 'scheduler_' + name)
                if self.opt.lr_policy == 'plateau':
                    scheduler.step(self.metric)
                else:
                    scheduler.step()
        
        for name in self.optimizer_names:
            if isinstance(name, str):
                optimizer = getattr(self, 'optimizer_' + name)
                lr = optimizer.param_groups[0]['lr']
                print(name + '_lr = %.7f' % lr)
    
    def print_networks(self, verbose=True):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth'%(epoch, name)
                load_path = os.path.join(self.checkpoints_dir, load_filename)
                net = getattr(self, model_name)
                self.load_state(net, load_path)
    
    def load_state(self, net, load_path):
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path)
        net.load_state_dict(state_dict, strict=False)
                
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.checkpoints_dir, save_filename)
                net = getattr(self, model_name)
                self.save_state(net, save_path)
                
    def save_state(self, net, save_path):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)
                
                    
    def get_loss_vis(self):
        loss_vis = {}
        for name in self.loss_names:
            if isinstance(name, str):
                loss = getattr(self, 'loss_' + name)
                loss_vis[name] = loss.item()
        return loss_vis