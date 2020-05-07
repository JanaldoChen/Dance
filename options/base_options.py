import os
import argparse

class BaseOptions(object):
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--data_root', type=str, default='data/Multi-Garment_dataset', help='path to dataset')
        self._parser.add_argument('--checkpoints_dir', type=str, default='outputs/checkpoints',
                                  help='models are saved here')
        self._parser.add_argument('--image_size', type=int, default=224, help='the image size for input')
        self._parser.add_argument('--num_frame', type=int, default=3, help='number of frames')
        self._parser.add_argument('--tex_size', type=int, default=3, help='texture size for renderer')
        self._parser.add_argument('--deformed', type=float, default=0.1, help='range [-deformed, deformed] for mesh deformation')
        self._parser.add_argument('--gen_tex', action='store_true', help='if need to generate texture')
        self._parser.add_argument('--isHres', action='store_true', help='if use unpooling smpl')
        self._parser.add_argument('--hmr_state_path', type=str, default='outputs/checkpoints/hmr_tf2pt.pth', help='HMR pretrained state path')
        self._parser.add_argument('--smpl_path', type=str, default='assets/smpl_model.pkl', help='SMPL model path')
        self._parser.add_argument('--adj_mat_path', type=str, default='assets/adj_mat_info.pkl', help='adjacency matrix path')
        self._parser.add_argument('--adj_mat_hres_path', type=str, default='assets/adj_mat_hres_info.pkl', help='adjacency matrix hres path')

    def parse(self):
        if not self._initialized:
            self.initialize()
        opt = self._parser.parse_args()
        opt.is_train = self.is_train
        self.print_options(opt)
        self.opt = opt
        return self.opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self._parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)