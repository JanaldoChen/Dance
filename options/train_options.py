from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        
        self._parser.add_argument('--train_file', type=str, default='train.txt', help='train ids file')
        self._parser.add_argument('--val_file', type=str, default='val.txt', help='val ids file')
        self._parser.add_argument('--G_lr', type=float, default=1e-5, help='learning rate for generator')
        self._parser.add_argument('--D_lr', type=float, default=1e-5, help='learning rate for discriminator')
        self._parser.add_argument('--G_adam_b1', type=float, default=0.5, help='adam beta1 for generator')
        self._parser.add_argument('--G_adam_b2', type=float, default=0.999, help='adam beta2 for generator')
        self._parser.add_argument('--D_adam_b1', type=float, default=0.5, help='adam beta1 for discriminator')
        self._parser.add_argument('--D_adam_b2', type=float, default=0.999, help='adam beta2 for discriminator')
        self._parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
        self._parser.add_argument('--gan_mode', type=str, default='ls', help='gan mode [ls | original | w | hinge]')
        
        self._parser.add_argument('--use_loss_gan', action='store_true', help='if use gan loss')
        self._parser.add_argument('--use_loss_img_masked', action='store_true', help='if use img_masked loss')
        self._parser.add_argument('--use_loss_mask', action='store_true', help='if use mask loss')
        self._parser.add_argument('--use_loss_mask_personal', action='store_true', help='if use mask_personal loss')
        self._parser.add_argument('--use_loss_shape', action='store_true', help='if use shape loss')
        self._parser.add_argument('--use_loss_pose', action='store_true', help='if use pose loss')
        self._parser.add_argument('--use_loss_verts', action='store_true', help='if use verts loss')
        self._parser.add_argument('--use_loss_verts_personal', action='store_true', help='if use verts_personal loss')
        self._parser.add_argument('--use_loss_v_personal', action='store_true', help='if use v_personal loss')
        
        self._parser.add_argument('--lambda_gan', type=float, default=1, help='lambda gan loss')
        self._parser.add_argument('--lambda_shape', type=float, default=0.1, help='lambda shape loss')
        self._parser.add_argument('--lambda_pose', type=float, default=1, help='lambda pose loss')
        self._parser.add_argument('--lambda_verts', type=float, default=100, help='lambda verts loss')
        self._parser.add_argument('--lambda_verts_personal', type=float, default=100, help='lambda verts_personal loss')
        self._parser.add_argument('--lambda_v_personal', type=float, default=100, help='lambda v_personal loss')
        self._parser.add_argument('--lambda_img_masked', type=float, default=10, help='lambda image masked loss')
        self._parser.add_argument('--lambda_mask', type=float, default=0.1, help='lambda mask loss')
        self._parser.add_argument('--lambda_mask_personal', type=float, default=0.1, help='lambda mask_personal loss')
        
        self._parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoing')
        self._parser.add_argument('--start_epoch', type=int, default=0, help='the start epoch for training')
        self._parser.add_argument('--epochs', type=int, default=20, help='the epochs for training')
        self._parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        self._parser.add_argument('--print_freq', type=int, default=2, help='print train information per print_freq')
        self._parser.add_argument('--log_dir', type=str, default='outputs/logs', help='the logs for visualize')
        self._parser.add_argument('--hmr_no_grad', action='store_true', help='if hmr need grad')
        self._parser.add_argument('--step_size', type=int, default=2, help='multiply by a gamma every step_size epochs')
        self._parser.add_argument('--lr_gamma', type=float, default=0.5, help='lr decay')
        self._parser.add_argument('--pose_cam_path', type=str, default='assets/pose_cam.pkl', help='the pose cam set path')
        
        self.is_train = True