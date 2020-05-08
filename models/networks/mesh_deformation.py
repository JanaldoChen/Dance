import torch
import torch.nn as nn
import torch.nn.functional as F

from .gbottleneck import GBottleneck
from utils.util import load_pickle_file, torch_sparse_tensor

class MeshDeformation(nn.Module):
    
    def __init__(self, feat_dim, hid_dim, out_dim, deformed=0.1, adj_mat_pkl_path='assets/adj_mat_info.pkl'):
        super(MeshDeformation, self).__init__()
        self.deformed = deformed
        
        adj_mat_info = load_pickle_file(adj_mat_pkl_path)
        adj_mat = torch_sparse_tensor(adj_mat_info['indices'], adj_mat_info['value'], adj_mat_info['size'])
        
        self.deformed = GBottleneck(4, feat_dim, hid_dim, out_dim, adj_mat, activation='relu')
        
        self.to_verts = nn.Tanh()
        
        
    def forward(self, verts_feats):
        verts = self.deformed(verts_feats)
        verts = self.to_verts(verts) * self.deformed
        return verts
    
        
        