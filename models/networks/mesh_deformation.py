import torch
import torch.nn as nn
import torch.nn.functional as F

from .gconv import GConv
from .gbottleneck import GBottleneck
from utils.util import load_pickle_file, torch_sparse_tensor

class MeshDeformation(nn.Module):
    
    def __init__(self, feat_dim, hid_dim, out_dim, deformed=0.1, adj_mat_pkl_path='assets/adj_mat_info.pkl'):
        super(MeshDeformation, self).__init__()
        self.deformed = deformed
        
        adj_mat_info = load_pickle_file(adj_mat_pkl_path)
        adj_mat = torch_sparse_tensor(adj_mat_info['indices'], adj_mat_info['value'], adj_mat_info['size'])
        
        self.extract_feats = GBottleneck(1, feat_dim, hid_dim, hid_dim, adj_mat, activation='relu')
        
        self.to_verts = nn.Sequential(
            GBottleneck(3, hid_dim, hid_dim, out_dim, adj_mat, activation='relu'),
            nn.Tanh()
        )
        
        
    def forward(self, verts_feats):
        verts_hidden_feats = self.extract_feats(verts_feats)
        vertices_deformed = self.to_verts(verts_hidden_feats) * self.deformed
        return vertices_deformed
    
        
        