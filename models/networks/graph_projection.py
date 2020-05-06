import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphProjection(nn.Module):
    """
    Graph Projection layer, which pool 2D features to mesh
    The layer projects a vertex of the mesh to the 2D image and use
    bi-linear interpolation to get the corresponging feature.
    """

    def __init__(self):
        super(GraphProjection, self).__init__()

    def forward(self, img_feats, points):
        feats = []
        for img_feat in img_feats:
            feats.append(self.project(img_feat, points))
        output = torch.cat(feats, 2)
        return output

    def project(self, img_feat, sample_points):
        """
        :param img_feat: [batch_size, channel, h, w]
        :param smaple_points: [batch_size, num_points, 2], in range [-1, 1]
        :return: [batch_size, num_points, channel]
        """

        output = F.grid_sample(img_feat, sample_points.unsqueeze(1))
        output = torch.transpose(output.squeeze(2), 1, 2)

        return output
