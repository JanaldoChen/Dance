import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import neural_renderer as nr

def orthographic_proj_withz_idrot(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 3: [sc, tx, ty]
    No rotation!
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X
    proj = X

    proj_xy = scale * (proj[:, :, :2] + trans)
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)

class SMPLRenderer(nn.Module):
    def __init__(self, faces, image_size=256, tex_size=3, anti_aliasing=True, fill_back=False, background_color=(0, 0, 0), viewing_angle=30, near=0.1, far=25.0):
        super(SMPLRenderer, self).__init__()
        
        self.background_color = background_color
        self.anti_aliasing = anti_aliasing
        self.image_size = image_size
        self.tex_size = tex_size
        self.rasterizer_eps = 1e-3
        self.fill_back = fill_back
        
        #faces
        #faces = np.load(faces_path)
        #faces = torch.tensor(faces.astype(np.int32)).int()
        self.nf = faces.shape[0]
        self.register_buffer('faces', faces)
        self.register_buffer('coords', self.create_coords(self.tex_size))
        
        # project function and camera
        self.near = near
        self.far = far
        self.proj_func = orthographic_proj_withz_idrot
        self.viewing_angle = viewing_angle
        self.eye = [0, 0, -(1. / np.tan(np.radians(self.viewing_angle)) + 1)]
        
        # fill back
        if self.fill_back:
            faces = np.concatenate((faces, faces[:, ::-1]), axis=0)
            
        # light
        self.light_intensity_ambient = 1
        self.light_intensity_directional = 0
        self.light_color_ambient = [1, 1, 1]
        self.light_color_directional = [1, 1, 1]
        self.light_direction = [0, 1, 0]
    
    def set_ambient_light(self, int_dir=0.3, int_amb=0.7, direction=(1, 0.5, 1)):
        self.light_intensity_directional = int_dir
        self.light_intensity_ambient = int_amb
        if direction is not None:
            self.light_direction = direction
    
    def set_bg_color(self, background_color=(1, 1, 1)):
        self.background_color = background_color
        
    def forward(self, vertices, texture=None):
        if texture is None:
            texture = self.debug_textures().to(vertices.device)
        imgs = self.render(vertices, texture)
        img_sils = self.render_silhouettes(vertices)
        return imgs, img_sils
    
    def render(self, vertices, texture=None, faces=None):
        if faces is None:
            bs = vertices.shape[0]
            faces = self.faces.repeat(bs, 1, 1)
            
        if texture is None:
            texture = self.debug_textures().to(vertices.device)
            texture = texture.unsqueeze(0).repeat(bs, 1, 1, 1, 1, 1)
        # lighting is inplace operation
        texture = texture.clone()
        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        texture = nr.lighting(
            faces_lighting,
            texture,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)
        # set offset_z for persp proj
        #proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        #proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(vertices, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        image = nr.rasterize(faces, texture, self.image_size, self.anti_aliasing,
                              self.near, self.far, self.rasterizer_eps, self.background_color)
        
        return image
        
    def render_silhouettes(self, vertices, faces=None):
        if faces is None:
            bs = vertices.shape[0]
            faces = self.faces.repeat(bs, 1, 1)

        # set offset_z for persp proj
        #proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        #proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(vertices, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images
    
    def render_fim(self, vertices, faces=None):
        if faces is None:
            bs = vertices.shape[0]
            faces = self.faces.repeat(bs, 1, 1)
        
        # calculate the look_at vertices.
        vertices = nr.look_at(vertices, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim = nr.rasterize_face_index_map(faces, self.image_size, False)
        return fim
    
    def render_fim_wim(self, vertices, faces=None):
        if faces is None:
            bs = vertices.shape[0]
            faces = self.faces.repeat(bs, 1, 1)

        
        # calculate the look_at vertices.
        vertices = nr.look_at(vertices, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim, wim = nr.rasterize_face_index_map_and_weight_map(faces, self.image_size, False)
        return fim, wim
    
    def project_to_image(self, vertices, cam, offset_z=0., flip=False, withz=False):
        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam, offset_z)
        
        # if flipping the y-axis here to make it align with the image coordinate system!
        if flip:
            proj_verts[:, :, 1] *= -1
        
        # if preserving the z
        if not withz:
            proj_verts = proj_verts[:, :, 0:2]
            
        return proj_verts

    def projection(self, vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
        '''
        Calculate projective transformation of vertices given a projection matrix
        Input parameters:
        K: batch_size * 3 * 3 intrinsic camera matrix
        R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
        dist_coeffs: vector of distortion coefficients
        orig_size: original size of image captured by the camera
        Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
        pixels and z is the depth
        '''

        # instead of P*x we compute x'*P'
        vertices = torch.matmul(vertices, R.transpose(2,1)) + t.unsqueeze(1)
        x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)

        # Get distortion coefficients from vector
        k1 = dist_coeffs[:, None, 0]
        k2 = dist_coeffs[:, None, 1]
        p1 = dist_coeffs[:, None, 2]
        p2 = dist_coeffs[:, None, 3]
        k3 = dist_coeffs[:, None, 4]

        # we use x_ for x' and x__ for x'' etc.
        r = torch.sqrt(x_ ** 2 + y_ ** 2)
        x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
        y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
        vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
        vertices = torch.matmul(vertices, K.transpose(1,2))
        u, v = vertices[:, :, 0], vertices[:, :, 1]
        v = orig_size - v
        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        u = 2 * (u - orig_size / 2.) / orig_size
        v = 2 * (v - orig_size / 2.) / orig_size
        vertices = torch.stack([u, v, z], dim=-1)
        return vertices
    
    def extract_tex(self, uv_img, uv_sampler, repeat=True):
        """
        :param uv_img: (bs, 3, h, w)
        :param uv_sampler: (bs, nf, T*T, 2)
        :return:
        """
        uv_sampler = uv_sampler * 2 - 1
        # (bs, 3, nf, T*T)
        tex = F.grid_sample(uv_img, uv_sampler)
        # (bs, 3, nf, T, T)
        tex = tex.view(-1, 3, self.nf, self.tex_size, self.tex_size)
        # (bs, nf, T, T, 3)
        tex = tex.permute(0, 2, 3, 4, 1)
        # (bs, nf, T, T, T, 3)
        if repeat:
            tex = tex.unsqueeze(4).repeat(1, 1, 1, 1, self.tex_size, 1)

        return tex
    
    @staticmethod
    def create_coords(tex_size=3):
        """
        :param tex_size: int
        :return: 2 x (tex_size * tex_size)
        """
        if tex_size == 1:
            step = 1
        else:
            step = 1 / (tex_size - 1)

        alpha_beta = torch.arange(0, 1+step, step, dtype=torch.float32).cuda()
        xv, yv = torch.meshgrid([alpha_beta, alpha_beta])

        coords = torch.stack([xv.flatten(), yv.flatten()], dim=0)

        return coords
    
    def points_to_sampler(self, f2vts):
        """
        :param f2vts: [batch size, number of faces, 3, 2]
        :return: [batch_size, number of faces, T*T, 2]
        """

        # Compute alpha, beta (this is the same order as NMR)
        nf = f2vts.shape[1]
        v2 = f2vts[:, :, 2]  # (bs, nf, 2)
        v0v2 = f2vts[:, :, 0] - f2vts[:, :, 2]  # (bs, nf, 2)
        v1v2 = f2vts[:, :, 1] - f2vts[:, :, 2]  # (bs, nf, 2)

        # bs x  F x 2 x T*2
        samples = torch.matmul(torch.stack((v0v2, v1v2), dim=-1), self.coords) + v2.view(-1, nf, 2, 1)
        # bs x F x T*2 x 2 points on the sphere
        samples = samples.permute(0, 1, 3, 2)
        samples = torch.clamp(samples, min=-1.0, max=1.0)
        return samples
    
    def extract_tex_from_image(self, images, vertices, repeat=True):
        bs = images.shape[0]
        faces = self.faces.repeat(bs, 1, 1)

        sampler = self.dynamic_sampler(vertices, faces)  # (bs, nf, T*T, 2)

        tex = self.extract_tex(images, sampler, repeat=repeat)

        return tex
    
    def dynamic_sampler(self, vertices, faces):
        # ipdb.set_trace()
        points = vertices[:, :, :2].clone() # (bs, nv, 2)
        points[:, :, 1] *= -1
        points = (points + 1) / 2
        faces_points = self.points_to_faces(points, faces)   # (bs, nf, 3, 2)
        # print(faces_points.shape)
        sampler = self.points_to_sampler(faces_points)  # (bs, nf, T*T, 2)
        return sampler
    
    def points_to_faces(self, points, faces=None):
        """
        :param points:
        :param faces
        :return:
        """
        bs, nv = points.shape[:2]
        device = points.device

        if faces is None:
            faces = self.faces.repeat(bs, 1, 1)

        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        points = points.reshape((bs * nv, 2))
        # pytorch only supports long and byte tensors for indexing
        return points[faces.long()]
    
    def debug_textures(self):
        return torch.ones((self.nf, self.tex_size, self.tex_size, self.tex_size, 3), dtype=torch.float32)

    def get_vis_f2pts(f2pts, fims):
        """
        Args:
            f2pts: (bs, f, 3, 2) or (bs, f, 3, 3)
            fims:  (bs, 256, 256)
        Returns:
        """

        def get_vis(orig_f2pts, fim):
            """
            Args:
                orig_f2pts: (f, 3, 2) or (f, 3, 3)
                fim: (256, 256)
            Returns:
                vis_f2pts: (f, 3, 2)
            """
            vis_f2pts = torch.zeros_like(orig_f2pts) - 2.0
            # 0 is -1
            face_ids = fim.unique()[1:].long()
            vis_f2pts[face_ids] = orig_f2pts[face_ids]

            return vis_f2pts

        # import ipdb
        # ipdb.set_trace()
        if f2pts.dim() == 4:
            all_vis_f2pts = []
            bs = f2pts.shape[0]
            for i in range(bs):
                all_vis_f2pts.append(get_vis(f2pts[i], fims[i]))

            all_vis_f2pts = torch.stack(all_vis_f2pts, dim=0)

        else:
            all_vis_f2pts = get_vis(f2pts, fims)

        return all_vis_f2pts