import cv2
import torch
import pickle
import numpy as np
from scipy.sparse import coo_matrix

# def sparse_batch_mm(matrix, matrix_batch):
#     """
#     :param matrix: Sparse or dense matrix, size (m, n).
#     :param matrix_batch: Batched dense matrices, size (b, n, k).
#     :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
#     """
    
#     batch_size = matrix_batch.shape[0]
#     # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
#     vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

#     # A matrix-matrix product is a batched matrix-vector product of the columns.
#     # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
#     return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)

def sparse_batch_mm(matrix, batch):
    """
    https://github.com/pytorch/pytorch/issues/14489
    """
    # TODO: accelerate this with batch operations
    return torch.stack([matrix.mm(b) for b in batch], dim=0)

def coo_matrix_to_torch_sparse_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, shape)

def torch_sparse_tensor(indices, value, size):
    coo = coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=size)
    return coo_matrix_to_torch_sparse_tensor(coo)

def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)
        
def save_to_obj(verts, faces, path):
    """
    Save the SMPL model into .obj file.
    Parameter:
    ---------
    path: Path to save.
    """

    with open(path, 'w') as fp:
        fp.write('g\n')
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        fp.write('s off\n')


def load_obj(obj_file):
    with open(obj_file, 'r') as fp:
        verts = []
        faces = []
        vts = []
        #vns = []
        faces_vts = []
        #faces_vns = []

        for line in fp:
            line = line.rstrip()
            line_splits = line.split()
            prefix = line_splits[0]

            if prefix == 'v':
                verts.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

#             elif prefix == 'vn':
#                 vns.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vt':
                vts.append(np.array([line_splits[1], line_splits[2]], dtype=np.float32))

            elif prefix == 'f':
                f = []
                f_vt = []
                f_vn = []
                for p_str in line_splits[1:4]:
                    p_split = p_str.split('/')
                    f.append(p_split[0])
                    f_vt.append(p_split[1])
                    #f_vn.append(p_split[2])

                faces.append(np.array(f, dtype=np.int32) - 1)
                faces_vts.append(np.array(f_vt, dtype=np.int32) - 1)
                #faces_vns.append(np.array(f_vn, dtype=np.int32) - 1)

            else:
                #raise ValueError(prefix)
                continue

        obj_dict = {
            'vertices': np.array(verts, dtype=np.float32),
            'faces': np.array(faces, dtype=np.int32),
            'vts': np.array(vts, dtype=np.float32),
            #'vns': np.array(vns, dtype=np.float32),
            'faces_vts': np.array(faces_vts, dtype=np.int32),
            #'faces_vns': np.array(faces_vns, dtype=np.int32)
        }

        return obj_dict

def get_f2vts(uv_mapping_path, fill_back=False):
    obj_info = load_obj(uv_mapping_path)

    vts = obj_info['vts']
    vts[:, 1] = 1 - vts[:, 1]
    faces_vts = obj_info['faces_vts']

    if fill_back:
        faces_vts = np.concatenate((faces_vts, faces_vts[:, ::-1]), axis=0)

    # F x 3 x 2
    f2vts = vts[faces_vts]

    return f2vts

def get_camera(camera_pkl):
    K = np.array([[camera_pkl['camera_f'][0], 0, camera_pkl['camera_c'][0]],
         [0, camera_pkl['camera_f'][1], camera_pkl['camera_c'][1]],
         [0, 0, 1]])
    R = cv2.Rodrigues(camera_pkl['camera_rt'])[0]
    t = camera_pkl['camera_t']
    dist_coeffs = camera_pkl['camera_k']
    orig_size = max(camera_pkl['height'], camera_pkl['width'])
    camera = {
        'K': K,
        'R': R,
        't': t,
        'dist_coeffs': dist_coeffs,
        'orig_size': orig_size
    }
    return camera

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count