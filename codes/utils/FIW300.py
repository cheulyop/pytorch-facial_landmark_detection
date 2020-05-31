import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FIW300(Dataset):
    '''
    300W Dataset class

    '''

    def __init__(self, data_dir, indoor=True, outdoor=True, size=(224, 224)):
        assert indoor or outdoor, 'Indoor and outdoor cannot be both set to False.'
        self.size = size
        self.config = {
            'indoor': (os.path.join(data_dir, '01_Indoor'), indoor),
            'outdoor': (os.path.join(data_dir, '02_Outdoor'), outdoor)
        }

        self.paths = self._get_paths()
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def _get_paths(self):
        '''
        Get paths to files

        Parameters
        ----------
        None
        
        Returns
        -------
        data: list
            of the form [(img_path, pts_path), ...]

        '''

        img_paths, pts_paths = [], []
        for env in ['indoor', 'outdoor']:
            env_dir, env_flag = self.config[env]

            if env_flag:
                for f_name in os.listdir(env_dir):
                    if 'png' in f_name:
                        img_paths.append(os.path.join(env_dir, f_name))
                    elif 'pts' in f_name:
                        pts_paths.append(os.path.join(env_dir, f_name))

        img_paths.sort()
        pts_paths.sort()
        paths = list(zip(img_paths, pts_paths))
        return paths

    def _get_pts(self, pts_path):
        with open(pts_path) as f:
            lines = [line.strip() for line in f]
            head, tail = lines.index('{')+1, lines.index('}')
            points = lines[head:tail]

            pts = [
                tuple([float(point) for point in point.split()]) 
                for point in points
                ]
                    
        return pts

    def _normalize_pts(self, pts, w, h):
        return [(x/w, y/h) for x, y in pts]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img_path, pts_path = self.paths[i]
        img = Image.open(img_path)
        w, h = img.width, img.height
        img = self.transform(img)

        pts = self._get_pts(pts_path)
        pts = self._normalize_pts(pts, w, h)
        pts = torch.Tensor(pts).view(136)

        return img, pts