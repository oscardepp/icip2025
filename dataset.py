import torch
import random

class XRFDataset(object):
    def __init__(self, xrf, crop_size, k, ups_win, training):
        super(XRFDataset, self).__init__()

        self.shape = xrf.shape[-2:]
        self.xrf = torch.nn.functional.pad(xrf,
                                           pad = [crop_size // 2] * 4,
                                           mode = 'reflect')
        self.crop_size = crop_size
        self.k = k
        self.ups_win = ups_win
        self.training = training

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __getitem__(self, idx):
        
        # Get crop
        r1 = idx // self.shape[1]
        c1 = idx % self.shape[1]
        r2 = r1 + self.crop_size
        c2 = c1 + self.crop_size
        crop = self.xrf[:,r1:r2,c1:c2]

        if not self.training:
            return crop, r1, c1
        else:
            
            # Rotation augment
            crop = crop.rot90(random.randint(0,4),
                              dims = [-2, -1])
            # Get k pixel mask
            k_pix = torch.randperm(self.crop_size ** 2)[:self.k]
            # print(f'k_pix shape: {k_pix.shape}')
            pix_inds = torch.stack((k_pix // self.crop_size,
                                    k_pix % self.crop_size))
            # print(f'k_pix shape: {pix_inds.shape}')

            # Get UPS pixels
            w = self.ups_win // 2
            ups_inds = (pix_inds + torch.randint(-w, w+1, pix_inds.shape)).clip(0, crop.shape[-2]-1)

            return crop, pix_inds, ups_inds
