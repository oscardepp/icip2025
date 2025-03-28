import torch
import random

class XRFDataset(object):
    def __init__(self, xrf, crop_size, t0,k, ups_win, training):
        super(XRFDataset, self).__init__()

        self.shape = xrf.shape[-2:]
        self.xrf = torch.nn.functional.pad(xrf,
                                           pad = [crop_size // 2] * 4,
                                           mode = 'reflect')
        self.t0 = torch.nn.functional.pad(t0,
                                          pad=[crop_size // 2] * 4,
                                          mode='reflect')
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

        t0_crop = self.t0[:,r1:r2, c1:c2]


        if not self.training:
            return crop, t0_crop, r1, c1
        else:
            
            # Rotation augment
            num_rot90 = random.randint(0,4)
            crop = crop.rot90(num_rot90,
                              dims = [-2, -1])
            
            # Get k pixel mask
            k_pix = torch.randperm(self.crop_size ** 2)[:self.k]
            pix_inds = torch.stack((k_pix // self.crop_size,
                                    k_pix % self.crop_size))
            # Get UPS pixels
            w = self.ups_win // 2
            ups_offsets = torch.randint(-w, w+1, pix_inds.shape)
            ups_inds = (pix_inds + ups_offsets).clip(0, crop.shape[-2]-1)
            
            rows = torch.arange(crop.shape[-2])
            cols = torch.arange(crop.shape[-1])
            # get all indices for ij 
            r, c = torch.meshgrid(rows, cols, indexing = 'ij')
            # row and col offsets
            row_o = torch.randint(-w, w + 1, (crop.shape[-2], crop.shape[-1]))
            col_o = torch.randint(-w, w + 1, (crop.shape[-2], crop.shape[-1]))
            row_ups = (r + row_o).clip(0, self.crop_size - 1)
            col_ups = (c + col_o).clip(0, self.crop_size - 1)
            crop_all_ups = crop[:, row_ups, col_ups]

            return crop, t0_crop, crop_all_ups, pix_inds, ups_inds

