import torch
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
from PIL import Image
import PIL
from pdb import set_trace as st


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_size = A_img.size
        B_size = B_img.size
        A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        A_img = A_img.resize(A_size, Image.BICUBIC)
        B_img = B_img.resize(B_size, Image.BICUBIC)


        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        if self.opt.resize_or_crop == 'no':
            pass
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            size = [8,16,22]
            from random import randint
            size_index = randint(0,2)
            Cropsize = size[size_index]*16

            w_offset = random.randint(0, max(0, w - Cropsize - 1))
            h_offset = random.randint(0, max(0, h - Cropsize - 1))

            A_img = A_img[:, h_offset:h_offset + Cropsize,
                   w_offset:w_offset + Cropsize]

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
