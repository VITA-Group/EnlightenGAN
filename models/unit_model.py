import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.util import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .unit_network import *
import sys

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)

class UNITModel(BaseModel):
    def name(self):
        return 'UNITModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        
        self.config = get_config(opt.config)
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.gen_a = VAEGen(self.config['input_dim_a'], self.config['gen'])
        self.gen_b = VAEGen(self.config['input_dim_a'], self.config['gen'])

        if self.isTrain:
            self.dis_a = MsImageDis(self.config['input_dim_a'], self.config['dis'])  # discriminator for domain a
            self.dis_b = MsImageDis(self.config['input_dim_b'], self.config['dis'])  # discriminator for domain b
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.gen_a, 'G_A', which_epoch)
            self.load_network(self.gen_b, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.dis_a, 'D_A', which_epoch)
                self.load_network(self.dis_b, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = self.config['lr']
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # Setup the optimizers
            beta1 = self.config['beta1']
            beta2 = self.config['beta2']
            dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                            lr=self.config['lr'], betas=(beta1, beta2), weight_decay=self.config['weight_decay'])
            self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr=self.config['lr'], betas=(beta1, beta2), weight_decay=self.config['weight_decay'])
            self.dis_scheduler = get_scheduler(self.dis_opt, self.config)
            self.gen_scheduler = get_scheduler(self.gen_opt, self.config)

            # Network weight initialization
            # self.apply(weights_init(self.config['init']))
            self.dis_a.apply(weights_init('gaussian'))
            self.dis_b.apply(weights_init('gaussian'))

            # Load VGG model if needed
            if 'vgg_w' in self.config.keys() and self.config['vgg_w'] > 0:
                self.vgg = load_vgg16(self.config['vgg_model_path'] + '/models')
                self.vgg.eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
        self.gen_a.cuda()
        self.gen_b.cuda()
        self.dis_a.cuda()
        self.dis_b.cuda()

        print('---------- Networks initialized -------------')
        networks.print_network(self.gen_a)
        networks.print_network(self.gen_b)
        if self.isTrain:
            networks.print_network(self.dis_a)
            networks.print_network(self.dis_b)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_A = Variable(self.input_A.cuda())
        self.real_B = Variable(self.input_B.cuda())

    # def forward(self):
        # self.real_A = Variable(self.input_A)
        # self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A.cuda(), volatile=True)
        self.real_B = Variable(self.input_B.cuda(), volatile=True)
        h_a, n_a = self.gen_a.encode(self.real_A)
        h_b, n_b = self.gen_b.encode(self.real_B)
        x_a_recon = self.gen_a.decode(h_a + n_a) + x_a*1
        x_b_recon = self.gen_b.decode(h_b + n_b) + x_b*1
        x_ba = self.gen_a.decode(h_b + n_b) + x_b*1
        x_ab = self.gen_b.decode(h_a + n_a) + x_a*1
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) + x_ab*1 if self.config['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) + x_ba*1 if self.config['recon_x_cyc_w'] > 0 else None
        self.x_a_recon, self.x_ab, self.x_aba = x_a_recon, x_ab, x_aba
        self.x_b_recon, self.x_ba, self.x_bab = x_b_recon, x_ba, x_bab

    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def optimize_parameters(self):
        self.gen_update(self.real_A, self.real_B)
        self.dis_update(self.real_A, self.real_B)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a) + 0*x_a
        x_b_recon = self.gen_b.decode(h_b + n_b) + 0*x_b
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b) + 0*x_b
        x_ab = self.gen_b.decode(h_a + n_a) + 0*x_a
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) +  0*x_ab if self.config['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) + 0*x_ba if self.config['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if self.config['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if self.config['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = self.config['gan_w'] * self.loss_gen_adv_a + \
                              self.config['gan_w'] * self.loss_gen_adv_b + \
                              self.config['recon_x_w'] * self.loss_gen_recon_x_a + \
                              self.config['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              self.config['recon_x_w'] * self.loss_gen_recon_x_b + \
                              self.config['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              self.config['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              self.config['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              self.config['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              self.config['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              self.config['vgg_w'] * self.loss_gen_vgg_a + \
                              self.config['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()
        self.x_a_recon, self.x_ab, self.x_aba = x_a_recon, x_ab, x_aba
        self.x_b_recon, self.x_ba, self.x_bab = x_b_recon, x_ba, x_bab

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def dis_update(self, x_a, x_b):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = self.config['gan_w'] * self.loss_dis_a + self.config['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def get_current_errors(self):
        D_A = self.loss_dis_a.data[0]
        G_A = self.loss_gen_adv_a.data[0]
        kl_A = self.loss_gen_recon_kl_a.data[0]
        Cyc_A = self.loss_gen_cyc_x_a.data[0]
        D_B = self.loss_dis_b.data[0]
        G_B = self.loss_gen_adv_b.data[0]
        kl_B = self.loss_gen_recon_kl_b.data[0]
        Cyc_B = self.loss_gen_cyc_x_b.data[0]
        if self.config['vgg_w'] > 0:
            vgg_A = self.loss_gen_vgg_a
            vgg_B = self.loss_gen_vgg_b
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('kl_A', kl_A), ('vgg_A', vgg_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('kl_B', kl_B), ('vgg_B', vgg_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('kl_A', kl_A), ('Cyc_A', Cyc_A), 
                                ('D_B', D_B), ('G_B', G_B), ('kl_B', kl_B), ('Cyc_B', Cyc_B)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        recon_A = util.tensor2im(self.x_a_recon.data)
        A_B = util.tensor2im(self.x_ab.data)
        ABA = util.tensor2im(self.x_aba.data)
        real_B = util.tensor2im(self.real_B.data)
        recon_B = util.tensor2im(self.x_b_recon.data)
        B_A = util.tensor2im(self.x_ba.data)
        BAB = util.tensor2im(self.x_b_recon.data)
        return OrderedDict([('real_A', real_A), ('A_B', A_B), ('recon_A', recon_A), ('ABA', ABA),
                            ('real_B', real_B), ('B_A', B_A), ('recon_B', recon_B), ('BAB', BAB)])

    def save(self, label):
        self.save_network(self.gen_a, 'G_A', label, self.gpu_ids)
        self.save_network(self.dis_a, 'D_A', label, self.gpu_ids)
        self.save_network(self.gen_b, 'G_B', label, self.gpu_ids)
        self.save_network(self.dis_b, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.config['lr'] / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.gen_a.param_groups:
            param_group['lr'] = lr
        for param_group in self.gen_b.param_groups:
            param_group['lr'] = lr
        for param_group in self.dis_a.param_groups:
            param_group['lr'] = lr
        for param_group in self.dis_b.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr