import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class MultiModel(BaseModel):
    def name(self):
        return 'MultiGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        if opt.isTrain:
            self.netG_A.train()
            self.netG_B.train()
        else:
            self.netG_A.eval()
            self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        if self.opt.skip == 1:
            self.rec_B, self.latent_fake_A = self.netG_A.forward(self.fake_A)
        else:
            self.rec_B = self.netG_A.forward(self.fake_A)

    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        if self.opt.skip == 1:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ("latent_real_A", latent_real_A), ("rec_A", rec_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ("rec_A", rec_A)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_fake = pred_fake.mean()
        else:
            loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        if self.opt.use_wgan:
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, real.data, fake.data)
        else:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            if self.opt.skip == 1:
                self.idt_A, _ = self.netG_A.forward(self.real_B)
            else:
                self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
         # = self.latent_real_A + self.opt.skip * self.real_A
        pred_fake = self.netD_A.forward(self.fake_B)
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        else:
            self.loss_G_A = self.criterionGAN(pred_fake, True)
        self.L1_AB = self.criterionL1(self.fake_B, self.real_B) * self.opt.l1
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.L1_BA = self.criterionL1(self.fake_A, self.real_A) * self.opt.l1
        if self.opt.use_wgan:
            self.loss_G_B = -pred_fake.mean()
        else:
            self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        
        if lambda_A > 0:
            self.rec_A = self.netG_B.forward(self.fake_B)
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        else:
            self.loss_cycle_A = 0
        # Backward cycle loss
        
         # = self.latent_fake_A + self.opt.skip * self.fake_A
        if lambda_B > 0:
            if self.opt.skip == 1:
                self.rec_B, self.latent_fake_A = self.netG_A.forward(self.fake_A)
            else:
                self.rec_B = self.netG_A.forward(self.fake_A)
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        else:
            self.loss_cycle_B = 0
        self.loss_vgg_a = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_A, self.real_B) * self.opt.vgg if self.opt.vgg > 0 else 0
        self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.L1_AB + self.L1_BA + self.loss_cycle_A + self.loss_cycle_B + \
                        self.loss_vgg_a + self.loss_vgg_b + \
                        self.loss_idt_A + self.loss_idt_B
        # self.loss_G = self.L1_AB + self.L1_BA
        self.loss_G.backward()


    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()


    def get_current_errors(self):
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        L1 = (self.L1_AB + self.L1_BA).data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        vgg = (self.loss_vgg_a.data[0] + self.loss_vgg_b.data[0])/self.opt.vgg if self.opt.vgg > 0 else 0
        if self.opt.identity > 0:
            idt = self.loss_idt_A.data[0] + self.loss_idt_B.data[0]
            if self.opt.lambda_A > 0.0:
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ('L1', L1), ('Cyc_A', Cyc_A),
                                    ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ("vgg", vgg), ("idt", idt)])
            else:
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ('L1', L1),
                                    ('D_B', D_B), ('G_B', G_B)], ("vgg", vgg), ("idt", idt))
        else:
            if self.opt.lambda_A > 0.0:
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ('L1', L1), ('Cyc_A', Cyc_A),
                                    ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ("vgg", vgg)])
            else:
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ('L1', L1),
                                    ('D_B', D_B), ('G_B', G_B)], ("vgg", vgg))

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
        
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)

        if self.opt.identity > 0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            if self.opt.lambda_A > 0.0:
                rec_A = util.tensor2im(self.rec_A.data)
                rec_B = util.tensor2im(self.rec_B.data)
                if self.opt.skip > 0:
                    latent_fake_A = util.tensor2im(self.latent_fake_A.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), ('rec_A', rec_A), 
                                        ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('latent_fake_A', latent_fake_A), 
                                        ("idt_A", idt_A), ("idt_B", idt_B)])
                else:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), 
                                    ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ("idt_A", idt_A), ("idt_B", idt_B)])
            else:
                if self.opt.skip > 0:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), 
                                        ('real_B', real_B), ('fake_A', fake_A), ("idt_A", idt_A), ("idt_B", idt_B)])
                else:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                                        ('real_B', real_B), ('fake_A', fake_A), ("idt_A", idt_A), ("idt_B", idt_B)])
        else:
            if self.opt.lambda_A > 0.0:
                rec_A = util.tensor2im(self.rec_A.data)
                rec_B = util.tensor2im(self.rec_B.data)
                if self.opt.skip > 0:
                    latent_fake_A = util.tensor2im(self.latent_fake_A.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), ('rec_A', rec_A), 
                                        ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('latent_fake_A', latent_fake_A)])
                else:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), 
                                    ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
            else:
                if self.opt.skip > 0:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), 
                                        ('real_B', real_B), ('fake_A', fake_A)])
                else:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                                        ('real_B', real_B), ('fake_A', fake_A)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
