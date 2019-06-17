import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--test", action='store_true')
opt = parser.parse_args()

if opt.train:
	os.system("python train.py \
		--dataroot /home/yifan/compete_LOL \
		--no_dropout \
		--name unet_unsuper_add_bs8_BN_lsgan_0.5vgg \
		--model cycle_gan \
		--dataset_mode unaligned \
		--which_model_netG sid_unet \
		--skip 1 \
		--l1 0 \
		--batchSize 8 \
		--use_norm 1 \
		--use_wgan 0 \
		--instance_norm 0 \
		--vgg 0.5 \
		--display_port=" + opt.port)

elif opt.test:
	for i in range(19):
	        os.system("python test.py \
	        	--dataroot /home/yifan/compete_LOL \
	        	--name unet_unsuper_add_bs8_BN_lsgan_0.5vgg \
	        	--model pair \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode pair \
	        	--which_model_netG sid_unet \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 \
	        	--which_epoch " + str(i*10+10))