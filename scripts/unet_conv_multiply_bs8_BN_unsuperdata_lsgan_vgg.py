import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
	os.system("python train.py \
		--dataroot /hdd2/yifan/unsuper_compete \
		--no_dropout \
		--name unet_conv_multiply_bs8_BN_unsuperdata_lsgan_vgg \
		--model cycle_gan \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_resize \
		--fineSize 256 \
		--skip 1 \
		--multiply \
		--l1 0 \
		--batchSize 8 \
		--use_norm 1 \
		--use_wgan 0 \
		--instance_norm 0 \
		--vgg 1 \
		--gpu_ids 0 \
		--display_port=" + opt.port)

elif opt.test:
	for i in range(19):
	        os.system("python test.py \
	        	--dataroot /hdd2/yifan/compete_LOL \
	        	--name unet_conv_multiply_bs8_BN_unsuperdata_lsgan_vgg \
	        	--model pair \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode pair \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--multiply \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 \
	        	--which_epoch " + str(i*10+10))

elif opt.predict:
	for i in range(19):
	        os.system("python predict.py \
	        	--dataroot /hdd2/yifan/common_dataset \
	        	--name unet_conv_multiply_bs8_BN_unsuperdata_lsgan_vgg \
	        	--model cycle_gan \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--multiply \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch " + str(i*10+10))