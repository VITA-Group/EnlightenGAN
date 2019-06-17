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
		--name single_unet_add_bs8_BN_unsuperdata_lsgan_0.05vgg \
		--model single \
		--dataset_mode unaligned \
		--which_model_netG sid_unet \
		--fineSize 320 \
		--skip 1 \
		--batchSize 8 \
		--use_norm 1 \
		--use_wgan 0 \
		--instance_norm 0 \
		--vgg 0.01 \
		--display_port=" + opt.port)

elif opt.test:
	for i in range(19):
	        os.system("python test.py \
	        	--dataroot /hdd2/yifan/compete_LOL \
	        	--name single_unet_add_bs8_BN_unsuperdata_lsgan_0.05vgg \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode pair \
	        	--which_model_netG sid_unet \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 \
	        	--which_epoch " + str(i*10+10))

elif opt.predict:
	for i in range(19):
	        os.system("python predict.py \
	        	--dataroot /hdd2/yifan/common_dataset \
	        	--name single_unet_add_bs8_BN_unsuperdata_lsgan_0.05vgg \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch " + str(i*10+10))