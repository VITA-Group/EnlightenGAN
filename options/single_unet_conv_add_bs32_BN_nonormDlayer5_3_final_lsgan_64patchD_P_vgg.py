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
		--dataroot /vita1_ssd1/yifan/final_dataset \
		--no_dropout \
		--name single_unet_conv_add_bs32_BN_nonormDlayer5_3_final_lsgan_64patchD_P_vgg \
		--model single \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_resize \
        --which_model_netD no_norm_4 \
        --n_layers_D 5 \
        --n_layers_patchD 3 \
        --patchD \
		--fineSize 320 \
        --patchSize 64 \
		--skip 1 \
		--batchSize 30 \
		--use_norm 1 \
		--use_wgan 0 \
		--instance_norm 0 \
		--vgg 1 \
		--gpu_ids 0,1,2,3 \
		--display_port=" + opt.port)

elif opt.test:
	for i in range(20):
	        os.system("python test.py \
	        	--dataroot /vita1_ssd1/yifan/compete_LOL \
	        	--name single_unet_conv_add_bs32_BN_nonormDlayer5_3_final_lsgan_64patchD_P_vgg \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode pair \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 \
	        	--which_epoch " + str(i*5+100))

elif opt.predict:
	for i in range(20):
	        os.system("python predict.py \
	        	--dataroot /vita1_ssd1/yifan/common_dataset \
	        	--name single_unet_conv_add_bs32_BN_nonormDlayer5_3_final_lsgan_64patchD_P_vgg \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch " + str(200 - i*10))
