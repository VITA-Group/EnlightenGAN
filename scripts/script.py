import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
	os.system("python train.py \
		--dataroot ../final_dataset \
		--no_dropout \
		--name enlightening \
		--model single \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
		--fineSize 320 \
        --patchSize 32 \
		--skip 1 \
		--batchSize 32 \
        --self_attention \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
		--instance_norm 0 \
		--vgg 1 \
        --vgg_choose relu5_1 \
		--gpu_ids 0,1,2 \
		--display_port=" + opt.port)

elif opt.predict:
	for i in range(1):
	        os.system("python predict.py \
	        	--dataroot ../test_dataset \
	        	--name enlightening \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
                --self_attention \
                --times_residual \
	        	--instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch " + str(200 - i*5))