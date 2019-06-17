import os
for i in range(15):
        os.system("python test.py \
        	--dataroot /home/yifan/compete_LOL \
        	--name Unet_unpair_add_l1_10_wgan_compete_LOL \
        	--model pair \
        	--which_direction AtoB \
        	--no_dropout \
        	--dataset_mode pair \
        	--which_model_netG unet_256 \
        	--skip 1 \
        	--use_norm 1 \
        	--use_wgan 1 \
        	--which_epoch " + str(i*10+30))

