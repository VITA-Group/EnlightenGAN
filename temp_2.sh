#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_lsgan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_vary_bs32_BN_nonormDlayer5_final_ragan_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --train --port=13002
ps -ef|grep single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=4,5,6 python scripts/single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9
