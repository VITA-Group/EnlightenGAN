#!/bin/bash

CUDA_VISIBLE_DEVICES=6,0,3 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py --train --port=13003
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=6,0,3 python scripts/single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py --predict
ps -ef|grep single3_unet_conv_add_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=6,0,3 python scripts/single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py --train --port=13003
ps -ef|grep single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=6,0,3 python scripts/single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py --predict
ps -ef|grep single3_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_3.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=6,0,3 python scripts/single3_unet_conv_add_vary_lighten_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --train --port=13003
ps -ef|grep single3_unet_conv_add_vary_lighten_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9

CUDA_VISIBLE_DEVICES=6,0,3 python scripts/single3_unet_conv_add_vary_lighten_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py --predict
ps -ef|grep single3_unet_conv_add_vary_lighten_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1.py|grep -v grep|cut -c 9-15|xargs kill -9