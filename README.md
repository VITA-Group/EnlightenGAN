# EnlightenGAN: Deep Light Enhancement without Paired Supervision
[Yifan Jiang](https://yifanjiang19.github.io/), Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, Zhangyang Wang

[[Paper]](https://arxiv.org/abs/1906.06972) [[Supplementary Materials]](https://yifanjiang.net/files/EnlightenGAN_Supplementary.pdf)


### Representitive Results
![representive_results](/assets/show_3.png)

### Overal Architecture
![architecture](/assets/arch.png)

## Environment Preparing
```
python3.5
```
You should prepare at least 3 1080ti gpus or change the batch size. 


```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and then put it into the directory `model`.

### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

### Testing process

Download [pretrained model](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view?usp=sharing) and put it into `./checkpoints/enlightening`

Create directories `../test_dataset/testA` and `../test_dataset/testB`. Put your test images on `../test_dataset/testA` (And you should keep whatever one image in `../test_dataset/testB` to make sure program can start.)

Run

```python scripts/script.py --predict ```

### Dataset preparing

Training data [[Google Drive]](https://drive.google.com/drive/folders/1KivxOm79VidSJnJrMV9osr751UD68pCu?usp=sharing) (unpaired images collected from multiple datasets)

Testing data [[Google Drive]](https://drive.google.com/drive/folders/1PrvL8jShZ7zj2IC3fVdDxBY1oJR72iDf?usp=sharing) (including LIME, MEF, NPE, VV, DICP)

And [[BaiduYun]](https://github.com/TAMU-VITA/EnlightenGAN/issues/28) is available now thanks to @YHLelaine!

### Faster Inference
https://github.com/arsenyinfo/EnlightenGAN-inference from @arsenyinfo



If you find this work useful for you, please cite
```
@article{jiang2021enlightengan,
  title={Enlightengan: Deep light enhancement without paired supervision},
  author={Jiang, Yifan and Gong, Xinyu and Liu, Ding and Cheng, Yu and Fang, Chen and Shen, Xiaohui and Yang, Jianchao and Zhou, Pan and Wang, Zhangyang},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={2340--2349},
  year={2021},
  publisher={IEEE}
}
```
