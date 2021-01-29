# EnlightenGAN

IEEE Transaction on Image Processing, 2020, [EnlightenGAN: Deep Light Enhancement without Paired Supervision](https://arxiv.org/abs/1906.06972)

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

```python scripts/script.py --predict```

### Dataset preparing

Training data [[Google Drive]](https://drive.google.com/drive/folders/1fwqz8-RnTfxgIIkebFG2Ej3jQFsYECh0?usp=sharing) (unpaired images collected from multiple datasets)

Testing data [[Google Drive]](https://drive.google.com/open?id=1PrvL8jShZ7zj2IC3fVdDxBY1oJR72iDf) (including LIME, MEF, NPE, VV, DICP)

And [[BaiduYun]](https://github.com/TAMU-VITA/EnlightenGAN/issues/28) is available now thanks to @YHLelaine!


If you find this work useful for you, please cite
```
@ARTICLE{9334429,
  author={Y. {Jiang} and X. {Gong} and D. {Liu} and Y. {Cheng} and C. {Fang} and X. {Shen} and J. {Yang} and P. {Zhou} and Z. {Wang}},
  journal={IEEE Transactions on Image Processing}, 
  title={EnlightenGAN: Deep Light Enhancement Without Paired Supervision}, 
  year={2021},
  volume={30},
  number={},
  pages={2340-2349},
  doi={10.1109/TIP.2021.3051462}}
```
