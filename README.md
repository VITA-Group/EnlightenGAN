# EnlightenGAN

[EnlightenGAN: Deep Light Enhancement without Paired Supervision](https://arxiv.org/abs/1906.06972)

### Representitive Results
![representive_results](/assets/show_3.png)

### Overal Architecture
![architecture](/assets/arch.png)

## Environment Preparing
```pip install -r requirement.txt``` </br>
```mkdir model```
Download VGG pretrained model from [Google Drive](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), [Google Drive](https://drive.google.com/file/d/190BBev58S1QRS2nDKQR5Ijx04_GOJgW6/view?usp=sharing) and then put them into the directory `model`.

### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

### Testing process

```python scipts/script.py --predict```

### Dataset preparing

Training data [[Google Drive]](https://drive.google.com/file/d/1ESHwOxF7qKOauNUpGS8q3QL5aK22qJBS/view?usp=sharing) (unpaired images collected from multiple datasets)

Testing data [[Google Drive]](https://drive.google.com/open?id=1PrvL8jShZ7zj2IC3fVdDxBY1oJR72iDf) (including LIME, MEF, NPE, VV, DICP)

If you find this work useful for you, please cite
```
@article{jiang2019enlightengan,
  title={EnlightenGAN: Deep Light Enhancement without Paired Supervision},
  author={Jiang, Yifan and Gong, Xinyu and Liu, Ding and Cheng, Yu and Fang, Chen and Shen, Xiaohui and Yang, Jianchao and Zhou, Pan and Wang, Zhangyang},
  journal={arXiv preprint arXiv:1906.06972},
  year={2019}
}
```