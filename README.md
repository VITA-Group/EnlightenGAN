# EnlightenGAN

[EnlightenGAN: Deep Light Enhancement without Paired Supervision](https://arxiv.org/abs/1906.06972)

### Representitive Results
![representive_results](/assets/show_3.png)

### Overal Architecture
![architecture](/assets/arch.png)

##Environment Prearing
```pip install -r requirement.txt```
### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

### Testing process

```python scipts/script.py --predict```

### Dataset preparing

Training data [[Google Drive]](https://drive.google.com/file/d/1ESHwOxF7qKOauNUpGS8q3QL5aK22qJBS/view?usp=sharing) (unpaired images collected from multiple datasets)

Testing data [[Google Drive]](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T?usp=sharing) (including LIME, MEF, NPE, VV, DICP)