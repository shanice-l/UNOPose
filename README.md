# UNOPose

This repo provides for the implementation of the CVPR'25 paper:

**UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image**
[[arXiv](https://arxiv.org/abs/2411.16106)]

## Overview
![<img src='https://github.com/shanice-l/UNOPose/blob/main/pics/network_v1.pdf' width='800'/>](pics/network_v1.pdf)


## Dependencies

## Datasets

## Reproduce the results
[TODO:prepare checkpoints]

## Training

```
./core/unopose/train_unopose.sh configs/main_cfg.py <gpu_ids> (other args)
```

## Testing
```
./core/unopose/test_unopose.sh configs/main_cfg.py <gpu_ids> <ckpt_path> (other args)
```


## Citation
If you find this repo useful in your research, please consider citing:
```
@InProceedings{liu_2025_unopose,
  title     = {{UNOPose}: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image},
  author    = {Liu, Xingyu and Wang, Gu and Zhang, Ruida and Zhang, Chenyangguang and Ji, Xiangyang},
  booktitle = {CVPR},
  month     = {June},
  year      = {2025}
}
```