# UNOPose

This repo provides for the implementation of the CVPR'25 paper:
**UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image**

NOTE: UNO (/ˈuːnoʊ/) means one in Spanish and Italian.

## Overview
Given a query image presenting a target object unseen during training, we aim to estimate its segmentation and 6DoF pose w.r.t. a reference frame. While previous methods often rely on the CAD model or multiple RGB(-D) images for reference, we merely use one unposed RGB-D reference image.
![Teaser](./assets/teaser.jpg "")

The network architecture of UNOPose.
![Net](./assets/net.jpg "")

## Dependencies

## Datasets

## Reproduce the results
[TODO:prepare checkpoints]

## Testing
```
./core/unopose/test_unopose.sh configs/main_cfg.py <gpu_ids> <ckpt_path> (other args)
```

## Training

```
./core/unopose/dp_train_unopose.sh configs/main_cfg.py <gpu_ids> (other args)
```

## Citation
If you find this repo useful in your research, please consider citing:
```
@InProceedings{liu_2025_unopose,
  title     = {{UNOPose}: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image},
  author    = {Liu, Xingyu and Wang, Gu and Zhang, Ruida and Zhang, Chenyangguang and Tombari, Federico and Ji, Xiangyang},
  booktitle = {CVPR},
  month     = {June},
  year      = {2025}
}
```