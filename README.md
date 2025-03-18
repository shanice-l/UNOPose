# UNOPose

This repo provides for the implementation of the CVPR'25 paper:

**UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image**
[[arXiv](https://arxiv.org/abs/2411.16106)]

NOTE: UNO (/ˈuːnoʊ/) means one in Spanish and Italian.

## Overview
Given a query image presenting a target object unseen during training, we aim to estimate its segmentation and 6DoF pose w.r.t. a reference frame. While previous methods often rely on the CAD model or multiple RGB(-D) images for reference, we merely use one unposed RGB-D reference image.
![Teaser](./assets/teaser.jpg "")

The network architecture of UNOPose.
![Net](./assets/net.jpg "")

## Dependencies

```
conda create --name unopose python=3.10.12
conda activate unopose
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt

# install bop toolkit for evaluation
cd third_party/bop_toolkit
pip install -r requirements.txt -e .
```

## Datasets
Prepare datasets folder like this:

```
datasets/
├── BOP_DATASETS
    ├──ycbv
        ├──test # download from BOP website
        └──test_ref_targets_crossscene_rot50.json # provided by us
├── segmentation
    └── CustomSamAutomaticMaskGenerator_test_oneref_targets_crossscene_rot50_refvisib_ycbv.json # provided by us
└── MegaPose-Training-Data # Optional, for re-training
    ├──MegaPose-GSO # download from BOP website
        ├── Google_Scanned_Objects # models
        └── train_pbr_web # data
    ├──megapose_gso_fixed_obj_id_to_visib0_8_scene_im_inst_ids.json # provided by us
    ├──megapose_gso_fixed_valid_inst_ids.json # provided by us
    ├──MegaPose-ShapeNetCore # download from BOP website
        ├── shapenetcorev2 # models
        └── train_pbr_web # data
    ├──megapose_shapenetcore_fixed_obj_id_to_visib0_8_scene_im_inst_ids.json # provided by us
    └──megapose_shapenetcore_fixed_valid_inst_ids.json # provided by us
```

## Reproduce the results
Download checkpoints from #TODO, and put it into <ckpt_path>.

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