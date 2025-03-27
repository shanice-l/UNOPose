import os.path as osp

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.join(cur_dir, "../")

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params
from omegaconf import OmegaConf

import torch

from lib.torch_utils.solver.lr_scheduler import flat_and_anneal_lr_scheduler

from core.unopose.model.oneref_grf_predator_pose_estimation_model import UNOPose

from core.unopose.provider.pfoneref_training_dataset_v2 import DatasetPoseFreeOneRefv2
from core.unopose.provider.pfoneref_bop_test_dataset_v2 import BOPTestsetPoseFreeOneRefv2

from core.unopose.provider.build_data_loader import build_train_loader, build_test_loader


# NOTE: change this for other datasets
# one shard has ~1000 samples
dataset_len = 2008971  # 1000 * 1000 * 2  # TODO: update
train_batch_size_per_rank = 8
world_size = 4
train_batch_size = train_batch_size_per_rank * world_size
iters_per_epoch = dataset_len // train_batch_size
# if bs=12 (in code), sam6d epoch~3.58
# num_epoch = 0.3
num_epoch = 3
max_iter = int(iters_per_epoch * num_epoch)
# mimic that in sam6d, re-sample M times, each time resample 1/M from the whole dataset
# set to 1 to disable this behavior
resample_times = 1

misc = OmegaConf.create(
    dict(
        output_dir=osp.abspath(__file__).replace("configs/", "output/", 1)[0:-3],
        load_from="",  # model weights
        # load_from_type="official",
        exp_name="Pfoneref50",  # set to "" to auto set
        debug=False,
        train_batch_size=train_batch_size,
        world_size=world_size,
        # options for DistributedDataParallel
        ddp=dict(
            broadcast_buffers=False,
            # find_unused_parameters=True,
            find_unused_parameters=False,
            fp16_compression=False,
        ),
    )
)

train = OmegaConf.create(
    dict(
        max_iter=max_iter,
        resample_times=resample_times,
        eval_period=max_iter,
        checkpointer=dict(
            period=5000,
            max_to_keep=2,
        ),
        clip_grad=dict(
            enabled=False,
            params=dict(
                max_norm=35,
                norm_type=2,
            ),
        ),
        # TODO: model ema
        model_ema=dict(
            enabled=False,
        ),
        seed=1,
        log_period=50,
        amp=dict(enabled=False),
        amp_dtype="bfloat16",
        device="cuda",
        cudnn_benchmark=True,
        vis=False,  # vis train
        vis_img_tbx=False,
    )
)

test = dict(
    amp=dict(enabled=False),
    # mixed_precision="no",
    save_results_only=False,
    oneref_type="v1",
    instance_batch_size=16,
    vis=False,
)


optimizer = L(torch.optim.Adam)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        # weight_decay_norm=0.0,
        # weight_decay_bias=0.0,
        # overrides=dict(backbone=dict(lr=1e-5)),
    ),
    lr=1e-4,
    betas=(0.5, 0.999),
    # weight_decay=1e-4,
    weight_decay=0.0,
    eps=1e-6,
)

# lr scheduler
lr_multiplier = L(flat_and_anneal_lr_scheduler)(
    warmup_method="linear",
    warmup_factor=0.001,
    warmup_iters=1000,
    # to be set
    # optimizer=
    total_iters=max_iter,
    # warmup_iters=iters_per_epoch * 3,
    # anneal_point=5 / (total_epochs - 15),
    # anneal_point=0.72,
    anneal_point=min(1000 / max_iter, 1.0),
    anneal_method="cosine",  # "cosine",
    target_lr_factor=0.0,
)

model = L(UNOPose)(
    cfg=dict(
        coarse_npoint=196,
        fine_npoint=2048,
        feature_extraction=dict(
            vit_type="vit_base_patch14_reg4_dinov2",
            up_type="linear",
            embed_dim=768,
            out_dim=256,
            use_pyramid_feat=True,
            pretrained=True,
            vit_ckpt=osp.join(PROJ_ROOT, "checkpoints/timm_vit_base_patch14_reg4_dinov2_lvd142m.pth"),
            freeze_vit=True,
        ),
        geo_embedding=dict(
            sigma_d=0.2,
            sigma_a=15,
            angle_k=3,
            reduction_a="max",
            hidden_dim=256,
        ),
        coarse_point_matching=dict(
            nblock=3,
            input_dim=256,
            hidden_dim=256,
            out_dim=256,
            temp=0.1,
            sim_type="cosine",
            normalize_feat=True,
            loss_predator_thres=0.15,
            loss_dis_thres=0.3,
            nproposal1=6000,
            nproposal2=300,
        ),
        fine_point_matching=dict(
            nblock=3,
            input_dim=256,
            hidden_dim=256,
            out_dim=256,
            pe_radius1=0.1,
            pe_radius2=0.2,
            focusing_factor=3,
            temp=0.1,
            sim_type="cosine",
            normalize_feat=True,
            loss_predator_thres=0.15,
            loss_dis_thres=0.3,
            use_lrf=True,
            use_xyz=True,
            nsample1=64,
            nsample2=256,
        ),
    ),
)


dataloader = OmegaConf.create(
    dict(
        train=L(build_train_loader)(
            dataset=L(DatasetPoseFreeOneRefv2)(
                cfg=dict(
                    data_dir=osp.join(PROJ_ROOT, "datasets/MegaPose-Training-Data"),
                    img_size=224,
                    n_sample_observed_point=2048,
                    n_sample_model_point=2048,
                    n_sample_template_point=5000,
                    min_visib_fract=0.1,
                    min_px_count_visib=512,
                    shift_range=0.01,
                    rgb_mask_flag=True,
                    dilate_mask=True,
                    rgb_to_bgr=False,
                ),
                # dp: use train_batch_size; ddp: use per rank bs
                num_img_per_epoch=(max_iter // resample_times) * train_batch_size,
            ),
            total_batch_size=train_batch_size,
            num_workers=24,
        ),
        test=L(build_test_loader)(
            dataset=L(BOPTestsetPoseFreeOneRefv2)(
                cfg=dict(
                    data_dir=osp.join(PROJ_ROOT, "datasets/BOP_DATASETS"),
                    ref_targets_name="test_ref_targets_crossscene_rot50.json",  # inner scene
                    img_size=224,
                    n_sample_observed_point=2048,
                    n_sample_model_point=1024,
                    n_sample_template_point=5000,
                    minimum_n_point=8,
                    rgb_mask_flag=True,
                    seg_filter_score=0.25,
                    rgb_to_bgr=False,
                ),
                eval_dataset_name="ycbv",
                detetion_path=osp.join(
                    PROJ_ROOT,
                    "datasets/segmentation/CustomSamAutomaticMaskGenerator_test_oneref_targets_crossscene_rot50_refvisib_ycbv.json",
                ),
            ),
            num_workers=16,
        ),
    )
)

bop_eval = dict(
    split="test",
)

# main exp
