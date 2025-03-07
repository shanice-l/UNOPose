import logging
import os
import os.path as osp
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent.parent.parent
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel

import time
import numpy as np
from collections import OrderedDict

from detectron2.config import instantiate
from detectron2.utils.events import EventStorage
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.utils.logger import log_first_n
from detectron2.engine import create_ddp_model

from detectron2.utils import comm

from core.unopose.utils.my_checkpoint import MyCheckpointer
from core.unopose.utils.my_writer import MyCommonMetricPrinter, MyJSONWriter
from core.unopose.utils.data_utils import denormalize_image

from core.unopose.utils.loss_utils import process_loss
from core.unopose.engine.engine_utils import optim_step, get_tbx_event_writer
from core.unopose.engine.inference_utils import inference_and_save
from core.unopose.engine.oneref_inference_utils_v1 import inference_and_save_oneref_v1

logger = logging.getLogger(__name__)


def do_save_results(cfg, model, iteration=None):
    model_name = Path(cfg.misc.load_from).stem
    # currently only support the first one;
    # for multiple datasets, we can let cfg.dataloader.test to be a list as well
    # dataset_name = cfg.DATASETS.TEST[0]
    dataset_name = cfg.dataloader.test.dataset.eval_dataset_name
    if iteration is not None:
        save_out_dir = osp.join(cfg.misc.output_dir, f"inference_iter_{iteration}", dataset_name)
    else:
        save_out_dir = osp.join(cfg.misc.output_dir, f"inference_{model_name}", dataset_name)
    data_loader = instantiate(cfg.dataloader.test)

    os.makedirs(save_out_dir, exist_ok=True)
    # f"{method_name}_{val_cfg.DATASET_NAME}-{val_cfg.SPLIT}{split_type_str}.csv"
    save_name = f"result{cfg.misc.get('exp_name', '')}_{dataset_name}-{cfg.bop_eval.split}.csv"
    save_path = osp.join(save_out_dir, save_name)

    if "oneref_type" in cfg.test:
        if cfg.test.oneref_type == "v1":
            infer_func = inference_and_save_oneref_v1
        else:
            raise NotImplementedError(f"Unsupported oneref_type: {cfg.test.oneref_type}")
    else:
        infer_func = inference_and_save
    logger.info(f"infer_func: {infer_func}")

    if cfg.test.amp.enabled:
        logger.info("amp test")
    with torch.cuda.amp.autocast(enabled=cfg.test.amp.enabled):
        infer_func(
            model,
            data_loader,
            save_path=save_path,
            # output_dir=save_out_dir,
            instance_batch_size=cfg.test.instance_batch_size,
            # dataset_name=dataset_name,
        )

    logger.info("evaluation...")
    cmd = f"PYTHONPATH=$PYTHONPATH:{PROJ_DIR}:{PROJ_DIR}/third_party/bop_toolkit \
        python {PROJ_DIR}/core/unopose/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types 'vsd,mssd,mspd' \
        --split test \
        --dataset {dataset_name} \
        --result_names {save_name} \
        --result_dir {save_out_dir}  "
    logger.info(cmd)
    os.system(cmd)


def do_train(cfg, args, model, optimizer, resume=False):
    model.train()
    # load data ===================================
    # train_dset_names = cfg.DATASETS.TRAIN
    data_loader = instantiate(cfg.dataloader.train)
    data_loader.dataset.reset()
    data_loader_iter = iter(data_loader)

    cfg.lr_multiplier.optimizer = optimizer
    scheduler = instantiate(cfg.lr_multiplier)

    AMP_ON = cfg.train.amp.enabled
    logger.info(f"AMP enabled: {AMP_ON}")
    grad_scaler = GradScaler()

    # resume or load model ===================================
    checkpointer = MyCheckpointer(
        model,
        cfg.misc.ckpt_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        gradscaler=grad_scaler,
        save_to_disk=comm.is_main_process(),
    )
    start_iter = checkpointer.resume_or_load(cfg.misc.load_from, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.train.max_iter

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        cfg.train.checkpointer.period,
        max_iter=max_iter,
        max_to_keep=cfg.train.checkpointer.max_to_keep,
    )

    # build writers ==============================================
    tbx_event_writer = get_tbx_event_writer(cfg.misc.output_dir, backup=not resume)
    tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
    writers = (
        [
            MyCommonMetricPrinter(max_iter),
            MyJSONWriter(osp.join(cfg.misc.output_dir, "metrics.json")),
            tbx_event_writer,
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    logger.info("Starting training from iteration {}".format(start_iter))
    iter_time = None
    with EventStorage(start_iter) as storage:
        optimizer.zero_grad(set_to_none=True)
        for iteration in range(start_iter, max_iter):
            storage.iter = iteration

            if iteration > 0 and iteration % (max_iter // cfg.train.get("resample_times", 1)) == 0:
                # FIXME: this causes cuda OOM (when ddp?)
                logger.warning("reset dataloader to resample image indices")
                data_loader.dataset.reset()
                data_loader_iter = iter(data_loader)

            data = next(data_loader_iter)

            if iter_time is not None:
                storage.put_scalar("time", time.perf_counter() - iter_time)
            iter_time = time.perf_counter()

            batch = data
            for key in batch:
                log_first_n(logging.INFO, f"key: {key}, len: {len(batch[key])}")
                if key not in ["gt_node_corr_indices", "gt_node_corr_overlaps"]:
                    batch[key] = batch[key].cuda()
                else:
                    batch[key] = [item.cuda() for item in batch[key]]
            amp_dtype = torch.bfloat16 if cfg.train.get("amp_dtype", "float16") == "bfloat16" else torch.float16
            with autocast(dtype=amp_dtype, **cfg.train.amp):
                out_dict = model(
                    batch,
                )
                dict_info = process_loss(out_dict)
                loss_all = dict_info["loss"]
                assert torch.isfinite(loss_all).all(), loss_all

                loss_dict = {k: v for k, v in dict_info.items() if "loss" in k and k != "loss"}
                loss_dict_reduced = {f"{k}": v.item() for k, v in comm.reduce_dict(loss_dict).items() if "loss" in k}
                log_first_n(logging.INFO, f"loss_dict_reduced: {loss_dict_reduced}", n=5)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                non_loss_dict = {k: v for k, v in dict_info.items() if "loss" not in k}
                non_loss_dict_reduced = {
                    f"{k}": v.item() for k, v in comm.reduce_dict(non_loss_dict).items() if "loss" not in k
                }
                if comm.is_main_process():
                    storage.put_scalars(**{f"total_loss": losses_reduced})
                    storage.put_scalars(**loss_dict_reduced)
                    # storage.put_scalars(**non_loss_dict_reduced)  # do not print
                    for k, v in non_loss_dict_reduced.items():
                        tbx_writer.add_scalar(k, v, iteration)

            optim_step(loss_all, model, optimizer, grad_scaler, AMP_ON, clip_grad_cfg=cfg.train.clip_grad)
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.train.eval_period > 0
                and ((iteration + 1) % cfg.train.eval_period == 0)
                and (iteration != max_iter - 1)
            ):
                if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
                    do_save_results(cfg, model.module, iteration=iteration)
                else:
                    do_save_results(cfg, model, iteration=iteration)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                ((iteration + 1) % cfg.train.log_period == 0)
                or (iteration == max_iter - 1)
                or (iteration - start_iter < 20)
            ):
                for writer in writers:
                    writer.write()
                # visualize some images ========================================
                if cfg.train.vis_img_tbx:
                    with torch.no_grad():
                        vis_i = 0
                        roi_img_vis = batch["roi_img"][vis_i].cpu().numpy()
                        roi_img_vis = denormalize_image(roi_img_vis, cfg).transpose(1, 2, 0).astype("uint8")
                        tbx_writer.add_image("input_image", roi_img_vis, iteration)

                        gt_mask_vis = batch["roi_mask"][vis_i].detach().cpu().numpy()
                        tbx_writer.add_image("gt_mask", gt_mask_vis, iteration)
            periodic_checkpointer.step(iteration)
