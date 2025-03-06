import os
import os.path as osp
import mmengine
import torch
import logging

from detectron2.utils.logger import log_first_n
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from core.unopose.utils.my_writer import MyTensorboardXWriter
from lib.torch_utils.misc import nan_to_num


def set_grad_nan_to_0(model):
    # set nan grads to 0
    for param in model.parameters():
        if param.grad is not None:
            nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)


def get_tbx_event_writer(out_dir, backup=False):
    tb_logdir = osp.join(out_dir, "tb")
    mmengine.mkdir_or_exist(tb_logdir)
    if backup and comm.is_main_process():
        old_tb_logdir = osp.join(out_dir, "tb_old")
        mmengine.mkdir_or_exist(old_tb_logdir)
        os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))

    tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="pytorch")
    return tbx_event_writer


def get_total_norm_of_params(model):
    grads = [
        param.grad.detach().flatten() for param in model.parameters() if param.grad is not None and param.requires_grad
    ]
    if len(grads) == 0:
        total_norm = 0.0
    else:
        total_norm = torch.cat(grads).norm()
    return total_norm


def clip_grad(model, clip_args):
    if "max_norm" in clip_args:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), **clip_args)
    else:
        torch.nn.utils.clip_grad_value_(model.parameters(), **clip_args)
        total_norm = get_total_norm_of_params(model)
    return total_norm


def optim_step(losses, model, optimizer, grad_scaler, AMP_ON, clip_grad_cfg=None):
    if AMP_ON:
        grad_scaler.scale(losses).backward()
        set_grad_nan_to_0(model)
        # # Unscales the gradients of optimizer's assigned params in-place
        if clip_grad_cfg is not None and clip_grad_cfg.enabled:
            grad_scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            total_norm = clip_grad(model, clip_grad_cfg.params)
            log_first_n(logging.INFO, "clip grad done")
        else:
            total_norm = get_total_norm_of_params(model)

        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        losses.backward()
        set_grad_nan_to_0(model)
        if clip_grad_cfg is not None and clip_grad_cfg.enabled:
            total_norm = clip_grad(model, clip_grad_cfg.params)
            log_first_n(logging.INFO, "clip grad done")
        else:
            total_norm = get_total_norm_of_params(model)

        optimizer.step()

    storage = get_event_storage()
    storage.put_scalar("total_grad_norm", total_norm)

    # NOTE: if zero grad at end, must call it once at the very beginning
    optimizer.zero_grad(set_to_none=True)
