import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
import os.path as osp
from pathlib import Path
from loguru import logger as loguru_logger
import logging

import copy

from setproctitle import setproctitle
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.engine import launch, default_argument_parser, create_ddp_model
from detectron2.data import MetadataCatalog
from detectron2.config import LazyConfig, instantiate
from detectron2.utils import comm

import cv2

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from core.unopose.utils.default_args_setup import my_default_setup_v2
from core.unopose.utils.my_setup import setup_for_distributed
from core.unopose.utils.my_checkpoint import MyCheckpointer


from lib.utils.utils import iprint
from lib.utils.time_utils import get_time_str


from core.unopose.engine.engine import do_train, do_save_results


logger = logging.getLogger("detectron2")


def setup(args):
    """Create configs and perform basic setups."""
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    ############## pre-process some cfg options ######################
    out_root = cfg.misc.output_dir
    # log dir (with timestamp)
    cfg.misc.log_dir = osp.join(out_root, f"logs_{get_time_str()}")
    cfg.misc.ckpt_dir = osp.join(out_root, "ckpts")  # no timestamp

    if cfg.misc.get("exp_name", "") == "":
        proc_title = "{}.{}".format(Path(args.config_file).stem, get_time_str())
    else:
        proc_title = "{}.{}".format(cfg.misc.exp_name, get_time_str())
    if args.eval_only:
        proc_title += "-eval_only"
    if cfg.test.save_results_only:
        proc_title += "-save_results_only"
    setproctitle(proc_title)

    if torch.cuda.get_device_capability() <= (6, 1):
        iprint("Disable AMP for older GPUs")
        cfg.train.amp.enabled = False
        cfg.test.amp.enabled = False

    # -------------------------------------------------------------------------
    if cfg.misc.get("debug", False):
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.dataloader.train.num_workers = 0
        cfg.dataloader.test.num_workers = 0
        if "persistent_workers" in cfg.dataloader.train:
            cfg.dataloader.train.persistent_workers = False
        if "pin_memory" in cfg.dataloader.train:
            cfg.dataloader.train.pin_memory = False
        if "train2" in cfg.dataloader:
            cfg.dataloader.train2.num_workers = 0
        cfg.train.log_period = 1
    # register datasets
    # register_datasets_in_cfg(cfg)

    exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])

    if args.eval_only:
        exp_id += "_test"
    cfg.misc.exp_id = exp_id
    cfg.misc.resume = args.resume
    ####################################

    my_default_setup_v2(cfg, args)
    setup_for_distributed(is_master=comm.is_main_process())
    return cfg


# @loguru_logger.catch
def main(args):
    cfg = setup(args)
    distributed = (comm.get_world_size() > 1) and (not args.use_dp)

    # model, optimizer = eval(cfg.MODEL.UNSPRE.NAME).build_model_optimizer(cfg, is_test=args.eval_only)
    model = instantiate(cfg.model)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    if True:
        params_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / (1024 * 1024)
        params = sum(p.numel() for p in model.parameters()) / (1024 * 1024)
        logger.info(f"{params}M params, {params_train}M trainable params")

    if args.eval_only or cfg.test.save_results_only:
        MyCheckpointer(model, save_dir=cfg.misc.ckpt_dir).resume_or_load(cfg.misc.load_from, resume=args.resume)

    if cfg.test.save_results_only:  # save results only ------------------------------
        return do_save_results(cfg, model)

    # if args.eval_only:  # eval only --------------------------------------------------
    #     return do_test(cfg, model)

    optim_cfg = copy.deepcopy(cfg.optimizer)
    optim_cfg.params.model = model
    optimizer = instantiate(optim_cfg)

    if args.use_dp and args.num_gpus > 1:
        model = torch.nn.DataParallel(model, range(args.num_gpus))
    if distributed:
        model = create_ddp_model(model, **cfg.misc.ddp)

    if args.use_dp:
        assert cfg.misc.get("world_size", 1) == args.num_gpus
    else:
        assert comm.get_world_size() == cfg.misc.get("world_size", 1)

    if cfg.train.get("matmul_dtype", "float32") == "tf32":
        logger.warning("matmul use tf32")
        torch.backends.cuda.matmul.allow_tf32 = True

    do_train(cfg, args, model, optimizer, resume=args.resume)
    # return do_test(cfg, model)
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        return do_save_results(cfg, model.module)
    else:
        return do_save_results(cfg, model)


if __name__ == "__main__":
    import resource

    # RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(500000, hard_limit)
    iprint("soft limit: ", soft_limit, "hard limit: ", hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    parser = default_argument_parser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--use_dp", action="store_true", help="whether to use nn.DataParallel")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    iprint("Command Line Args:", args)

    if args.eval_only:
        torch.multiprocessing.set_sharing_strategy("file_system")

    if args.use_dp:
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
