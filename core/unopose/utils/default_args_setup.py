import argparse
import os
import os.path as osp
import sys
import torch
import PIL
from omegaconf import DictConfig
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.collect_env import collect_env_info
from detectron2.config import LazyConfig
from detectron2.utils import comm

from lib.utils.setup_logger import setup_my_logger
from lib.utils.time_utils import get_time_str


def my_default_setup_v2(cfg: DictConfig, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.misc.output_dir
    log_dir = cfg.misc.log_dir
    ckpt_dir = cfg.misc.ckpt_dir
    if comm.is_main_process():
        PathManager.mkdirs(output_dir)
        PathManager.mkdirs(log_dir)
        PathManager.mkdirs(ckpt_dir)

    rank = comm.get_rank()
    setup_my_logger(log_dir, distributed_rank=rank, name="fvcore")
    setup_my_logger(log_dir, distributed_rank=rank, name="detectron2")
    setup_my_logger(log_dir, distributed_rank=rank, name="timm")
    setup_my_logger(log_dir, distributed_rank=rank, name="core")
    setup_my_logger(log_dir, distributed_rank=rank, name="lib")
    logger = setup_my_logger(log_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if comm.is_main_process() and log_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(log_dir, "config.yaml")

        LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = cfg.train.seed
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.train.cudnn_benchmark


def my_default_setup_accelerate(cfg: DictConfig, args, accelerator):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    from accelerate import Accelerator

    accelerator: Accelerator

    output_dir = cfg.misc.output_dir
    log_dir = cfg.misc.log_dir
    ckpt_dir = cfg.misc.ckpt_dir
    if accelerator.is_main_process:
        PathManager.mkdirs(output_dir)
        PathManager.mkdirs(log_dir)
        PathManager.mkdirs(ckpt_dir)

    rank = accelerator.process_index
    setup_my_logger(log_dir, distributed_rank=rank, name="fvcore")
    setup_my_logger(log_dir, distributed_rank=rank, name="detectron2")
    setup_my_logger(log_dir, distributed_rank=rank, name="timm")
    setup_my_logger(log_dir, distributed_rank=rank, name="core")
    setup_my_logger(log_dir, distributed_rank=rank, name="lib")
    logger = setup_my_logger(log_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, accelerator.num_processes))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if accelerator.is_main_process and log_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(log_dir, "config.yaml")

        LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = cfg.train.seed
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.train.cudnn_benchmark


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code
