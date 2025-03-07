import logging

import torch
from torch.utils.data import dataloader
from detectron2.utils import comm
import operator
from detectron2.data.build import worker_init_reset_seed
from detectron2.data.common import AspectRatioGroupedDataset

from core.unopose.utils.my_distributed_sampler import (
    InferenceSampler,
    TrainingSampler,
)

logger = logging.getLogger(__name__)


# def trivial_batch_collator(batch):
#     """A batch collator that does nothing.

#     https://github.com/pytorch/fairseq/issues/1171
#     """
#     dataloader._use_shared_memory = False
#     return batch


def trivial_batch_collator(batch):
    """A batch collator that does nothing.

    https://github.com/pytorch/fairseq/issues/1171
    """
    dataloader._use_shared_memory = False
    new_batch = {}
    for key in batch[0]:
        if key not in ["gt_node_corr_indices", "gt_node_corr_overlaps"]:
            new_batch[key] = torch.stack([item[key] for item in batch])
        else:
            new_batch[key] = [item[key] for item in batch]
    return new_batch


def my_build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    use_trivial_collate_fn=False,  # only for non aspect ratio grouping
    num_workers=0,
    persistent_workers=False,
):
    """Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.
    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = comm.get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(total_batch_size, world_size)

    batch_size = total_batch_size // world_size

    kwargs = {"num_workers": num_workers}

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size
    if use_trivial_collate_fn:
        kwargs.update(
            collate_fn=trivial_batch_collator,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        worker_init_fn=worker_init_reset_seed,
        persistent_workers=persistent_workers,
        **kwargs,
    )


def build_train_loader(
    dataset,
    total_batch_size,
    sampler_name="TrainingSampler",
    use_trivial_collate_fn=False,  # only for non aspect ratio grouping
    num_workers=4,
    persistent_workers=False,
):
    """

    Returns:
        an infinite iterator of training data
    """
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        use_trivial_collate_fn=use_trivial_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


def build_train_loader_naive(
    dataset,
    batch_size,
    use_trivial_collate_fn=False,  # only for non aspect ratio grouping
    num_workers=4,
    persistent_workers=False,
    shuffle=True,
    pin_memory=False,
    drop_last=True,
    reset_worker_seed=False,
):
    """
    use batch size per device
    """
    kwargs = {"num_workers": num_workers}
    if use_trivial_collate_fn:
        kwargs.update(
            collate_fn=trivial_batch_collator,
        )
    if reset_worker_seed:
        kwargs.update(
            worker_init_fn=worker_init_reset_seed,
        )
    # NOTE: for dp, this will be the total batch size for all gpus
    # for ddp, this is the single device bs
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs,
    )


def build_test_loader(
    dataset,
    #
    num_workers=4,
    batch_size=1,
    use_trivial_collate_fn=False,
):
    """ """

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

    kwargs = {"num_workers": num_workers}
    if use_trivial_collate_fn:
        kwargs.update(
            collate_fn=trivial_batch_collator,
        )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        **kwargs,
    )
    return data_loader
