# encoding: utf-8
"""This file includes necessary params, info."""
import mmengine
import os.path as osp
import numpy as np
from dataclasses import dataclass
from lib.utils.utils import lazy_property

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
# directory storing experiment data (result, model checkpoints, etc).
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")
wildrgbd_root = osp.join(data_root, "wildrgbd/")


@dataclass
class wildrgbd:
    dataset_root = wildrgbd_root

    # object info
    objects = [
        "mouse",
    ]
    id2obj = {
        1: "mouse",
    }
    obj_num = len(id2obj)
    obj2id = {_name: _id for _id, _name in id2obj.items()}

    # Camera info
    width = 640
    height = 480
    zNear = 0.25
    zFar = 6.0
    center = (height / 2, width / 2)
    camera_matrix = np.array(
        [
            [599.68212890625, 0, 0],
            [0, 599.68212890625, 0],
            [240.56716918945312, 317.58502197265625, 1],
        ]
    )

    @lazy_property
    def objects(self):
        return list(self.id2obj.values())

    @lazy_property
    def obj_num(self):
        return len(self.id2obj)

    @lazy_property
    def obj2id(self):
        return {_name: _id for _id, _name in self.id2obj.items()}
