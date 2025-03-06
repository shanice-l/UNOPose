# encoding: utf-8
"""This file includes necessary params, info."""
import os
import mmengine
from pathlib import Path
import orjson as json
import os.path as osp
from dataclasses import dataclass
import numpy as np
from lib.utils.utils import lazy_property


# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
# directory storing experiment data (result, model checkpoints, etc).
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")


@dataclass
class shapenet_bop23:
    dataset_root = osp.join(data_root, "bop_web_datasets/shapenet_1M")
    models_root = osp.join(data_root, "bop_web_datasets/shapenet_model/models_normalized_ply/")

    # ---------------------------------------------------------------- #
    # SHAPENET DATASET
    # ---------------------------------------------------------------- #
    model_prefix = ""  # "shapenet_"

    @lazy_property
    def model_name_ids(self):
        # List of Dicts
        return json.loads(Path(osp.join(self.dataset_root, "shapenet_models.json")).read_bytes())

    @lazy_property
    def id2obj(self):
        id2obj_dict = dict()
        for model in self.model_name_ids:
            obj_syn_id, obj_src_id = model["shapenet_synset_id"], model["shapenet_source_id"]
            obj = f"{self.model_prefix}{obj_syn_id}_{obj_src_id}"
            id2obj_dict[model["obj_id"]] = obj
        return id2obj_dict

    @lazy_property
    def objects(self):
        return list(self.id2obj.values())

    @lazy_property
    def valid_objects(self):
        return mmengine.load(osp.join(self.models_root, "valid_objects.json"))

    @lazy_property
    def obj_num(self):
        return len(self.id2obj)

    @lazy_property
    def obj2id(self):
        return {_name: _id for _id, _name in self.id2obj.items()}

    # TODO: diameter
    vertex_scale = 0.001

    # TODO: Camera info
    width = 720
    height = 540
    zNear = 0.25
    zFar = 6.0
    center = (height / 2, width / 2)
    camera_matrix = None  # diff focal length!

    def model_path(self, obj_name):
        obj_syn_id, obj_src_id = obj_name.removeprefix(self.model_prefix).split("_")
        return osp.join(self.models_root, obj_syn_id, obj_src_id, "models/model_normalized_scaled.ply")

    @lazy_property
    def fps_points(self):
        fps_points_path = osp.join(self.models_root, "fps_points.pkl")
        assert osp.exists(fps_points_path), fps_points_path
        fps_dict = mmengine.load(fps_points_path)
        return fps_dict

    @lazy_property
    def keypoints_3d(self):
        keypoints_3d_path = osp.join(self.models_root, "keypoints_3d.pkl")
        assert osp.exists(keypoints_3d_path), keypoints_3d_path
        kpts_dict = mmengine.load(keypoints_3d_path)
        return kpts_dict

    @lazy_property
    def extents_3d(self):
        extents_3d_path = osp.join(self.models_root, "extents_3d.pkl")
        assert osp.exists(extents_3d_path), extents_3d_path
        extents_dict = mmengine.load(extents_3d_path)
        return extents_dict
