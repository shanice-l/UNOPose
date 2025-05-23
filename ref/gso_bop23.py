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
class gso_bop23:
    dataset_root = osp.join(data_root, "bop_web_datasets/gso_1M")
    models_root = osp.join(data_root, "bop_web_datasets/gso_model/models_normalized_ply/")

    # ---------------------------------------------------------------- #
    # GSO DATASET
    # ---------------------------------------------------------------- #
    model_prefix = ""  # "gso_"

    @lazy_property
    def model_name_ids(self):
        # List of Dicts
        return json.loads(Path(osp.join(self.dataset_root, "gso_models.json")).read_bytes())

    @lazy_property
    def id2obj(self):
        return {model["obj_id"]: f'{self.model_prefix}{model["gso_id"]}' for model in self.model_name_ids}

    @lazy_property
    def objects(self):
        return list(self.id2obj.values())

    @lazy_property
    def invalid_objects(self):
        invalid_names = json.loads(
            Path(osp.join(data_root, "bop_web_datasets/gso_model/invalid_meshes.json")).read_bytes()
        )
        return [f"{self.model_prefix}{_n}" for _n in invalid_names]

    @lazy_property
    def valid_objects(self):
        return [obj for obj in self.objects if obj not in self.invalid_objects]

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
        return osp.join(self.models_root, obj_name.removeprefix(self.model_prefix), "meshes/model.ply")

    @lazy_property
    def fps_points(self):
        """key is obj_name generated by
        core/unspre/tools/gso_bop23/compute_fps.py."""
        fps_points_path = osp.join(self.models_root, "fps_points.pkl")
        assert osp.exists(fps_points_path), fps_points_path
        fps_dict = mmengine.load(fps_points_path)
        return fps_dict

    @lazy_property
    def keypoints_3d(self):
        """key is obj_name generated by
        core/unspre/tools/gso_bop23/compute_kps.py."""
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
