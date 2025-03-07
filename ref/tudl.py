# encoding: utf-8
"""This file includes necessary params, info."""
import os
import mmengine
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
bop_root = osp.join(data_root, "BOP_DATASETS/")


# --------------------------------------------------------- #
# TUDL DATASET
# ---------------------------------------------------------------- #
@dataclass
class tudl:
    dataset_root = os.path.join(bop_root, "tudl")

    train_real_dir = os.path.join(dataset_root, "train_real")
    train_pbr_dir = os.path.join(dataset_root, "train_pbr")  # bbnc6:/data3/BOP2020/dataset_pbr/tudl_train_pbr
    train_render_dir = osp.join(dataset_root, "train_render")

    test_dir = os.path.join(dataset_root, "test")

    # 3D models
    model_dir = os.path.join(dataset_root, "models")
    model_eval_dir = os.path.join(dataset_root, "models_eval")
    # scaled models (.obj)
    model_scaled_dir = osp.join(dataset_root, "models_rescaled")

    train_scenes = ["{:06d}".format(i) for i in range(50)]
    test_scenes = ["{:06d}".format(i) for i in range(1, 4)]

    id2obj = {1: "dragon", 2: "frog", 3: "can"}

    objects = list(id2obj.values())
    obj_num = len(id2obj)
    obj2id = {_name: _id for _id, _name in id2obj.items()}

    @lazy_property
    def model_paths(self):
        return [osp.join(self.model_dir, "obj_{:06d}.ply").format(_id) for _id in self.id2obj]  # TODO: check this

    texture_paths = None
    model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

    diameters = (
        np.array(
            [
                430.31,
                175.704,
                352.356,
            ]
        )
        / 1000.0
    )

    # Camera info
    width = 640
    height = 480
    zNear = 0.25
    zFar = 6.0
    center = (height / 2, width / 2)

    camera_matrix = np.array([[515.0, 0.0, 321.566], [0.0, 515.0, 214.08], [0, 0, 1]])
    vertex_scale = 0.001

    @lazy_property
    def models_info(self):
        """key is str(obj_id)"""
        models_info_path = osp.join(self.model_dir, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmengine.load(models_info_path)  # key is str(obj_id)
        return models_info

    # ref core/gdrn_modeling/tools/tudl/tudl_1_compute_fps.py
    @lazy_property
    def get_fps_points(self):
        fps_points_path = osp.join(self.model_dir, "fps_points.pkl")
        assert osp.exists(fps_points_path), fps_points_path
        fps_dict = mmengine.load(fps_points_path)
        return fps_dict

    @lazy_property
    # ref core/roi_pvnet/tools/tudl/tudl_1_compute_keypoints_3d.py
    def get_keypoints_3d(self):
        keypoints_3d_path = osp.join(self.model_dir, "keypoints_3d.pkl")
        assert osp.exists(keypoints_3d_path), keypoints_3d_path
        kpts_dict = mmengine.load(keypoints_3d_path)
        return kpts_dict


if __name__ == "__main__":
    data_ref = tudl()
