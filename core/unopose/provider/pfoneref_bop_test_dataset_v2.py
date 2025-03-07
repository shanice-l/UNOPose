import logging
import os
import os.path as osp

# import sys
import imageio

# import orjson as json
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as cocomask

# import open3d as o3d

import torch
import torchvision.transforms as transforms

from core.unopose.utils.data_utils import (
    get_bop_depth_map,
    get_bop_image,
    # get_bop_image_full,
    get_bbox,
    backproject,
    get_resize_rgb_choose,
    load_json,
)
import ref

logger = logging.getLogger(__name__)


class BOPTestsetPoseFreeOneRefv2:
    """
    v2:
        assume only one instance for each obj in an image (lm, lmo, ycbv, ...)
        provide a test_ref_targets.json with ref_scene_id, ref_im_id
        ref's pose is not assumed to be known for network
    """

    def __init__(self, cfg, eval_dataset_name="lmo", detetion_path=None):
        assert detetion_path is not None

        self.cfg = cfg
        self.dataset = eval_dataset_name
        self.data_dir = cfg.data_dir  # datasets/BOP_DATASETS
        # "test_ref_targets.json"
        self.ref_targets_name = cfg.ref_targets_name
        self.rgb_mask_flag = cfg.rgb_mask_flag  # True
        self.img_size = cfg.img_size  # 224
        # self.voxel_size = cfg.voxel_size
        self.n_sample_observed_point = cfg.n_sample_observed_point  # 2048
        # self.n_sample_model_point = cfg.n_sample_model_point  # 1024
        self.n_sample_template_point = cfg.n_sample_template_point  # 5000

        self.minimum_n_point = cfg.minimum_n_point  # 8
        self.seg_filter_score = cfg.seg_filter_score  # 0.25
        self.rgb_to_bgr = cfg.get("rgb_to_bgr", False)  # sam6d is True, which is weird
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # obj_id --> 0-based label
        if cfg.get("obj_idxs", None) is None:
            data_ref = ref.__dict__[eval_dataset_name]()
            self.obj_idxs = {obj_id: idx for idx, obj_id in enumerate(data_ref.id2obj.keys())}
        else:
            self.obj_idxs = cfg.obj_idxs
        # self.obj_idxs = obj_idxs

        self.data_folder = osp.join(self.data_dir, eval_dataset_name, "test")

        if cfg.get("oneref_percat", False):
            assert cfg.targets_name is not None
            self.targets_path = osp.join(self.data_dir, eval_dataset_name, cfg.targets_name)
            logger.info(f"test targets: {self.targets_path}")
            self.test_ref_target = self.load_single_ref_per_dset(self.targets_path, cfg.ref_scene_ims)
        else:
            self.test_ref_target_path = osp.join(self.data_dir, eval_dataset_name, self.ref_targets_name)
            logger.info(f"test ref targets: {self.test_ref_target_path}")
            self.test_ref_target = self.load_ref(self.test_ref_target_path)

        # keys: scene_id, image_id, category_id, bbox, score, segmentation
        dets = load_json(detetion_path)

        self.det_keys = []
        self.dets = {}
        for det in tqdm(dets, "processing detection results"):
            scene_id = det["scene_id"]  # data type: int
            img_id = det["image_id"]  # data type: int
            key = str(scene_id).zfill(6) + "_" + str(img_id).zfill(6)  # {scene_id}_{image_id}
            if key not in self.det_keys:
                self.det_keys.append(key)
                self.dets[key] = []
            self.dets[key].append(det)
        del dets
        logger.info("testing on {} images on {}...".format(len(self.det_keys), eval_dataset_name))

    def __len__(self):
        return len(self.det_keys)

    def __getitem__(self, index):
        dets = self.dets[self.det_keys[index]]

        instances = []
        inst_ids = []
        for det_i, det in enumerate(dets):
            if det["score"] > self.seg_filter_score:
                instance = self.get_instance(det)
                if instance is not None:
                    instances.append(instance)
                    inst_ids.append(det_i)

        # ensure one instance at least if no object is left after score filtering
        if len(instances) == 0:
            scores = [det["score"] for det in dets]
            max_score = max(scores)
            max_score_ind = scores.index(max_score)
            instance = self.get_instance(dets[max_score_ind])
            if instance is not None:
                instances.append(instance)
                inst_ids.append(max_score_ind)
            else:
                raise ValueError(f"no qulified instance in {self.det_keys[index]}")

        ret_dict = {}
        for key in instances[0].keys():
            if "pcd" not in key:
                # NOTE: use this
                ret_dict[key] = torch.stack([instance[key] for instance in instances])
            else:
                ret_dict[key] = [instance[key] for instance in instances]
        ret_dict["scene_id"] = torch.IntTensor([int(self.det_keys[index][0:6])])
        ret_dict["img_id"] = torch.IntTensor([int(self.det_keys[index][7:13])])
        ret_dict["inst_ids"] = torch.IntTensor(inst_ids)
        ret_dict["seg_time"] = torch.FloatTensor([dets[0]["time"]])
        return ret_dict

    def get_instance(self, data):
        scene_id = data["scene_id"]  # data type: int
        img_id = data["image_id"]  # data type: int
        obj_id = data["category_id"]  # data type: int  NOTE: the obj_id in dataset
        bbox = data["bbox"]  # list, len:4
        seg = data["segmentation"]  # keys: counts, size
        score = data["score"]

        # load target infos
        scene_folder = osp.join(self.data_folder, f"{scene_id:06d}")
        scene_camera = load_json(osp.join(scene_folder, "scene_camera.json"))
        K = np.array(scene_camera[str(img_id)]["cam_K"]).reshape((3, 3)).copy()
        depth_scale = scene_camera[str(img_id)]["depth_scale"]
        inst = dict(scene_id=scene_id, img_id=img_id, data_folder=self.data_folder)

        obj_idx = self.obj_idxs[obj_id]  # 0-based id

        # fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # depth
        depth = get_bop_depth_map(inst) * depth_scale

        # mask
        h, w = seg["size"]
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, depth > 0)  # filter mask by valid depth
        if np.sum(mask) > self.minimum_n_point:  # 8
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            return None
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = backproject(depth, K, [y1, y2, x1, x2])
        cloud = cloud.reshape(-1, 3)[choose, :]
        # TODO: maybe use median?
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]

        tem_rgb, tem_choose, tem_pts, pose_camref_obj = self._get_ref_instance(scene_id, img_id, obj_id)
        if tem_rgb is None:
            return None

        # minus mean and filter by radius
        tem_mean_point = np.mean(tem_pts, axis=0)
        tem_pts_minus_mean = tem_pts - tem_mean_point.reshape(1, 3)
        # may be inaccurate due to noise
        radius = np.max(np.linalg.norm(tem_pts_minus_mean, axis=1))
        flag = np.linalg.norm(tmp_cloud, axis=1) < 1.2 * radius

        if np.sum(flag) < self.minimum_n_point:
            return None
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= self.n_sample_observed_point:  # 2048
            choose_idx = np.random.choice(np.arange(len(choose)), size=self.n_sample_observed_point, replace=True)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), size=self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb: crop --> mask (True) -> resize
        rgb = get_bop_image(
            inst, [y1, y2, x1, x2], self.img_size, mask if self.rgb_mask_flag else None, rgb_to_bgr=self.rgb_to_bgr
        )
        rgb = self.transform(np.array(rgb))  # [0,1] -> normalize
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        ret_dict = {}
        ret_dict["pts"] = torch.FloatTensor(cloud)
        ret_dict["rgb"] = torch.FloatTensor(rgb)
        ret_dict["rgb_choose"] = torch.IntTensor(rgb_choose).long()
        ret_dict["obj"] = torch.IntTensor([obj_idx]).long()  # 0-based id

        ret_dict["obj_id"] = torch.IntTensor([obj_id])  # real obj_id in dataset
        ret_dict["score"] = torch.FloatTensor([score])

        # tem1
        ret_dict["tem1_rgb"] = torch.FloatTensor(tem_rgb)
        ret_dict["tem1_choose"] = torch.IntTensor(tem_choose).long()
        ret_dict["tem1_pts"] = torch.FloatTensor(tem_pts)
        ret_dict["tem1_pose"] = torch.FloatTensor(pose_camref_obj)
        return ret_dict

    def _get_ref_instance(self, scene_id, img_id, obj_id):
        # load ref infos
        key = f"{scene_id}_{img_id}_{obj_id}"
        if key not in self.test_ref_target:
            return None, None, None, None

        ref_scene_id, ref_im_id = self.test_ref_target[key].split("_")
        ref_scene_id = int(ref_scene_id)
        ref_im_id = int(ref_im_id)

        if self.dataset == "ycbv":
            test_scene_ids = list(range(48, 60))
            if ref_scene_id not in test_scene_ids:
                data_folder = osp.join(self.data_dir, self.dataset, "train_real")
            else:
                data_folder = self.data_folder
        elif self.dataset == "tudl":
            # ref only in train
            data_folder = osp.join(self.data_dir, self.dataset, "train_real")
        else:
            data_folder = self.data_folder

        scene_folder = osp.join(data_folder, f"{ref_scene_id:06d}")
        scene_camera = load_json(osp.join(scene_folder, "scene_camera.json"))
        K = np.array(scene_camera[str(ref_im_id)]["cam_K"]).reshape((3, 3)).copy()

        # load ref pose
        scene_gt = load_json(osp.join(scene_folder, "scene_gt.json"))
        ref_gt_infos = scene_gt[str(ref_im_id)]
        found = False
        for i, ref_gt_info in enumerate(ref_gt_infos):
            if ref_gt_info["obj_id"] == obj_id:
                ref_mask_path = osp.join(data_folder, f"{ref_scene_id:06d}/mask_visib/{ref_im_id:06d}_{i:06d}.png")

                ref_rot = np.array(ref_gt_info["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                ref_trans = np.array(ref_gt_info["cam_t_m2c"], dtype=np.float32).reshape(3) * 0.001
                pose_camref_obj = np.eye(4, dtype=np.float32)
                pose_camref_obj[:3, :3] = ref_rot
                pose_camref_obj[:3, 3] = ref_trans
                found = True
                break
        if not found:
            return None, None, None, None
        # fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # depth
        depth_scale = scene_camera[str(ref_im_id)]["depth_scale"]
        inst = dict(scene_id=ref_scene_id, img_id=ref_im_id, data_folder=data_folder)
        depth = (get_bop_depth_map(inst) * depth_scale).astype("float32")
        # h, w = depth.shape

        # mask
        mask = np.array(imageio.imread(ref_mask_path)).astype(bool)  # 0, 1

        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]  # cropped mask

        # ref xyz in camera space
        ref_xyz = backproject(depth, K, bbox)
        ref_xyz *= mask.astype("float32")[:, :, None]

        # load rgb crop, resized
        # rgb: crop --> mask (True) -> resize
        ref_rgb = get_bop_image(
            inst, [y1, y2, x1, x2], self.img_size, mask if self.rgb_mask_flag else None, rgb_to_bgr=self.rgb_to_bgr
        )
        ref_rgb = self.transform(np.array(ref_rgb))  # [0,1] -> normalize

        ref_choose = (mask > 0).astype(np.float32).flatten().nonzero()[0]
        if len(ref_choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(ref_choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(ref_choose)), self.n_sample_template_point, replace=False)
        ref_choose = ref_choose[choose_idx]

        ref_xyz = ref_xyz.reshape(-1, 3)[ref_choose, :]  # num_choose x 3

        # ref_xyz, fps_choose = farthest_point_sampling(ref_xyz, self.n_sample_observed_point, init_center=True)
        # ref_choose = ref_choose[fps_choose]

        ref_rgb_choose = get_resize_rgb_choose(ref_choose, [y1, y2, x1, x2], self.img_size)
        return ref_rgb, ref_rgb_choose, ref_xyz, pose_camref_obj

    def load_ref(self, test_ref_target_path):
        """
        Load all test ref targets from json
        """
        # keys: scene_id, image_id, category_id, bbox, score, segmentation
        test_ref_target_list = load_json(test_ref_target_path)

        test_ref_target = {}
        for ref_target in test_ref_target_list:
            scene_id = ref_target["scene_id"]
            img_id = ref_target["im_id"]
            obj_id = ref_target["obj_id"]
            ref_scene_id = ref_target["ref_scene_id"]
            ref_im_id = ref_target["ref_im_id"]

            test_ref_target[f"{scene_id}_{img_id}_{obj_id}"] = f"{ref_scene_id}_{ref_im_id}"

        return test_ref_target

    def load_single_ref_per_dset(self, test_target_path, ref_scene_ims):
        # keys: scene_id, image_id, category_id, bbox, score, segmentation
        test_target_list = load_json(test_target_path)

        test_ref_target = {}
        assert len(ref_scene_ims) == len(self.obj_idxs)
        ref_dict = {}
        for idx, ref_scene_im in enumerate(ref_scene_ims):
            ref_scene, ref_im = ref_scene_im.split("_")
            ref_scene = int(ref_scene)
            ref_im = int(ref_im)
            ref_dict[idx + 1] = (ref_scene, ref_im)

        for target in test_target_list:
            obj_id = target["obj_id"]
            scene_id = target["scene_id"]
            img_id = target["im_id"]
            ref_scene_id, ref_im_id = ref_dict[obj_id]

            test_ref_target[f"{scene_id}_{img_id}_{obj_id}"] = f"{ref_scene_id}_{ref_im_id}"

        return test_ref_target


if __name__ == "__main__":
    from easydict import EasyDict as edict
    from pathlib import Path
    from core.unopose.utils.vis_utils import plot_3d

    np.random.seed(1)

    PROJ_ROOT = Path(__file__).parent.parent.parent.parent
    print(PROJ_ROOT)

    cfg = edict(
        data_dir=osp.join(PROJ_ROOT, "datasets/BOP_DATASETS"),
        ref_targets_name="test_ref_targets_crossscene_rot50.json",
        img_size=224,
        n_sample_observed_point=2048,
        n_sample_model_point=1024,
        n_sample_template_point=5000,
        minimum_n_point=8,
        rgb_mask_flag=True,
        seg_filter_score=0.25,
        rgb_to_bgr=False,
        img_H=480,
        img_W=640,
    )
    detetion_path = (
        PROJ_ROOT
        / "datasets/segmentation/CustomSamAutomaticMaskGenerator_test_oneref_targets_crossscene_rot50_refvisib_ycbv.json"
    )
    dataset = BOPTestsetPoseFreeOneRefv2(cfg, eval_dataset_name="ycbv", detetion_path=detetion_path)

    for data in dataset:
        pts = data["pts"][0]
        pts_tem1 = data["tem1_pts"][0]
