from pathlib import Path
import os
import logging
import os.path as osp
import numpy as np
import orjson as json
import warnings
import imageio
import cv2
import mmcv

# from detectron2.utils.logger import log_first_n
# import open3d
import torch


def _empty(_n, _c, _h, _w, dtype=torch.float32, device="cuda"):
    _tensor_kwargs = {"dtype": dtype, "device": device}
    return torch.empty(_n, _c, _h, _w, **_tensor_kwargs).detach()


def denormalize_image(image, PIXEL_MEAN, PIXEL_STD):
    # CHW
    pixel_mean = np.array(PIXEL_MEAN).reshape(-1, 1, 1)
    pixel_std = np.array(PIXEL_STD).reshape(-1, 1, 1)
    return image * pixel_std + pixel_mean


def load_im(path, format="RGB"):
    """Loads an image from a file.

    :param path: Path to the image file to load.
    :return: ndarray with the loaded image.
    """
    im = imageio.imread(path)
    # im = read_image_mmcv(path, format=format)
    return im


def load_json(json_path):
    return json.loads(Path(json_path).read_bytes())


def read_image_mmcv(file_name, format=None):
    """# NOTE modified from detectron2, use mmcv instead of PIL to read an
    image into the given format.

    Args:
        file_name (str): image file path
        format (str): "BGR" | "RGB" | "L" | "unchanged"
    Returns:
        image (np.ndarray): an HWC image
    """
    flag = "color"
    channel_order = "bgr"
    if format == "RGB":
        channel_order = "rgb"
    elif format == "L":
        flag = "grayscale"
    elif format == "unchanged":
        flag = "unchanged"
    else:
        if format not in [None, "BGR"]:
            raise ValueError(f"Invalid format: {format}")

    # NOTE: use turbojpeg if available
    # TODO: check if tiffile is faster for .tif
    ext = osp.splitext(file_name)[-1]
    if ext.lower() in [".jpg", ".jpeg"]:
        backend = "turbojpeg"
        mmcv.use_backend(backend)
    elif ext.lower() in [".tif"]:
        # backend = "tifffile"
        backend = "imageio"  # imageio is slightly faster in reading tif
    else:
        backend = "cv2"
    if backend == "imageio":
        image = imageio.imread(file_name)
    else:
        image = mmcv.imread(file_name, flag, channel_order, backend=backend)
    return image


def io_load_gt(
    gt_file,
    instance_ids=None,
):
    """Load ground truth from an I/O object.
    Instance_ids can be specified to load only a
    subset of object instances.

    :param gt_file: I/O object that can be read with json
    :param instance_ids: List of instance ids.
    :return: List of ground truth annotations (one dict per object instance).
    """
    gt = json.loads(Path(gt_file).read_bytes())
    if instance_ids is not None:
        gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
    gt = [_gt_as_numpy(gt_n) for gt_n in gt]
    return gt


def load_gt_from_bytes(
    gt_bytes,
    instance_ids=None,
):
    """Load ground truth from an I/O object.
    Instance_ids can be specified to load only a
    subset of object instances.

    :param gt_file: I/O object that can be read with json
    :param instance_ids: List of instance ids.
    :return: List of ground truth annotations (one dict per object instance).
    """
    gt = json.loads(gt_bytes)
    if instance_ids is not None:
        gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
    gt = [_gt_as_numpy(gt_n) for gt_n in gt]
    return gt


def io_load_masks(mask_file, instance_ids=None):
    """Load object masks from an I/O object.
    Instance_ids can be specified to apply RLE
    decoding to a subset of object instances contained
    in the file.

    :param mask_file: I/O object that can be read with json.load.
    :param masks_path: Path to json file.
    :return: a [N,H,W] binary array containing object masks.
    """
    masks_rle = json.loads(Path(mask_file).read_bytes())
    masks_rle = {int(k): v for k, v in masks_rle.items()}
    if instance_ids is None:
        instance_ids = masks_rle.keys()
        instance_ids = sorted(instance_ids)
    masks = np.stack([rle_to_binary_mask(masks_rle[instance_id]) for instance_id in instance_ids])
    return masks


def load_masks_from_bytes(mask_bytes, instance_ids=None):
    """Load object masks from an I/O object.
    Instance_ids can be specified to apply RLE
    decoding to a subset of object instances contained
    in the file.

    :param mask_file: I/O object that can be read with json.load.
    :param masks_path: Path to json file.
    :return: a [N,H,W] binary array containing object masks.
    """
    masks_rle = json.loads(mask_bytes)
    masks_rle = {int(k): v for k, v in masks_rle.items()}
    if instance_ids is None:
        instance_ids = masks_rle.keys()
        instance_ids = sorted(instance_ids)
    masks = np.stack([rle_to_binary_mask(masks_rle[instance_id]) for instance_id in instance_ids])
    return masks


def _gt_as_numpy(gt):
    if "cam_R_m2c" in gt.keys():
        gt["cam_R_m2c"] = np.array(gt["cam_R_m2c"], np.float32).reshape((3, 3))
    if "cam_t_m2c" in gt.keys():
        gt["cam_t_m2c"] = np.array(gt["cam_t_m2c"], np.float32).reshape((3, 1))
    return gt


def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get("size")), dtype=bool)
    counts = rle.get("counts")

    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get("size"), order="F")

    return binary_mask


# slow!!
# def get_point_cloud_from_depth(depth, K, bbox=None):
#     """
#     Args:
#         bbox: y1, y2, x1, x2
#     """
#     warnings.warn("this is slow! use backproject instead!")
#     cam_fx, cam_fy, cam_cx, cam_cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

#     im_H, im_W = depth.shape
#     xmap = np.array([[i for i in range(im_W)] for j in range(im_H)], dtype=np.float32)
#     ymap = np.array([[j for i in range(im_W)] for j in range(im_H)], dtype=np.float32)

#     if bbox is not None:
#         rmin, rmax, cmin, cmax = bbox
#         depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
#         xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
#         ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

#     pt2 = depth.astype(np.float32)
#     pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
#     pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy
#     # XYZ: 3HW --> HW3
#     # cloud = np.stack([pt0, pt1, pt2]).transpose((1, 2, 0))
#     cloud = np.stack([pt0, pt1, pt2], axis=-1)
#     return cloud


def backproject(depth, K, bbox=None):
    """Backproject a depth map to a cloud map  (faster implementation)
    depth:  depth
    bbox: y1, y2, x1, x2
    ----
    organized cloud map: (H,W,3)
    """
    H, W = depth.shape
    X, Y = np.meshgrid(np.asarray(range(W)) - K[0, 2], np.asarray(range(H)) - K[1, 2])
    cloud = np.stack((X * depth / K[0, 0], Y * depth / K[1, 1], depth), axis=2)
    if bbox is not None:
        rmin, rmax, cmin, cmax = bbox
        return cloud[rmin:rmax, cmin:cmax]
    return cloud


def get_resize_rgb_choose(choose, bbox, img_size):
    """
    Args:
        bbox: y1, y2, x1, x2
    """
    rmin, rmax, cmin, cmax = bbox
    crop_h = rmax - rmin
    ratio_h = img_size / crop_h
    crop_w = cmax - cmin
    ratio_w = img_size / crop_w

    row_idx = choose // crop_h
    col_idx = choose % crop_h
    choose = (np.floor(row_idx * ratio_w) * img_size + np.floor(col_idx * ratio_h)).astype(np.int64)
    return choose


def get_bbox(label):
    img_width, img_length = label.shape
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    b = min(max(r_b, c_b), min(img_width, img_length))
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

    rmin = center[0] - int(b / 2)
    rmax = center[0] + int(b / 2)
    cmin = center[1] - int(b / 2)
    cmax = center[1] + int(b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return [rmin, rmax, cmin, cmax]  # y1, y2, x1, x2


def get_random_rotation():
    """
    3 random euler angles [0, 2pi]
    """
    angles = np.random.rand(3) * 2 * np.pi
    rand_rotation = (
        np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
        @ np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        @ np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
    )
    return rand_rotation


def get_random_rotation_th():
    """
    3 random euler angles [0, 2pi]
    """
    angles = torch.rand(3) * 2 * torch.pi
    rand_rotation = (
        torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                [0, torch.sin(angles[0]), torch.cos(angles[0])],
            ]
        )
        @ torch.tensor(
            [
                [torch.cos(angles[1]), 0, torch.sin(angles[1])],
                [0, 1, 0],
                [-torch.sin(angles[1]), 0, torch.cos(angles[1])],
            ]
        )
        @ torch.tensor(
            [
                [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
    )
    return rand_rotation


def get_model_info(obj, return_color=False, sample_num=2048):
    if return_color:
        model_points, model_color, symmetry_flag = obj.get_item(return_color, sample_num)
        return (model_points, model_color, symmetry_flag)
    else:
        model_points, symmetry_flag = obj.get_item()
        return (model_points, symmetry_flag)


def get_bop_depth_map(inst):
    """
    the returned depth's unit is m
    """
    scene_id, img_id, data_folder = inst["scene_id"], inst["img_id"], inst["data_folder"]
    try:
        depth = (
            load_im(os.path.join(data_folder, f"{scene_id:06d}", "depth", f"{img_id:06d}.png"), "unchanged") / 1000.0
        )
    except:
        # imageio is slightly faster than mmcv tifftile backend
        depth = imageio.imread(os.path.join(data_folder, f"{scene_id:06d}", "depth", f"{img_id:06d}.tif")) / 1000.0
    return depth


def backproject_to_obj(depth, R, T, K, bbox=None):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3) in obj space
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = np.einsum(
        "ijkl,ijlm->ijkm",
        R.T.reshape(1, 1, 3, 3),
        depth.reshape(height, width, 1, 1)
        * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
        - T.reshape(1, 1, 3, 1),
    ).squeeze() * mask.reshape(height, width, 1)
    if bbox is not None:
        rmin, rmax, cmin, cmax = bbox
        return ProjEmb[rmin:rmax, cmin:cmax]
    return ProjEmb


def get_zoe_depth_map(inst):
    """
    the returned depth's unit is m
    """
    scene_id, img_id, data_folder = inst["scene_id"], inst["img_id"], inst["data_folder"]
    depth = (
        load_im(os.path.join(data_folder, f"{scene_id:06d}", "zoe_depth", f"{img_id:06d}.png"), "unchanged") / 1000.0
    )
    return depth


def get_depthanythingv2_map(inst):
    """
    the returned depth's unit is m
    """
    scene_id, img_id, data_folder = inst["scene_id"], inst["img_id"], inst["data_folder"]
    depth = (
        load_im(os.path.join(data_folder, f"{scene_id:06d}", "depth_anything_v2", f"{img_id:06d}.png"), "unchanged")
        / 1000.0
    )
    return depth


def get_bop_image(inst, bbox, img_size, mask=None, rgb_to_bgr=False):
    """
    mask is the cropped mask
    """
    scene_id, img_id, data_folder = inst["scene_id"], inst["img_id"], inst["data_folder"]
    rmin, rmax, cmin, cmax = bbox
    img_path = os.path.join(data_folder, f"{scene_id:06d}/")

    strs = [f"rgb/{img_id:06d}.jpg", f"rgb/{img_id:06d}.png", f"gray/{img_id:06d}.tif"]
    for s in strs:
        if os.path.exists(os.path.join(img_path, s)):
            img_path = os.path.join(img_path, s)
            break

    rgb = load_im(img_path).astype(np.uint8)
    if len(rgb.shape) == 2:
        rgb = np.concatenate([rgb[:, :, None], rgb[:, :, None], rgb[:, :, None]], axis=2)
    if rgb_to_bgr:
        rgb = rgb[..., ::-1][rmin:rmax, cmin:cmax, :3]
        # log_first_n(logging.WARNING, "sam6d's setting, rgb to bgr, this is wrong!")
        print("sam6d's setting, rgb to bgr, this is wrong!")
    else:
        rgb = rgb[rmin:rmax, cmin:cmax, :3]
    if mask is not None:
        rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return rgb


def get_bop_image_full(inst, mask=None):
    scene_id, img_id, data_folder = inst["scene_id"], inst["img_id"], inst["data_folder"]
    img_path = os.path.join(data_folder, f"{scene_id:06d}/")

    strs = [f"rgb/{img_id:06d}.jpg", f"rgb/{img_id:06d}.png", f"gray/{img_id:06d}.tif"]
    for s in strs:
        if os.path.exists(os.path.join(img_path, s)):
            img_path = os.path.join(img_path, s)
            break

    rgb = load_im(img_path).astype(np.uint8)
    if len(rgb.shape) == 2:
        rgb = np.concatenate([rgb[:, :, None], rgb[:, :, None], rgb[:, :, None]], axis=2)
    if mask is not None:
        rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    return rgb


def to_o3d_pcd(pts):
    """
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    return pcd


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    """
    Give source & target point clouds as well as the relative transformation between them, calculate correspondences according to give threshold
    :param src_pcd: source point cloud
    :param tgt_pcd: target point cloud
    :param trans: relative transformation between source and target point clouds
    :param search_voxel_size: given threshold
    :param K: if k is not none, select top k nearest neighbors from candidate set after radius search
    :return: (m, 2) torch tensor, consisting of m correspondences
    """
    src_pcd.transform(trans)
    pcd_tree = open3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    return correspondences


def square_distance(src, tgt, normalize=False):
    """
    Calculate Euclide distance between every two points
    :param src: source point cloud in shape [B, N, C]
    :param tgt: target point cloud in shape [B, M, C]
    :param normalize: whether to normalize calculated distances
    :return:
    """

    B, N, _ = src.shape
    _, M, _ = tgt.shape
    dist = -2.0 * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    if normalize:
        dist += 2
    else:
        dist += torch.sum(src**2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt**2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def point2node(nodes, points):
    """
    Assign each point to a certain node according to nearest neighbor search
    :param nodes: [M, 3]
    :param points: [N, 3]
    :return: idx [N], indicating the id of node that each point belongs to
    """
    M, _ = nodes.size()
    N, _ = points.size()
    dist = square_distance(points.unsqueeze(0), nodes.unsqueeze(0))[0]

    idx = dist.topk(k=1, dim=-1, largest=False)[1]  # [B, N, 1], ignore the smallest element as it's the query itself

    idx = idx.squeeze(-1)
    return idx


def point2node_correspondences(
    src_nodes,
    src_points,
    tgt_nodes,
    tgt_points,
    point_correspondences,
):
    """
    Based on point correspondences & point2node relationships, calculate node correspondences
    :param src_nodes: Nodes of source point cloud
    :param src_points: Points of source point cloud
    :param tgt_nodes: Nodes of target point cloud
    :param tgt_points: Points of target point cloud
    :param point_correspondences: Ground truth point correspondences
    :return: node_corr_mask: Overlap ratios between nodes
             node_corr: Node correspondences sampled for training
    """
    #####################################
    # calc visible ratio for each node
    src_visible, tgt_visible = point_correspondences[:, 0], point_correspondences[:, 1]

    src_vis, tgt_vis = torch.zeros((src_points.shape[0])), torch.zeros((tgt_points.shape[0]))

    src_vis[src_visible] = 1.0
    tgt_vis[tgt_visible] = 1.0

    src_vis = src_vis.nonzero().squeeze(1)
    tgt_vis = tgt_vis.nonzero().squeeze(1)

    src_vis_num = torch.zeros((src_nodes.shape[0]))
    src_tot_num = torch.ones((src_nodes.shape[0]))

    src_idx = point2node(src_nodes, src_points)
    idx, cts = torch.unique(src_idx, return_counts=True)
    src_tot_num[idx] = cts.float()

    src_idx_ = src_idx[src_vis]
    idx_, cts_ = torch.unique(src_idx_, return_counts=True)
    src_vis_num[idx_] = cts_.float()

    src_node_vis = src_vis_num / src_tot_num

    tgt_vis_num = torch.zeros((tgt_nodes.shape[0]))
    tgt_tot_num = torch.ones((tgt_nodes.shape[0]))

    tgt_idx = point2node(tgt_nodes, tgt_points)
    idx, cts = torch.unique(tgt_idx, return_counts=True)
    tgt_tot_num[idx] = cts.float()

    tgt_idx_ = tgt_idx[tgt_vis]
    idx_, cts_ = torch.unique(tgt_idx_, return_counts=True)
    tgt_vis_num[idx_] = cts_.float()

    tgt_node_vis = tgt_vis_num / tgt_tot_num

    src_corr = point_correspondences[:, 0]  # [K]
    tgt_corr = point_correspondences[:, 1]  # [K]

    src_node_corr = torch.gather(src_idx, 0, src_corr)
    tgt_node_corr = torch.gather(tgt_idx, 0, tgt_corr)

    index = src_node_corr * tgt_idx.shape[0] + tgt_node_corr

    index, counts = torch.unique(index, return_counts=True)

    src_node_corr = index // tgt_idx.shape[0]
    tgt_node_corr = index % tgt_idx.shape[0]

    node_correspondences = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32)

    # node_corr_mask = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32)
    node_correspondences[src_node_corr, tgt_node_corr] = counts.float()
    node_correspondences = node_correspondences[:-1, :-1]

    node_corr_sum_row = torch.sum(node_correspondences, dim=1, keepdim=True)
    node_corr_sum_col = torch.sum(node_correspondences, dim=0, keepdim=True)

    node_corr_norm_row = (node_correspondences / (node_corr_sum_row + 1e-10)) * src_node_vis.unsqueeze(1).expand(
        src_nodes.shape[0], tgt_nodes.shape[0]
    )
    node_corr_norm_col = (node_correspondences / (node_corr_sum_col + 1e-10)) * tgt_node_vis.unsqueeze(0).expand(
        src_nodes.shape[0], tgt_nodes.shape[0]
    )

    # both src -> tgt
    node_corr_norm_row = torch.cat(((1 - src_node_vis)[:, None], node_corr_norm_row), dim=1)  # [196, 197]
    node_corr_norm_col = torch.cat(((1 - tgt_node_vis)[None, :], node_corr_norm_col), dim=0)  # [197, 196]

    node_corr_norm_row = node_corr_norm_row.T.contiguous()  # [197,196]
    node_corr_norm_col = node_corr_norm_col.contiguous()  # [197,196]
    return node_corr_norm_row, node_corr_norm_col


def point2node_correspondences_cofinet(
    src_nodes, src_points, tgt_nodes, tgt_points, point_correspondences, thres=0.0, device="cpu"
):
    """
    Based on point correspondences & point2node relationships, calculate node correspondences
    :param src_nodes: Nodes of source point cloud
    :param src_points: Points of source point cloud
    :param tgt_nodes: Nodes of target point cloud
    :param tgt_points: Points of target point cloud
    :param point_correspondences: Ground truth point correspondences
    :return: node_corr_mask: Overlap ratios between nodes
             node_corr: Node correspondences sampled for training
    """
    #####################################
    # calc visible ratio for each node
    src_visible, tgt_visible = point_correspondences[:, 0], point_correspondences[:, 1]

    src_vis, tgt_vis = torch.zeros((src_points.shape[0])).to(device), torch.zeros((tgt_points.shape[0])).to(device)

    src_vis[src_visible] = 1.0
    tgt_vis[tgt_visible] = 1.0

    src_vis = src_vis.nonzero().squeeze(1)
    tgt_vis = tgt_vis.nonzero().squeeze(1)

    src_vis_num = torch.zeros((src_nodes.shape[0])).to(device)
    src_tot_num = torch.ones((src_nodes.shape[0])).to(device)

    src_idx = point2node(src_nodes, src_points)
    idx, cts = torch.unique(src_idx, return_counts=True)
    src_tot_num[idx] = cts.float()

    src_idx_ = src_idx[src_vis]
    idx_, cts_ = torch.unique(src_idx_, return_counts=True)
    src_vis_num[idx_] = cts_.float()

    src_node_vis = src_vis_num / src_tot_num

    tgt_vis_num = torch.zeros((tgt_nodes.shape[0])).to(device)
    tgt_tot_num = torch.ones((tgt_nodes.shape[0])).to(device)

    tgt_idx = point2node(tgt_nodes, tgt_points)
    idx, cts = torch.unique(tgt_idx, return_counts=True)
    tgt_tot_num[idx] = cts.float()

    tgt_idx_ = tgt_idx[tgt_vis]
    idx_, cts_ = torch.unique(tgt_idx_, return_counts=True)
    tgt_vis_num[idx_] = cts_.float()

    tgt_node_vis = tgt_vis_num / tgt_tot_num

    src_corr = point_correspondences[:, 0]  # [K]
    tgt_corr = point_correspondences[:, 1]  # [K]

    src_node_corr = torch.gather(src_idx, 0, src_corr)
    tgt_node_corr = torch.gather(tgt_idx, 0, tgt_corr)

    index = src_node_corr * tgt_idx.shape[0] + tgt_node_corr

    index, counts = torch.unique(index, return_counts=True)

    src_node_corr = index // tgt_idx.shape[0]
    tgt_node_corr = index % tgt_idx.shape[0]

    node_correspondences = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(
        device
    )

    node_corr_mask = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(device)
    node_correspondences[src_node_corr, tgt_node_corr] = counts.float()
    node_correspondences = node_correspondences[:-1, :-1]

    node_corr_sum_row = torch.sum(node_correspondences, dim=1, keepdim=True)
    node_corr_sum_col = torch.sum(node_correspondences, dim=0, keepdim=True)

    node_corr_norm_row = (node_correspondences / (node_corr_sum_row + 1e-10)) * src_node_vis.unsqueeze(1).expand(
        src_nodes.shape[0], tgt_nodes.shape[0]
    )

    node_corr_norm_col = (node_correspondences / (node_corr_sum_col + 1e-10)) * tgt_node_vis.unsqueeze(0).expand(
        src_nodes.shape[0], tgt_nodes.shape[0]
    )

    node_corr_mask[:-1, :-1] = 0.5 * (node_corr_norm_row + node_corr_norm_col)
    ############################################################
    # Binary masks
    # node_corr_mask[:-1, :-1] = (node_corr_mask[:-1, :-1] > 0.01)
    # node_corr_mask[-1, :-1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=0), min=0.)
    # node_corr_mask[:-1, -1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=1), min=0.)
    re_mask = node_corr_mask[:-1, :-1] > thres
    node_corr_mask[:-1, :-1] = node_corr_mask[:-1, :-1] * re_mask

    #####################################################
    # Soft weighted mask, best Performance
    node_corr_mask[:-1, -1] = 1.0 - src_node_vis
    node_corr_mask[-1, :-1] = 1.0 - tgt_node_vis
    #####################################################

    node_corr = node_corr_mask[:-1, :-1].nonzero()
    return node_corr_mask, node_corr


def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x**2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y**2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


def point_to_node_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    Fixed knn bug.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool)  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0])  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long)  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


def get_node_correspondences(
    ref_nodes: torch.Tensor,
    src_nodes: torch.Tensor,
    ref_knn_points: torch.Tensor,
    src_knn_points: torch.Tensor,
    transform: torch.Tensor,
    pos_radius: float,
    ref_masks=None,
    src_masks=None,
    ref_knn_masks=None,
    src_knn_masks=None,
):
    r"""Generate ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        ref_nodes: torch.Tensor (M, 3)
        src_nodes: torch.Tensor (N, 3)
        ref_knn_points: torch.Tensor (M, K, 3)
        src_knn_points: torch.Tensor (N, K, 3)
        transform: torch.Tensor (4, 4)
        pos_radius: float
        ref_masks (optional): torch.BoolTensor (M,) (default: None)
        src_masks (optional): torch.BoolTensor (N,) (default: None)
        ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
        src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)

    Returns:
        corr_indices: torch.LongTensor (C, 2)
        corr_overlaps: torch.Tensor (C,)
    """
    src_nodes = (src_nodes - transform[:3, 3].reshape(1, 3)) @ transform[:3, :3]
    src_knn_points = (src_knn_points - transform[:3, 3].reshape(1, 3)) @ transform[:3, :3]

    # generate masks
    if ref_masks is None:
        ref_masks = torch.ones(size=(ref_nodes.shape[0],), dtype=torch.bool)
    if src_masks is None:
        src_masks = torch.ones(size=(src_nodes.shape[0],), dtype=torch.bool)
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(size=(ref_knn_points.shape[0], ref_knn_points.shape[1]), dtype=torch.bool)
    if src_knn_masks is None:
        src_knn_masks = torch.ones(size=(src_knn_points.shape[0], src_knn_points.shape[1]), dtype=torch.bool)

    node_mask_mat = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))  # (M, N)

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]  # (M,)
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    intersect_mat = torch.gt(ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + pos_radius - dist_mat, 0)
    intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
    sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)

    # select potential patch pairs
    ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (B, K)
    src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
    ref_knn_points = ref_knn_points[sel_ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)

    point_mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    dist_mat = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)
    point_overlap_mat = torch.lt(dist_mat, pos_radius**2)  # (B, K, K)
    ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()  # (B,)
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()  # (B,)
    ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()  # (B,)
    src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    overlap_masks = torch.gt(overlaps, 0)
    ref_corr_indices = sel_ref_indices[overlap_masks]
    src_corr_indices = sel_src_indices[overlap_masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[overlap_masks]

    return corr_indices, corr_overlaps


def vis_mutual_correspondence(
    src_points,
    tgt_points,
    node_corr_norm_row,
    node_corr_norm_col,
    show_limit=300,
):
    # note: maybe sample
    from core.unopose.tests.test_node_correspondence_overlap import vis_correspondence

    matching_mask_row = node_corr_norm_row[1:].T  # (196,196)
    matching_mask_col = node_corr_norm_col[1:]  # (196,196)

    node_corr_row = matching_mask_row.nonzero()
    node_corr_col = matching_mask_col.nonzero()

    vis_correspondence(src_points, tgt_points, node_corr_row, show_limit=show_limit, matching_mask=matching_mask_row)
    vis_correspondence(src_points, tgt_points, node_corr_col, show_limit=show_limit, matching_mask=matching_mask_col)
