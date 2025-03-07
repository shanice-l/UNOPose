import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.cuda.amp import autocast
from einops import rearrange

from core.unopose.model.pointnet2.pointnet2_utils import (
    gather_operation,
    furthest_point_sample,
)

logger = logging.getLogger(__name__)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C H W
        u = x.mean(1, keepdim=True)  # B 1 H W
        s = (x - u).pow(2).mean(1, keepdim=True)  # B 1 H W
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MatchLayerNorm(nn.Module):
    """2x slow"""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # (C,)
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B P C
        with autocast(enabled=False, dtype=torch.float32):
            x = x.float()
            if self.scale is None:
                u = x.mean(2, keepdim=True)  # B P 1
                s = (x - u).pow(2).mean(2, keepdim=True)  # B P 1
                self.scale = (s + self.eps).sqrt()
                x = (x - u) / self.scale
            else:
                u = x.mean(2, keepdim=True)  # B P 1
                x = (x - u) / self.scale
                self.scale = None

            x = self.weight[None, :].float() * x + self.bias[None, :].float()
            return x


class MatchNorm(nn.Module):
    def __init__(self, num_channels=None, with_param=True, eps=1e-6):
        super().__init__()
        self.scale = None
        self.eps = eps
        self.with_param = with_param
        if with_param:
            self.weight = nn.Parameter(torch.ones(num_channels))  # (C,)
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, v_min=-1, v_max=1):
        # x: b p c
        with autocast(enabled=False, dtype=torch.float32):
            x = x.float()
            reshape = False
            if x.dim() == 4:
                x = x.contiguous()
                reshape = True
                x = rearrange(x, "b p g c -> b (p g) c")
                # bs, c, num_pt, num_gr = x.shape
                # x = x.view([bs, -1, num_pt])

            if self.scale is None:
                max, _ = x.max(1)  # b, c
                min, _ = x.min(1)  # b, c
                self.scale, _ = torch.max(torch.maximum(max, torch.abs(min)), 1)  # b
                self.scale = self.scale[:, None, None]  # to be shared by another point cloud
                # intv, _ = (max - min).max(1)
                # x = (x - max[:, :, None] - min[:, :, None])/intv[:, None, None]*(v_max - v_min) + v_min
                x /= self.scale + self.eps
            else:
                x /= self.scale + self.eps
                self.scale = None
                # from bpnet_utils.visual import plot_point_cloud
                # plot_point_cloud(x[2].cpu().numpy().T, x[2].cpu().numpy().T, np.eye(4))
                # import ipdb; ipdb.set_trace()
            x = x - x.mean(dim=1, keepdim=True)  # B, P, C
            if self.with_param:
                x = self.weight[None, :].float() * x + self.bias[None, :].float()
            if reshape:
                x = rearrange(x, "b (p g) c -> b p g c")
            return x


def interpolate_pos_embed(model, checkpoint_model):
    # https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/models/vision_transformer.py#L165
    # only interpolate patch emb, not cls emb (why?)
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        logger.warning(f"pos_embed_checkpoint shape: {pos_embed_checkpoint.shape}")
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        logger.warning(f"num_patches: {num_patches} num_extra_tokens: {num_extra_tokens}")
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            # original dinov2 pretrain img size 518=37x37
            logger.warning(
                "******WARN: Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def sample_pts_feats(pts, feats, npoint=2048, return_index=False):
    """
    pts: B*N*3
    feats: B*N*C
    """
    with autocast(enabled=False, dtype=torch.float32):
        pts = pts.to(dtype=torch.float32)
        feats = feats.to(dtype=torch.float32)
        sample_idx = furthest_point_sample(pts, npoint)
        pts = gather_operation(pts.transpose(1, 2).contiguous(), sample_idx)
        pts = pts.transpose(1, 2).contiguous()
        feats = gather_operation(feats.transpose(1, 2).contiguous(), sample_idx)
        feats = feats.transpose(1, 2).contiguous()
        if return_index:
            return pts, feats, sample_idx
        else:
            return pts, feats


def sample_pts_feats_wlrf(pts, pts_lrf, feats, npoint=2048, return_index=False):
    """
    pts: B*N*3
    pts_lrf: B*N*3
    feats: B*N*C
    """

    with autocast(enabled=False, dtype=torch.float32):
        pts = pts.to(dtype=torch.float32)
        pts_lrf = pts_lrf.to(dtype=torch.float32)
        feats = feats.to(dtype=torch.float32)
        sample_idx = furthest_point_sample(pts, npoint)
        pts = gather_operation(pts.transpose(1, 2).contiguous(), sample_idx)
        pts = pts.transpose(1, 2).contiguous()
        pts_lrf = gather_operation(pts_lrf.transpose(1, 2).contiguous(), sample_idx)
        pts_lrf = pts_lrf.transpose(1, 2).contiguous()
        feats = gather_operation(feats.transpose(1, 2).contiguous(), sample_idx)
        feats = feats.transpose(1, 2).contiguous()
        if return_index:
            return pts, pts_lrf, feats, sample_idx
        else:
            return pts, pts_lrf, feats


def gather_pts_feats(sample_idx, pts, feats):
    """
    sample_idx: # TODO
    pts: B*N*3
    feats: B*N*C
    """
    with autocast(enabled=False, dtype=torch.float32):
        pts = pts.to(dtype=torch.float32)
        feats = feats.to(dtype=torch.float32)
        pts = gather_operation(pts.transpose(1, 2).contiguous(), sample_idx)
        pts = pts.transpose(1, 2).contiguous()
        feats = gather_operation(feats.transpose(1, 2).contiguous(), sample_idx)
        feats = feats.transpose(1, 2).contiguous()
        return pts, feats


def gather_pts_feats_wlrf(sample_idx, pts, pts_lrf, feats):
    """
    sample_idx: # TODO
    pts_lrf: B*N*3
    feats: B*N*C
    """
    with autocast(enabled=False, dtype=torch.float32):
        pts = pts.to(dtype=torch.float32)
        pts_lrf = pts_lrf.to(dtype=torch.float32)
        feats = feats.to(dtype=torch.float32)
        pts = gather_operation(pts.transpose(1, 2).contiguous(), sample_idx)
        pts = pts.transpose(1, 2).contiguous()
        pts_lrf = gather_operation(pts_lrf.transpose(1, 2).contiguous(), sample_idx)
        pts_lrf = pts_lrf.transpose(1, 2).contiguous()
        feats = gather_operation(feats.transpose(1, 2).contiguous(), sample_idx)
        feats = feats.transpose(1, 2).contiguous()
        return pts, pts_lrf, feats


def get_chosen_pixel_feats(img, choose):
    shape = img.size()
    if len(shape) == 3:
        pass
    elif len(shape) == 4:
        B, C, H, W = shape
        img = img.reshape(B, C, H * W)
    else:
        assert False

    choose = choose.unsqueeze(1).repeat(1, C, 1)
    x = torch.gather(img, 2, choose).contiguous()
    return x.transpose(1, 2).contiguous()  # b, p, c


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


def compute_feature_similarity(feat1, feat2, type="cosine", temp=1.0, normalize_feat=True):
    r"""
    Args:
        feat1 (Tensor): (B, N, C)
        feat2 (Tensor): (B, M, C)

    Returns:
        atten_mat (Tensor): (B, N, M)
    """
    if normalize_feat:
        feat1 = F.normalize(feat1, p=2, dim=2)
        feat2 = F.normalize(feat2, p=2, dim=2)

    if type == "cosine":
        atten_mat = feat1 @ feat2.transpose(1, 2)
    elif type == "L2":
        atten_mat = torch.sqrt(pairwise_distance(feat1, feat2, normalized=True))
    else:
        assert False

    atten_mat = atten_mat / temp

    return atten_mat


def aug_pose_noise(gt_r, gt_t, std_rots=[15, 10, 5, 1.25, 1], max_rot=45, sel_std_trans=[0.2, 0.2, 0.2], max_trans=0.8):
    B = gt_r.size(0)
    device = gt_r.device

    std_rot = np.random.choice(std_rots)
    angles = torch.normal(mean=0, std=std_rot, size=(B, 3)).to(device=device)
    angles = angles.clamp(min=-max_rot, max=max_rot)
    ones = gt_r.new(B, 1, 1).zero_() + 1
    zeros = gt_r.new(B, 1, 1).zero_()
    a1 = angles[:, 0].reshape(B, 1, 1) * np.pi / 180.0
    a1 = torch.cat(
        [
            torch.cat([torch.cos(a1), -torch.sin(a1), zeros], dim=2),
            torch.cat([torch.sin(a1), torch.cos(a1), zeros], dim=2),
            torch.cat([zeros, zeros, ones], dim=2),
        ],
        dim=1,
    )
    a2 = angles[:, 1].reshape(B, 1, 1) * np.pi / 180.0
    a2 = torch.cat(
        [
            torch.cat([ones, zeros, zeros], dim=2),
            torch.cat([zeros, torch.cos(a2), -torch.sin(a2)], dim=2),
            torch.cat([zeros, torch.sin(a2), torch.cos(a2)], dim=2),
        ],
        dim=1,
    )
    a3 = angles[:, 2].reshape(B, 1, 1) * np.pi / 180.0
    a3 = torch.cat(
        [
            torch.cat([torch.cos(a3), zeros, torch.sin(a3)], dim=2),
            torch.cat([zeros, ones, zeros], dim=2),
            torch.cat([-torch.sin(a3), zeros, torch.cos(a3)], dim=2),
        ],
        dim=1,
    )
    rand_rot = a1 @ a2 @ a3

    rand_trans = torch.normal(
        mean=torch.zeros([B, 3]).to(device),
        std=torch.tensor(sel_std_trans, device=device).view(1, 3),
    )
    rand_trans = torch.clamp(rand_trans, min=-max_trans, max=max_trans)

    rand_rot = gt_r @ rand_rot
    rand_trans = gt_t + rand_trans
    rand_trans[:, 2] = torch.clamp(rand_trans[:, 2], min=1e-6)

    return rand_rot.detach(), rand_trans.detach()


def compute_coarse_Rt(
    atten,
    pts1,
    pts2,
    model_pts=None,
    n_proposal1=6000,
    n_proposal2=300,
):
    """
    Compute pose_tgt_src (src to tgt)
    Args:
        pts1: Pm, tgt, observed
        pts2: Po, src, rendered
    """
    WSVD = WeightedProcrustes()

    B, N1, _ = pts1.size()
    N2 = pts2.size(1)
    device = pts1.device

    if model_pts is None:
        model_pts = pts2

    atten = atten.float()
    pts1 = pts1.float()
    pts2 = pts2.float()
    model_pts = model_pts.float()

    expand_model_pts = model_pts.unsqueeze(1).repeat(1, n_proposal2, 1, 1).reshape(B * n_proposal2, -1, 3)

    # compute soft assignment matrix
    pred_score = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    pred_label1 = torch.max(pred_score[:, 1:, :], dim=2)[1]
    pred_label2 = torch.max(pred_score[:, :, 1:], dim=1)[1]
    weights1 = (pred_label1 > 0).float()
    weights2 = (pred_label2 > 0).float()

    pred_score = pred_score[:, 1:, 1:].contiguous()
    pred_score = pred_score * weights1.unsqueeze(2) * weights2.unsqueeze(1)
    pred_score = pred_score.reshape(B, N1 * N2) ** 1.5

    # sample pose hypothese
    cumsum_weights = torch.cumsum(pred_score, dim=1)
    cumsum_weights /= cumsum_weights[:, -1].unsqueeze(1).contiguous() + 1e-8  # norm to 0-1
    idx = torch.searchsorted(cumsum_weights, torch.rand(B, n_proposal1 * 3, device=device))  # B, n_proposal1 * 3
    idx1, idx2 = idx.div(N2, rounding_mode="floor"), idx % N2
    idx1 = torch.clamp(idx1, max=N1 - 1).unsqueeze(2).repeat(1, 1, 3)
    idx2 = torch.clamp(idx2, max=N2 - 1).unsqueeze(2).repeat(1, 1, 3)

    p1 = torch.gather(pts1, 1, idx1).reshape(B, n_proposal1, 3, 3).reshape(B * n_proposal1, 3, 3)
    p2 = torch.gather(pts2, 1, idx2).reshape(B, n_proposal1, 3, 3).reshape(B * n_proposal1, 3, 3)
    pred_rs, pred_ts = WSVD(p2, p1, None)
    pred_rs = pred_rs.reshape(B, n_proposal1, 3, 3)
    pred_ts = pred_ts.reshape(B, n_proposal1, 1, 3)

    p1 = p1.reshape(B, n_proposal1, 3, 3)
    p2 = p2.reshape(B, n_proposal1, 3, 3)
    dis = torch.norm((p1 - pred_ts) @ pred_rs - p2, dim=3).mean(2)
    idx = torch.topk(dis, n_proposal2, dim=1, largest=False)[1]
    pred_rs = torch.gather(pred_rs, 1, idx.reshape(B, n_proposal2, 1, 1).repeat(1, 1, 3, 3))
    pred_ts = torch.gather(pred_ts, 1, idx.reshape(B, n_proposal2, 1, 1).repeat(1, 1, 1, 3))

    # pose selection
    transformed_pts = (pts1.unsqueeze(1) - pred_ts) @ pred_rs
    transformed_pts = transformed_pts.reshape(B * n_proposal2, -1, 3)
    dis = torch.sqrt(pairwise_distance(transformed_pts, expand_model_pts))
    dis = dis.min(2)[0].reshape(B, n_proposal2, -1)
    scores = weights1.unsqueeze(1).sum(2) / ((dis * weights1.unsqueeze(1)).sum(2) + +1e-8)
    pose_score, idx = scores.max(1)
    pred_R = torch.gather(pred_rs, 1, idx.reshape(B, 1, 1, 1).repeat(1, 1, 3, 3)).squeeze(1)
    pred_t = torch.gather(pred_ts, 1, idx.reshape(B, 1, 1, 1).repeat(1, 1, 1, 3)).squeeze(2).squeeze(1)
    # no grad, coarse svd is not differentiable
    return pred_R, pred_t, pose_score


def compute_coarse_Rt_overlap(
    atten,
    score,
    pts1,
    pts2,
    model_pts=None,
    n_proposal1=6000,
    n_proposal2=300,
):
    """
    Compute pose_tgt_src (src to tgt)
    Args:
        pts1: Pm, tgt, observed
        pts2: Po, src, rendered
    """
    WSVD = WeightedProcrustes()

    B, N1, _ = pts1.size()
    N2 = pts2.size(1)
    device = pts1.device

    if model_pts is None:
        model_pts = pts2

    atten = atten.float()
    pts1 = pts1.float()
    pts2 = pts2.float()
    model_pts = model_pts.float()
    score1 = score[:, :N1].float()
    score2 = score[:, N2:].float()

    expand_model_pts = model_pts.unsqueeze(1).repeat(1, n_proposal2, 1, 1).reshape(B * n_proposal2, -1, 3)
    expand_dim = torch.ones((B, 1), device=device)
    score1 = torch.cat((expand_dim, score1), dim=1)[:, :, None].repeat(1, 1, N2 + 1)
    score2 = torch.cat((expand_dim, score2), dim=1)[:, None, :].repeat(1, N1 + 1, 1)

    # compute soft assignment matrix
    pred_score = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    pred_score = pred_score * score1 * score2
    pred_label1 = torch.max(pred_score[:, 1:, :], dim=2)[1]
    pred_label2 = torch.max(pred_score[:, :, 1:], dim=1)[1]
    weights1 = (pred_label1 > 0).float()
    weights2 = (pred_label2 > 0).float()

    pred_score = pred_score[:, 1:, 1:].contiguous()
    pred_score = pred_score * weights1.unsqueeze(2) * weights2.unsqueeze(1)
    pred_score = pred_score.reshape(B, N1 * N2) ** 1.5

    # sample pose hypothese
    cumsum_weights = torch.cumsum(pred_score, dim=1)
    cumsum_weights /= cumsum_weights[:, -1].unsqueeze(1).contiguous() + 1e-8  # norm to 0-1
    idx = torch.searchsorted(cumsum_weights, torch.rand(B, n_proposal1 * 3, device=device))  # B, n_proposal1 * 3
    idx1, idx2 = idx.div(N2, rounding_mode="floor"), idx % N2
    idx1 = torch.clamp(idx1, max=N1 - 1).unsqueeze(2).repeat(1, 1, 3)
    idx2 = torch.clamp(idx2, max=N2 - 1).unsqueeze(2).repeat(1, 1, 3)

    p1 = torch.gather(pts1, 1, idx1).reshape(B, n_proposal1, 3, 3).reshape(B * n_proposal1, 3, 3)
    p2 = torch.gather(pts2, 1, idx2).reshape(B, n_proposal1, 3, 3).reshape(B * n_proposal1, 3, 3)
    pred_rs, pred_ts = WSVD(p2, p1, None)
    pred_rs = pred_rs.reshape(B, n_proposal1, 3, 3)
    pred_ts = pred_ts.reshape(B, n_proposal1, 1, 3)

    p1 = p1.reshape(B, n_proposal1, 3, 3)
    p2 = p2.reshape(B, n_proposal1, 3, 3)
    dis = torch.norm((p1 - pred_ts) @ pred_rs - p2, dim=3).mean(2)
    idx = torch.topk(dis, n_proposal2, dim=1, largest=False)[1]
    pred_rs = torch.gather(pred_rs, 1, idx.reshape(B, n_proposal2, 1, 1).repeat(1, 1, 3, 3))
    pred_ts = torch.gather(pred_ts, 1, idx.reshape(B, n_proposal2, 1, 1).repeat(1, 1, 1, 3))

    # pose selection
    transformed_pts = (pts1.unsqueeze(1) - pred_ts) @ pred_rs
    transformed_pts = transformed_pts.reshape(B * n_proposal2, -1, 3)
    dis = torch.sqrt(pairwise_distance(transformed_pts, expand_model_pts))
    dis = dis.min(2)[0].reshape(B, n_proposal2, -1)
    scores = weights1.unsqueeze(1).sum(2) / ((dis * weights1.unsqueeze(1)).sum(2) + +1e-8)
    pose_score, idx = scores.max(1)
    pred_R = torch.gather(pred_rs, 1, idx.reshape(B, 1, 1, 1).repeat(1, 1, 3, 3)).squeeze(1)
    pred_t = torch.gather(pred_ts, 1, idx.reshape(B, 1, 1, 1).repeat(1, 1, 1, 3)).squeeze(2).squeeze(1)
    # no grad, coarse svd is not differentiable
    return pred_R, pred_t, pose_score


def compute_fine_Rt(atten, pts1, pts2, model_pts=None, dis_thres=0.15):
    if model_pts is None:
        model_pts = pts2
    atten = atten.float()
    pts1 = pts1.float()
    pts2 = pts2.float()
    model_pts = model_pts.float()

    # compute pose
    WSVD = WeightedProcrustes(weight_thresh=0.0)
    assginment_mat = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    label1 = torch.max(assginment_mat[:, 1:, :], dim=2)[1]
    label2 = torch.max(assginment_mat[:, :, 1:], dim=1)[1]

    assginment_mat = assginment_mat[:, 1:, 1:] * (label1 > 0).float().unsqueeze(2) * (label2 > 0).float().unsqueeze(1)
    # max_idx = torch.max(assginment_mat, dim=2, keepdim=True)[1]
    # pred_pts = torch.gather(pts2, 1, max_idx.expand_as(pts1))
    normalized_assginment_mat = assginment_mat / (assginment_mat.sum(2, keepdim=True) + 1e-6)
    pred_pts = normalized_assginment_mat @ pts2

    assginment_score = assginment_mat.sum(2)
    pred_R, pred_t = WSVD(pred_pts, pts1, assginment_score)

    # compute score
    pred_pts = (pts1 - pred_t.unsqueeze(1)) @ pred_R
    dis = torch.sqrt(pairwise_distance(pred_pts, model_pts)).min(2)[0]
    mask = (label1 > 0).float()
    pose_score = (dis < dis_thres).float()
    pose_score = (pose_score * mask).sum(1) / (mask.sum(1) + 1e-8)
    pose_score = pose_score * mask.mean(1)

    return pred_R, pred_t, pose_score


def compute_fine_Rt_overlap(atten, score, pts1, pts2, model_pts=None, dis_thres=0.15):
    WSVD = WeightedProcrustes(weight_thresh=0.001)
    if model_pts is None:
        model_pts = pts2
    atten = atten.float()
    pts1 = pts1.float()
    pts2 = pts2.float()
    model_pts = model_pts.float()
    B, N1 = pts1.shape[:2]
    N2 = pts2.shape[1]
    score1 = score[:, :N1]
    score2 = score[:, N1:]
    expand_dim = torch.ones((B, 1), device=score1.device)
    score1 = torch.cat((expand_dim, score1), dim=1)[:, :, None].repeat(1, 1, N2 + 1)
    score2 = torch.cat((expand_dim, score2), dim=1)[:, None, :].repeat(1, N1 + 1, 1)
    assginment_mat = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    assginment_mat = assginment_mat * score1 * score2

    # compute pose
    label1 = torch.max(assginment_mat[:, 1:, :], dim=2)[1]
    label2 = torch.max(assginment_mat[:, :, 1:], dim=1)[1]

    assginment_mat = assginment_mat[:, 1:, 1:] * (label1 > 0).float().unsqueeze(2) * (label2 > 0).float().unsqueeze(1)
    # max_idx = torch.max(assginment_mat, dim=2, keepdim=True)[1]
    # pred_pts = torch.gather(pts2, 1, max_idx.expand_as(pts1))
    normalized_assginment_mat = assginment_mat / (assginment_mat.sum(2, keepdim=True) + 1e-6)
    pred_pts = normalized_assginment_mat @ pts2

    assginment_score = assginment_mat.sum(2)
    pred_R, pred_t = WSVD(pred_pts, pts1, assginment_score)

    # compute score
    pred_pts = (pts1 - pred_t.unsqueeze(1)) @ pred_R
    dis = torch.sqrt(pairwise_distance(pred_pts, model_pts)).min(2)[0]
    mask = (label1 > 0).float()
    pose_score = (dis < dis_thres).float()
    pose_score = (pose_score * mask).sum(1) / (mask.sum(1) + 1e-8)
    pose_score = pose_score * mask.mean(1)

    return pred_R, pred_t, pose_score


def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def transform(pts, trans):
    if len(pts.shape) == 3:
        trans_pts = torch.einsum("bnm,bmk->bnk", trans[:, :3, :3], pts.permute(0, 2, 1)) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = torch.einsum("nm,mk->nk", trans[:3, :3], pts.T) + trans[:3, 3:4]
        return trans_pts.T


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
        torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6
    )
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
        torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6
    )

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)  # 升维度，然后变为对角阵
    H = Am.permute(0, 2, 1) @ Weight @ Bm  # permute : tensor中的每一块做转置

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)


def post_refinement(initial_trans, src_kpts, tgt_kpts, iters, inlier_threshold=0.1, weights=None):
    pre_inlier_count = 0
    for i in range(iters):
        pred_tgt = transform(src_kpts, initial_trans)
        L2_dis = torch.norm(pred_tgt - tgt_kpts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]
        inlier_count = torch.sum(pred_inlier)
        if inlier_count <= pre_inlier_count:
            break
        pre_inlier_count = inlier_count
        initial_trans = rigid_transform_3d(
            A=src_kpts[:, pred_inlier, :],
            B=tgt_kpts[:, pred_inlier, :],
            weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
        )
    return initial_trans


def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
    src_centroid=None,
    ref_centroid=None,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    if src_centroid is None:
        src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(src_centroid.size()) == 2:
        src_centroid = src_centroid.unsqueeze(1)
    src_points_centered = src_points - src_centroid  # (B, N, 3)

    if ref_centroid is None:
        ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(ref_centroid.size()) == 2:
        ref_centroid = ref_centroid.unsqueeze(1)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H)
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(src_points.device)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.5, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None, src_centroid=None, ref_centroid=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
            src_centroid=src_centroid,
            ref_centroid=ref_centroid,
        )


class LRF(nn.Module):
    def __init__(
        self,
        r_lrf,
        eps=1e-10,
    ):
        super(LRF, self).__init__()

        self.eps = eps
        self.r_lrf = r_lrf

    def forward(self, xyz, xyz_group):
        B, N, c = xyz_group.size()
        xyz_group = xyz_group.contiguous()  # dim = B x 3 x N
        xyz = xyz.contiguous()  # dim = B x 3 x 1

        # zp
        x = xyz - xyz_group  # pi->p = p - pi
        xxt = torch.bmm(x, x.transpose(1, 2)) / c

        _, _, v = torch.svd(xxt)

        # with torch.no_grad():
        #     sum_ = (v[..., -1].unsqueeze(1) @ x).sum(2)
        #     _sign = torch.ones((len(xyz_group), 1), device=xyz_group.device) - 2 * (sum_ < 0)
        with torch.no_grad():
            center_proj = v[..., -1].unsqueeze(1) @ x
            sum_ = (center_proj > 1e-3).sum(-1) - (center_proj < -1e-3).sum(-1)
            _sign = torch.ones((len(xyz_group), 1), device=xyz_group.device) - 2 * (sum_ < 0)

        zp = (_sign * v[..., -1]).unsqueeze(1)  # B x 1 x 3
        # zp = v[..., -1].unsqueeze(1)

        # xp
        x *= -1  # p->pi = pi - p
        norm = (zp @ x).transpose(1, 2)
        proj = norm * zp

        vi = x - proj.transpose(1, 2)

        x_l2 = torch.sqrt((x**2).sum(dim=1, keepdim=True))

        alpha = self.r_lrf[:, None, None] - x_l2
        alpha = alpha * alpha
        beta = (norm * norm).transpose(1, 2)
        vi_c = (alpha * beta * vi).sum(2)

        xp = vi_c / (torch.sqrt((vi_c**2).sum(1, keepdim=True)) + self.eps)

        # yp
        yp = torch.cross(xp, zp.squeeze(1), dim=1)

        lrf = torch.cat((xp.unsqueeze(2), yp.unsqueeze(2), zp.transpose(1, 2)), dim=2)

        _out_xp = (xyz_group - xyz) / self.r_lrf[:, None, None]
        out_xp = lrf.transpose(1, 2) @ _out_xp

        return out_xp


if __name__ == "__main__":
    pass
