import torch
import torch.nn as nn
import logging
from detectron2.utils.logger import log_first_n, log_every_n

from core.unopose.utils.model_utils import pairwise_distance


def compute_circle_loss(end_points, atten_list, gt_r, gt_t, loss_cfg, loss_str="coarse"):
    "circle loss for correspondence loss"
    weighted_circle_loss_func = WeightedCircleLoss(
        loss_cfg.positive_margin,
        loss_cfg.negative_margin,
        loss_cfg.positive_optimal,
        loss_cfg.negative_optimal,
        loss_cfg.log_scale,
    )

    batch_size = len(end_points["gt_node_corr_indices"])
    for idx, atten in enumerate(atten_list):
        atten = atten.float()
        loss = torch.zeros(batch_size).to(atten.device)
        for j in range(batch_size):
            gt_node_corr_indices = end_points["gt_node_corr_indices"][j]
            gt_node_corr_overlaps = end_points["gt_node_corr_overlaps"][j]
            gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
            gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
            atten_j = atten[j, 1:, 1:].T
            overlaps = torch.zeros_like(atten_j)
            overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
            pos_masks = torch.gt(overlaps, loss_cfg.positive_overlap)
            neg_masks = torch.eq(overlaps, 0)
            pos_scales = torch.sqrt(overlaps * pos_masks.float())
            loss[j] = weighted_circle_loss_func(pos_masks, neg_masks, atten_j, pos_scales)

        end_points[loss_str + "_loss" + str(idx)] = loss

    return end_points


def compute_soft_loss_debug(end_points, atten_list, pts1, pts2, gt_R, gt_t, loss_str="coarse"):
    """
    Calculating loss for coarse node matching
    :param scores: Predicted matching matrix
    :param matching_mask: The ground truth soft matching mask
    :return: Calculated loss
    """
    CE = nn.CrossEntropyLoss(reduction="none")
    node_corr_norm_row = end_points["node_corr_norm_row"].max(1).indices
    node_corr_norm_col = end_points["node_corr_norm_col"].max(1).indices
    for idx, atten in enumerate(atten_list):
        l1 = CE(atten.transpose(1, 2)[:, :, 1:].contiguous(), node_corr_norm_row).mean(1)
        l2 = CE(atten[:, :, 1:].contiguous(), node_corr_norm_col).mean(1)
        end_points[loss_str + "_loss" + str(idx)] = 0.5 * (l1 + l2)
        # mask_scores = matching_mask * torch.log(atten + 1e-6)
        # loss = torch.sum(-mask_scores, (2, 1)) / torch.sum(matching_mask, (2, 1))
        # end_points[loss_str + "_loss" + str(idx)] = loss
        # if not torch.isfinite(l1).all() or not torch.isfinite(l2).all():

    return end_points


def compute_soft_loss_v0(end_points, atten_list, pts1, pts2, gt_R, gt_t, loss_str="coarse"):
    """
    Calculating loss for coarse node matching
    :param scores: Predicted matching matrix
    :param matching_mask: The ground truth soft matching mask
    :return: Calculated loss
    """
    CE = nn.CrossEntropyLoss(reduction="none")
    node_correspondences = end_points["node_correspondences"]
    soft_label1 = node_correspondences.softmax(1)
    soft_label2 = node_correspondences.softmax(2)
    for idx, atten in enumerate(atten_list):
        l1 = CE(atten.transpose(1, 2).contiguous(), soft_label1.transpose(1, 2).contiguous()).mean(1)
        l2 = CE(atten.contiguous(), soft_label2.contiguous()).mean(1)
        end_points[loss_str + "_loss" + str(idx)] = 0.5 * (l1 + l2)
        # mask_scores = matching_mask * torch.log(atten + 1e-6)
        # loss = torch.sum(-mask_scores, (2, 1)) / torch.sum(matching_mask, (2, 1))
        # end_points[loss_str + "_loss" + str(idx)] = loss
        # if not torch.isfinite(l1).all() or not torch.isfinite(l2).all():

    return end_points


def compute_soft_loss(end_points, atten_list, pts1, pts2, gt_R, gt_t, loss_str="coarse"):
    """
    Calculating loss for coarse node matching
    :param scores: Predicted matching matrix
    :param matching_mask: The ground truth soft matching mask
    :return: Calculated loss
    """
    CE = nn.CrossEntropyLoss(reduction="none")
    node_corr_norm_row = end_points["node_corr_norm_row"]  # [197,196]
    node_corr_norm_col = end_points["node_corr_norm_col"]  # [197,196]
    for idx, atten in enumerate(atten_list):
        l1 = CE(atten.transpose(1, 2)[:, :, 1:].contiguous(), node_corr_norm_row).mean(1)
        l2 = CE(atten[:, :, 1:].contiguous(), node_corr_norm_col).mean(1)
        end_points[loss_str + "_loss" + str(idx)] = 0.5 * (l1 + l2)
        # mask_scores = matching_mask * torch.log(atten + 1e-6)
        # loss = torch.sum(-mask_scores, (2, 1)) / torch.sum(matching_mask, (2, 1))
        # end_points[loss_str + "_loss" + str(idx)] = loss
        # if not torch.isfinite(l1).all() or not torch.isfinite(l2).all():

    return end_points


def get_weighted_bce_loss(prediction, gt):
    loss = nn.BCELoss(reduction="none")

    class_loss = loss(prediction, gt)

    weights = torch.ones_like(gt)
    w_negative = gt.sum(1) / gt.size(1)
    w_positive = 1 - w_negative

    w_positive = w_positive[:, None].repeat(1, gt.shape[1])
    w_negative = w_negative[:, None].repeat(1, gt.shape[1])

    weights[gt >= 0.5] = w_positive[gt >= 0.5]
    weights[gt < 0.5] = w_negative[gt < 0.5]
    w_class_loss = (weights * class_loss).mean(1)

    #######################################
    # get classification precision and recall
    # predicted_labels = prediction.detach().cpu().round().numpy()
    # cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(),predicted_labels, average='binary')

    return w_class_loss


def compute_overlap_loss(
    end_points,
    atten_list,
    score_list,
    saliency_list,
    pts1,
    pts2,
    gt_r,
    gt_t,
    predator_thres=0.15,
    dis_thres=0.15,
    loss_str="coarse",
):
    """InfoNCE loss for correspondence loss
    1: tgt
    2: template (src)
    """
    B, n1 = pts1.shape[:2]
    CE = nn.CrossEntropyLoss(reduction="none")
    # gt pose should be T_1_2, i.e., T_tgt_src
    # equivalent to: R^T @ (p - t) --transpose--> (p^T - t^T) @ R
    gt_pts = (pts1 - gt_t.unsqueeze(1)) @ gt_r
    dis_mat = torch.sqrt(pairwise_distance(gt_pts, pts2))
    gt_overlap = torch.zeros_like(score_list[0])  # bs, n1+n2
    for ind in range(B):
        corr = torch.stack(torch.where(dis_mat[ind] <= predator_thres), dim=-1)
        idx1, idx2 = torch.unique(corr[:, 0]), torch.unique(corr[:, 1])
        idx2 += n1
        idx = torch.cat((idx1, idx2), dim=0)
        gt_overlap[ind][idx] = 1

    # calculate score loss
    for idx, score in enumerate(score_list):
        score = score.float()
        end_points[loss_str + "_score_loss" + str(idx)] = get_weighted_bce_loss(score, gt_overlap)

    # calculate saliency loss
    for idx, saliency in enumerate(saliency_list):
        saliency = saliency.float()
        end_points[loss_str + "_saliency_loss" + str(idx)] = get_weighted_bce_loss(saliency, gt_overlap)

    dis1, label1 = dis_mat.min(2)
    fg_label1 = (dis1 <= dis_thres).float()  # (min dist) & (< thr)
    label1 = (fg_label1 * (label1.float() + 1.0)).long()  # idx label

    dis2, label2 = dis_mat.min(1)
    fg_label2 = (dis2 <= dis_thres).float()
    label2 = (fg_label2 * (label2.float() + 1.0)).long()
    # loss
    for idx, atten in enumerate(atten_list):
        atten = atten.float()
        l1 = CE(atten.transpose(1, 2)[:, :, 1:].contiguous(), label1).mean(1)
        l2 = CE(atten[:, :, 1:].contiguous(), label2).mean(1)
        end_points[loss_str + "_atten_loss" + str(idx)] = 0.5 * (l1 + l2)

    # acc
    pred_label = torch.max(atten_list[-1][:, 1:, :], dim=2)[1]
    end_points[loss_str + "_acc"] = (pred_label == label1).float().mean(1)

    # pred foreground num
    fg_mask = (pred_label > 0).float()
    end_points[loss_str + "_fg_num"] = fg_mask.sum(1)

    # foreground point dis
    fg_label = fg_mask * (pred_label - 1)
    fg_label = fg_label.long()
    pred_pts = torch.gather(pts2, 1, fg_label.unsqueeze(2).repeat(1, 1, 3))
    pred_dis = torch.norm(pred_pts - gt_pts, dim=2)
    pred_dis = (pred_dis * fg_mask).sum(1) / (fg_mask.sum(1) + 1e-8)
    end_points[loss_str + "_dis"] = pred_dis

    return end_points


def compute_correspondence_loss(end_points, atten_list, pts1, pts2, gt_r, gt_t, dis_thres=0.15, loss_str="coarse"):
    """InfoNCE loss for correspondence loss
    1: tgt
    2: template (src)
    """
    CE = nn.CrossEntropyLoss(reduction="none")
    # gt pose should be T_1_2, i.e., T_tgt_src
    # equivalent to: R^T @ (p - t) --transpose--> (p^T - t^T) @ R
    gt_pts = (pts1 - gt_t.unsqueeze(1)) @ gt_r
    dis_mat = torch.sqrt(pairwise_distance(gt_pts, pts2))

    dis1, label1 = dis_mat.min(2)
    fg_label1 = (dis1 <= dis_thres).float()  # (min dist) & (< thr)
    label1 = (fg_label1 * (label1.float() + 1.0)).long()  # idx label

    dis2, label2 = dis_mat.min(1)
    fg_label2 = (dis2 <= dis_thres).float()
    label2 = (fg_label2 * (label2.float() + 1.0)).long()
    # loss
    for idx, atten in enumerate(atten_list):
        atten = atten.float()
        l1 = CE(atten.transpose(1, 2)[:, :, 1:].contiguous(), label1).mean(1)
        l2 = CE(atten[:, :, 1:].contiguous(), label2).mean(1)
        end_points[loss_str + "_loss" + str(idx)] = 0.5 * (l1 + l2)

    # acc
    pred_label = torch.max(atten_list[-1][:, 1:, :], dim=2)[1]
    end_points[loss_str + "_acc"] = (pred_label == label1).float().mean(1)

    # pred foreground num
    fg_mask = (pred_label > 0).float()
    end_points[loss_str + "_fg_num"] = fg_mask.sum(1)

    # foreground point dis
    fg_label = fg_mask * (pred_label - 1)
    fg_label = fg_label.long()
    pred_pts = torch.gather(pts2, 1, fg_label.unsqueeze(2).repeat(1, 1, 3))
    pred_dis = torch.norm(pred_pts - gt_pts, dim=2)
    pred_dis = (pred_dis * fg_mask).sum(1) / (fg_mask.sum(1) + 1e-8)
    end_points[loss_str + "_dis"] = pred_dis

    return end_points


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, end_points):
        out_dicts = {"loss": 0}
        for key in end_points.keys():
            if "coarse_" in key or "fine_" in key:
                out_dicts[key] = end_points[key].mean()
                if "loss" in key:
                    out_dicts["loss"] = out_dicts["loss"] + end_points[key]
        out_dicts["loss"] = torch.clamp(out_dicts["loss"], max=100.0).mean()
        return out_dicts


def process_loss(end_points):
    out_dicts = {"loss": 0}
    for key in end_points.keys():
        if "coarse_" in key or "fine_" in key:
            log_first_n(logging.INFO, f"end_points: key: {key}, len: {len(end_points[key])}")
            out_dicts[key] = end_points[key].mean()
            if "loss" in key:
                out_dicts["loss"] = out_dicts["loss"] + end_points[key]
    out_dicts["loss"] = torch.clamp(out_dicts["loss"], max=100.0).mean()
    return out_dicts
