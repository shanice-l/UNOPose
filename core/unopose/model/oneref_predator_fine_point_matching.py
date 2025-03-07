import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

from core.unopose.model.transformer import SparseToDenseTransformer
from core.unopose.utils.model_utils import compute_feature_similarity, compute_fine_Rt_overlap
from core.unopose.utils.loss_utils import compute_overlap_loss
from core.unopose.model.pointnet2.pointnet2_utils import QueryAndGroup, QueryAndLRFGroup
from core.unopose.model.pointnet2.pytorch_utils import SharedMLP, Conv1d


class FinePointMatchingOneRef(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(FinePointMatchingOneRef, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock

        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

        self.dis_proj = nn.Linear(2 * cfg.hidden_dim, 3)

        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.PE = PositionalEncoding(
            cfg.hidden_dim,
            r1=cfg.pe_radius1,
            r2=cfg.pe_radius2,
            nsample1=cfg.get("nsample1", 32),
            nsample2=cfg.get("nsample2", 64),
            use_lrf=cfg.use_lrf,
            use_xyz=cfg.use_xyz,
            use_feature=cfg.get("use_feature", False),
        )
        self.score_heads = []
        for _ in range(self.nblock):
            self.score_heads.append(nn.Linear(cfg.hidden_dim, 1))
        self.score_heads = nn.ModuleList(self.score_heads)

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(
                SparseToDenseTransformer(
                    cfg.hidden_dim,
                    num_heads=4,
                    sparse_blocks=["self", "cross"],
                    dropout=None,
                    activation_fn="ReLU",
                    focusing_factor=cfg.focusing_factor,
                    with_bg_token=True,
                    replace_bg_token=True,
                )
            )
        self.transformers = nn.ModuleList(self.transformers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, p1, f1, geo1, fps_idx1, p2, f2, geo2, fps_idx2, radius, end_points):
        """
        p1 (B, n1, 3): tgt, m, observed
        p2 (B, n2, 3): src, o, rendered
        """
        B, n1 = p1.size(0), p1.size(1)
        n2 = p2.size(1)
        if "init_R" in end_points and "init_t" in end_points:
            init_R = end_points["init_R"]
            init_t = end_points["init_t"]
            p1_ = (p1 - init_t.unsqueeze(1)) @ init_R
        else:
            p1_ = p1

        f1 = self.in_proj(f1) + self.PE(p1_)
        f1 = torch.cat([self.bg_token.repeat(B, 1, 1), f1], dim=1)  # adding bg

        f2 = self.in_proj(f2) + self.PE(p2)
        f2 = torch.cat([self.bg_token.repeat(B, 1, 1), f2], dim=1)  # adding bg

        atten_list = []
        score_list = []
        saliency_list = []
        for idx in range(self.nblock):
            f1, f2 = self.transformers[idx](f1, geo1, fps_idx1, f2, geo2, fps_idx2)
            scores = self.score_heads[idx](torch.cat((f1, f2), dim=1))

            if self.training or idx == self.nblock - 1:
                atten_list.append(
                    compute_feature_similarity(
                        self.out_proj(f1), self.out_proj(f2), self.cfg.sim_type, self.cfg.temp, self.cfg.normalize_feat
                    )
                )  # bs, n1+1, n2+1
                s1, s2 = scores[:, 1 : (n1 + 1)], scores[:, (n1 + 2) :]  # bs, n1, 1; bs, n2, 1
                m1 = torch.matmul(F.softmax(atten_list[-1][:, 1:, 1:], dim=2), s2)  # bs, n1, 1
                m2 = torch.matmul(F.softmax(atten_list[-1][:, 1:, 1:].transpose(1, 2), dim=2), s1)  # bs, n2, 1
                score = torch.cat((s1, s2), dim=1).squeeze(-1)
                score = torch.clamp(self.sigmoid(score), min=0, max=1)
                score_list.append(score)
                saliency = torch.cat((m1, m2), dim=1).squeeze(-1)
                saliency = torch.clamp(self.sigmoid(saliency), min=0, max=1)
                saliency_list.append(saliency)

        if self.training:
            gt_R = end_points["rotation_label"]
            gt_t = end_points["translation_label"] / (radius.reshape(-1, 1) + 1e-6)

            end_points = compute_overlap_loss(
                end_points,
                atten_list,
                score_list,
                saliency_list,
                p1,
                p2,
                gt_R,
                gt_t,
                predator_thres=self.cfg.loss_predator_thres,
                dis_thres=self.cfg.loss_dis_thres,
                loss_str="fine",
            )
        else:
            # TODO: change find RT
            pred_R, pred_t, pred_pose_score = compute_fine_Rt_overlap(
                atten_list[-1],
                score_list[-1],
                p1,
                p2,
                # end_points["model"] / (radius.reshape(-1, 1, 1) + 1e-6),
                None,
            )
            end_points["pred_R"] = pred_R
            end_points["pred_t"] = pred_t * (radius.reshape(-1, 1) + 1e-6)
            end_points["pred_pose_score"] = pred_pose_score

        if self.return_feat:
            return end_points, self.out_proj(f1), self.out_proj(f2)
        else:
            return end_points


class PositionalEncoding(nn.Module):
    def __init__(
        self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_lrf=True, use_xyz=False, use_feature=False, bn=True
    ):
        super(PositionalEncoding, self).__init__()
        if use_lrf:
            self.group1 = QueryAndLRFGroup(r1, nsample1, use_xyz=use_xyz, use_feature=use_feature)
            self.group2 = QueryAndLRFGroup(r2, nsample2, use_xyz=use_xyz, use_feature=use_feature)
        else:
            self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
            self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 3
        if use_xyz:
            input_dim += 3
        if use_feature:
            input_dim += 3

        self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp3 = Conv1d(256, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):
        if pts2 is None:
            pts2 = pts1

        pts1 = pts1.to(dtype=torch.float32)
        pts2 = pts2.to(dtype=torch.float32)
        with autocast(enabled=False):
            # scale1
            feat1 = self.group1(pts1.contiguous(), pts2.contiguous(), pts1.transpose(1, 2).contiguous())
            feat1 = self.mlp1(feat1)
            feat1 = F.max_pool2d(feat1, kernel_size=[1, feat1.size(3)])

            # scale2
            feat2 = self.group2(pts1.contiguous(), pts2.contiguous(), pts1.transpose(1, 2).contiguous())
            feat2 = self.mlp2(feat2)
            feat2 = F.max_pool2d(feat2, kernel_size=[1, feat2.size(3)])

            feat = torch.cat([feat1, feat2], dim=1).squeeze(-1)
            feat = self.mlp3(feat).transpose(1, 2)
            return feat
