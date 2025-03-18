import torch
import torch.nn as nn
import torch.nn.functional as F

from core.unopose.model.transformer import GeometricTransformer
from core.unopose.utils.model_utils import (
    compute_feature_similarity,
    aug_pose_noise,
    compute_coarse_Rt_overlap,
)
from core.unopose.utils.loss_utils import compute_overlap_loss, compute_soft_loss


class CoarsePointMatchingOneRef(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(CoarsePointMatchingOneRef, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock

        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)

        self.score_heads = []
        for _ in range(self.nblock):
            self.score_heads.append(nn.Linear(cfg.hidden_dim, 1))
        self.score_heads = nn.ModuleList(self.score_heads)

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(
                GeometricTransformer(
                    blocks=["self", "cross"],
                    d_model=cfg.hidden_dim,
                    num_heads=4,
                    dropout=None,
                    activation_fn="ReLU",
                    return_attention_scores=False,
                )
            )
        self.transformers = nn.ModuleList(self.transformers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, p1, f1, geo1, p2, f2, geo2, radius, end_points):
        B, n1 = f1.size(0), f1.size(1)
        n2 = f2.size(1)

        f1 = self.in_proj(f1)
        f1 = torch.cat([self.bg_token.repeat(B, 1, 1), f1], dim=1)  # adding bg
        f2 = self.in_proj(f2)
        f2 = torch.cat([self.bg_token.repeat(B, 1, 1), f2], dim=1)  # adding bg

        atten_list = []
        score_list = []
        saliency_list = []
        for idx in range(self.nblock):
            f1, f2 = self.transformers[idx](f1, geo1, f2, geo2)
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
            init_R, init_t = aug_pose_noise(gt_R, gt_t)

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
                loss_str="coarse_hard",
            )
            if self.cfg.get("softloss_weight", 0.0) > 0:
                end_points = compute_soft_loss(end_points, atten_list, p1, p2, gt_R, gt_t, loss_str="coarse_soft")
        else:
            init_R, init_t, init_score = compute_coarse_Rt_overlap(
                atten_list[-1],
                score_list[-1],
                p1,
                p2,
                # end_points["model"] / (radius.reshape(-1, 1, 1) + 1e-6),
                None,
                self.cfg.nproposal1,
                self.cfg.nproposal2,
            )
            end_points["init_pose_score"] = init_score

        end_points["init_R"] = init_R
        end_points["init_t"] = init_t

        if self.return_feat:
            return end_points, self.out_proj(f1), self.out_proj(f2)
        else:
            return end_points
