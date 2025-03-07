import torch
import torch.nn as nn

from core.unopose.model.oneref_feature_extraction import ViTEncoderOneRef
from core.unopose.model.oneref_predator_coarse_point_matching import CoarsePointMatchingOneRef
from core.unopose.model.oneref_predator_fine_point_matching import FinePointMatchingOneRef
from core.unopose.model.transformer import GeometricStructureEmbedding
from core.unopose.utils.model_utils import sample_pts_feats_wlrf, LRF


class UNOPose(nn.Module):
    def __init__(self, cfg):
        super(UNOPose, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint
        self.use_ref_rad = cfg.get("use_ref_rad", False)
        self.test_coarse_only = cfg.get("test_coarse_only", False)

        self.feature_extraction = ViTEncoderOneRef(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatchingOneRef(cfg.coarse_point_matching)
        self.fine_point_matching = FinePointMatchingOneRef(cfg.fine_point_matching)

    def forward(self, end_points):
        dense_pm, dense_fm, dense_po, dense_fo, radius = self.feature_extraction(end_points)

        # turn dense_pm and dense_po in lrf space
        dense_pm_lrf = self.get_batch_lrf(end_points["pts"])
        dense_po_lrf = self.get_batch_lrf(end_points["tem1_pts"])

        # pre-compute geometric embeddings for geometric transformer
        bg_point = torch.ones(dense_pm.size(0), 1, 3).float().to(dense_pm.device)

        sparse_pm, sparse_pm_lrf, sparse_fm, fps_idx_m = sample_pts_feats_wlrf(
            dense_pm, dense_pm_lrf, dense_fm, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm_lrf], dim=1))

        sparse_po, sparse_po_lrf, sparse_fo, fps_idx_o = sample_pts_feats_wlrf(
            dense_po, dense_po_lrf, dense_fo, self.coarse_npoint, return_index=True
        )
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po_lrf], dim=1))

        # coarse_point_matching
        end_points = self.coarse_point_matching(
            sparse_pm,
            sparse_fm,
            geo_embedding_m,
            sparse_po,
            sparse_fo,
            geo_embedding_o,
            radius,
            end_points,
        )
        if not self.training and self.test_coarse_only:
            end_points["pred_R"] = end_points["init_R"]
            end_points["pred_t"] = end_points["init_t"] * (radius.reshape(-1, 1) + 1e-6)
            end_points["pred_pose_score"] = end_points["init_pose_score"]
            return end_points

        # fine_point_matching
        end_points = self.fine_point_matching(
            dense_pm,
            dense_fm,
            geo_embedding_m,
            fps_idx_m,
            dense_po,
            dense_fo,
            geo_embedding_o,
            fps_idx_o,
            radius,
            end_points,
        )

        return end_points

    def get_batch_lrf(self, pts):
        # pts: B*N*3
        centroids = torch.mean(pts, 1, True)  # [B, 1, 3]
        if self.use_ref_rad:
            r_lrf = torch.ones(pts.shape[0], device=pts.device)
        else:
            # max_pts = torch.max(pts, 1, False).values # [B, 3]
            # min_pts = torch.min(pts, 1, False).values # [B, 3]
            # r_lrf = torch.norm(max_pts - min_pts, dim=1) / 2.0 # [B]
            pts_minus_mean = pts - centroids
            r_lrf = torch.norm(pts_minus_mean, dim=2).max(1)[0]

        batch_lrf = LRF(r_lrf)
        pts_lrf = batch_lrf(centroids.transpose(1, 2), pts.transpose(1, 2))
        pts_lrf = pts_lrf.transpose(1, 2).contiguous()
        return pts_lrf
