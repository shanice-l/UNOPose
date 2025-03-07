import os
import os.path as osp
import logging

cur_dir = osp.abspath(osp.dirname(__file__))
import torch
import torch.nn as nn
from torch.nn import functional as F

# import torch.utils.model_zoo as model_zoo
from functools import partial
import timm.models.vision_transformer

from core.unopose.utils.model_utils import (
    LayerNorm2d,
    interpolate_pos_embed,
    get_chosen_pixel_feats,
    sample_pts_feats,
)

logger = logging.getLogger(__name__)


class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        out = []
        d = len(self.blocks)
        n = d // 4
        idx_nblock = [d - 1, d - n - 1, d - 2 * n - 1, d - 3 * n - 1]

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in idx_nblock:
                out.append(self.norm(x))
        return out


class ViT_AE(nn.Module):
    def __init__(
        self,
        cfg,
    ) -> None:
        super(ViT_AE, self).__init__()
        self.cfg = cfg
        self.vit_type = cfg.vit_type
        self.up_type = cfg.up_type
        self.embed_dim = cfg.embed_dim
        self.out_dim = cfg.out_dim
        self.use_pyramid_feat = cfg.use_pyramid_feat
        self.pretrained = cfg.pretrained
        self.freeze_vit = cfg.get("freeze_vit", False)

        self.vit_ckpt = cfg.vit_ckpt
        self.patch_size = 14 if "patch14" in self.vit_type else 16
        self.img_size = 224
        self.patch_num_side = self.img_size // self.patch_size
        logger.info(f"{self.vit_type}")

        # NOTE: use img_size 224
        if self.vit_type == "vit_base":
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        elif self.vit_type == "vit_large":
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        elif self.vit_type == "vit_small_patch14_dinov2":
            assert self.embed_dim == 384
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,  # 384,
                depth=12,
                num_heads=6,
                init_values=1e-5,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        elif self.vit_type == "vit_base_patch14_dinov2":
            assert self.embed_dim == 768
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,  # 768,
                depth=12,
                num_heads=12,
                init_values=1e-5,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        elif self.vit_type == "vit_large_patch14_dinov2":
            assert self.embed_dim == 1024
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=24,
                num_heads=16,
                init_values=1e-5,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        elif self.vit_type == "vit_small_patch14_reg4_dinov2":
            assert self.embed_dim == 384
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,  # 384,
                depth=12,
                num_heads=6,
                init_values=1e-5,
                reg_tokens=4,
                no_embed_class=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        elif self.vit_type == "vit_base_patch14_reg4_dinov2":
            assert self.embed_dim == 768
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,  # 768,
                depth=12,
                num_heads=12,
                init_values=1e-5,
                reg_tokens=4,
                no_embed_class=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        elif self.vit_type == "vit_large_patch14_reg4_dinov2":
            assert self.embed_dim == 1024
            self.vit = ViT(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=24,
                num_heads=16,
                init_values=1e-5,
                reg_tokens=4,
                no_embed_class=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        else:
            assert False

        if self.use_pyramid_feat:
            nblock = 4
        else:
            nblock = 1

        if self.up_type == "linear":
            self.output_upscaling = nn.Linear(self.embed_dim * nblock, 16 * self.out_dim, bias=True)
        elif self.up_type == "deconv":
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim * nblock, self.out_dim * 2, kernel_size=2, stride=2),
                LayerNorm2d(self.out_dim * 2),
                nn.GELU(),
                nn.ConvTranspose2d(self.out_dim * 2, self.out_dim, kernel_size=2, stride=2),
            )
        else:
            assert False

        if self.pretrained:
            # ckpt_dir = os.path.join(cur_dir, "../../../checkpoints")
            # vit_checkpoint = osp.join(ckpt_dir, "mae_pretrain_" + self.vit_type + ".pth")
            # if not osp.exists(vit_checkpoint):
            #     os.makedirs(ckpt_dir, exist_ok=True)
            #     model_zoo.load_url(
            #         "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_" + self.vit_type + ".pth", "checkpoints"
            #     )
            assert osp.exists(self.vit_ckpt), self.vit_ckpt
            checkpoint = torch.load(self.vit_ckpt, map_location="cpu")
            logger.info(f"load pre-trained checkpoint from: {self.vit_ckpt}")
            checkpoint_model = checkpoint["model"]
            state_dict = self.vit.state_dict()
            for k in ["head.weight", "head.bias"]:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    logger.warning(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(self.vit, checkpoint_model)
            msg = self.vit.load_state_dict(checkpoint_model, strict=False)

        if self.freeze_vit:
            assert self.pretrained, "freeze must use pretrained!"
            logger.warning("freeze vit")
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, x):
        B, _, H, W = x.size()
        vit_outs = self.vit(x)
        # list of [B, token_num, emb_dim]
        cls_tokens = vit_outs[-1][:, 0, :].contiguous()  # [B, emb_dim]
        if "reg4" in self.vit_type:
            # cls_token, reg_tokens, patch_tokens
            vit_outs = [l[:, 5:, :].contiguous() for l in vit_outs]
        else:
            vit_outs = [l[:, 1:, :].contiguous() for l in vit_outs]

        if self.use_pyramid_feat:
            x = torch.cat(vit_outs, dim=2)
        else:
            x = vit_outs[-1]
        # [B, token_num, emb_dim*block_num]

        # import ipdb; ipdb.set_trace()
        if self.up_type == "linear":
            # [B, token_num, 16*out_dim]  --> [B, sqrt(token_num), sqrt(token_num), 4, 4, out_dim]
            # --> [B, out_dim, sqrt(token_num), 4, sqrt(token_num), 4]
            x = (
                self.output_upscaling(x)
                .reshape(B, self.patch_num_side, self.patch_num_side, 4, 4, self.out_dim)
                .permute(0, 5, 1, 3, 2, 4)
                .contiguous()
            )
            # [B, out_dim, sqrt(token_num)*4, sqrt(token_num)*4]
            x = x.reshape(B, -1, 4 * self.patch_num_side, 4 * self.patch_num_side)
            x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        elif self.up_type == "deconv":
            # [B, emb_dim*block_num, token_num] --> [B, emb_dim*block_num, sqrt(token_num), sqrt(token_num)]
            x = x.transpose(1, 2).reshape(B, -1, self.patch_num_side, self.patch_num_side)
            x = self.output_upscaling(x)  # [B, out_dim, sqrt(token_num)*4, sqrt(token_num)*4]
            x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        # import ipdb; ipdb.set_trace()
        return x, cls_tokens


class ViTEncoderOneRef(nn.Module):
    def __init__(self, cfg, npoint=None):
        super(ViTEncoderOneRef, self).__init__()
        self.npoint = npoint
        self.rgb_net = ViT_AE(cfg)

    def forward(self, end_points):
        rgb = end_points["rgb"]
        rgb_choose = end_points["rgb_choose"]
        dense_fm = self.get_img_feats(rgb, rgb_choose)
        dense_pm = end_points["pts"]
        assert rgb_choose.size(1) == self.npoint

        if not self.training and "dense_po" in end_points.keys() and "dense_fo" in end_points.keys():
            dense_po = end_points["dense_po"].clone()
            dense_fo = end_points["dense_fo"].clone()

            # normalize point clouds
            # radius = torch.norm(dense_po, dim=2).max(1)[0]
            tem1_mean_point = torch.mean(dense_po, dim=1, keepdim=True)
            tem1_pts_minus_mean = dense_po - tem1_mean_point
            radius = torch.norm(tem1_pts_minus_mean, dim=2).max(1)[0]

            dense_pm = dense_pm / (radius.reshape(-1, 1, 1) + 1e-6)
            dense_po = dense_po / (radius.reshape(-1, 1, 1) + 1e-6)
        else:  # train / test where template is obtained in dataset
            tem1_rgb = end_points["tem1_rgb"]  # b,h,w,3
            tem1_choose = end_points["tem1_choose"]
            tem1_pts = end_points["tem1_pts"]

            # normalize point clouds
            # dense_po = tem1_pts  # b, p, 3
            # radius = torch.norm(dense_po, dim=2).max(1)[0]  # should be in object space
            tem1_mean_point = torch.mean(tem1_pts, dim=1, keepdim=True)
            tem1_pts_minus_mean = tem1_pts - tem1_mean_point
            radius = torch.norm(tem1_pts_minus_mean, dim=2).max(1)[0]

            dense_pm = dense_pm / (radius.reshape(-1, 1, 1) + 1e-6)
            tem1_pts = tem1_pts / (radius.reshape(-1, 1, 1) + 1e-6)

            dense_po, dense_fo = self.get_obj_feats([tem1_rgb], [tem1_pts], [tem1_choose])
        # target(observed), src (ref/template/obj)
        # [b, p, 3]; [b, p, c]
        return dense_pm, dense_fm, dense_po, dense_fo, radius

    def get_img_feats(self, img, choose):
        return get_chosen_pixel_feats(self.rgb_net(img)[0], choose)

    def get_obj_feats(self, tem_rgb_list, tem_pts_list, tem_choose_list, npoint=None):
        if npoint is None:
            npoint = self.npoint

        tem_feat_list = []
        for tem, tem_choose in zip(tem_rgb_list, tem_choose_list):
            # (n_obj, p, c)
            tem_feat_list.append(self.get_img_feats(tem, tem_choose))

        tem_pts = torch.cat(tem_pts_list, dim=1)
        tem_feat = torch.cat(tem_feat_list, dim=1)  # (n_obj, n_tem x p, c)
        return sample_pts_feats(tem_pts, tem_feat, npoint)


if __name__ == "__main__":
    from pathlib import Path
    from easydict import EasyDict as edict
    import timm
    from tqdm import tqdm

    PROJ_ROOT = Path(__file__).parent.parent.parent.parent

    # model = timm.create_model("vit_large_patch14_dinov2", pretrained=True)
    # torch.save(model.state_dict(), PROJ_ROOT / "checkpoints/timm_vit_large_patch14_dinov2_lvd142m.pth")

    cfg = edict(
        vit_type="vit_base",
        up_type="linear",
        embed_dim=768,
        out_dim=256,
        use_pyramid_feat=True,
        pretrained=True,
        vit_ckpt=osp.join(str(PROJ_ROOT), "checkpoints/mae_pretrain_vit_base.pth"),
    )
    # cfg = edict(
    #     vit_type="vit_base_patch14_dinov2",
    #     up_type="linear",
    #     embed_dim=768,
    #     out_dim=256,
    #     use_pyramid_feat=True,
    #     pretrained=True,
    #     vit_ckpt=osp.join(str(PROJ_ROOT), "checkpoints/timm_vit_base_patch14_dinov2_lvd142m.pth"),
    # )
    # cfg = edict(
    #     vit_type="vit_base_patch14_reg4_dinov2",
    #     up_type="linear",
    #     embed_dim=768,
    #     out_dim=256,
    #     use_pyramid_feat=True,
    #     pretrained=True,
    #     vit_ckpt=osp.join(str(PROJ_ROOT), "checkpoints/timm_vit_base_patch14_reg4_dinov2_lvd142m.pth"),
    # )

    # cfg = edict(
    #     vit_type="vit_large_patch14_dinov2",
    #     up_type="linear",
    #     embed_dim=1024,
    #     out_dim=256,
    #     use_pyramid_feat=True,
    #     pretrained=True,
    #     vit_ckpt=osp.join(str(PROJ_ROOT), "checkpoints/timm_vit_large_patch14_dinov2_lvd142m.pth"),
    # )

    vit_ae = ViT_AE(cfg)
    vit_ae.to("cuda")

    im = torch.rand(1, 3, 224, 224).float().cuda()
    for i in tqdm(range(2000)):
        outputs = vit_ae(im)
    print(outputs[0].shape)
    print(outputs[1].shape)
