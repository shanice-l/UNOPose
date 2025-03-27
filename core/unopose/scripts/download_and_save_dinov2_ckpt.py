from pathlib import Path
import timm
import torch
from tqdm import tqdm

PROJ_ROOT = Path(__file__).parent.parent.parent.parent

# name = "vit_large_patch14_dinov2"
for name in tqdm(
    [
        # "vit_small_patch14_dinov2",
        # "vit_small_patch14_reg4_dinov2",
        # "vit_base_patch14_dinov2",
        "vit_base_patch14_reg4_dinov2",
        # "vit_large_patch14_reg4_dinov2",  # "vit_large_patch14_dinov2"
        # "vit_giant_patch14_dinov2",
        # "vit_giant_patch14_reg4_dinov2",
    ]
):
    print(name)
    model = timm.create_model(name, pretrained=True)
    data = {"model": model.state_dict()}
    save_path = PROJ_ROOT / f"checkpoints/timm_{name}_lvd142m.pth"
    torch.save(data, save_path)
    print(save_path)
