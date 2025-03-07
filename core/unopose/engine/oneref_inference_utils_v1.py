from copy import deepcopy
from tqdm import tqdm
import torch
import time
import numpy as np
import logging
import orjson as json
from pathlib import Path

logger = logging.getLogger(__name__)


def inference_and_save_oneref_v1(
    model,
    data_loader,
    save_path,
    instance_batch_size=16,
):
    model.eval()

    # prepare for target objects
    # n_tem_view x n_obj
    # all_tem[tem_id][obj_idx]
    # all_tem, all_tem_pts, all_tem_choose = data_loader.dataset.get_templates()
    # TODO: move this into batch step
    # given a batch obj_ids, tem_ids, get the rgb, pts, choose for a batch of instances
    # with torch.no_grad():
    #     dense_po, dense_fo = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    # pred poses to be added below
    dets = deepcopy(data_loader.dataset.dets)

    bs = instance_batch_size
    lines = []
    with tqdm(total=len(data_loader)) as pbar:
        for i, data in enumerate(data_loader):
            torch.cuda.synchronize()
            end = time.perf_counter()

            for key in data:
                data[key] = data[key].cuda()
            n_instance = data["pts"].size(1)
            n_batch = int(np.ceil(n_instance / bs))

            pred_Rs = []
            pred_Ts = []
            pred_scores = []
            for j in range(n_batch):
                start_idx = j * bs
                end_idx = n_instance if j == n_batch - 1 else (j + 1) * bs
                # test image bs should be 1, so img_idx is always 0
                # obj = data["obj"][0][start_idx:end_idx].reshape(-1)

                # process inputs
                inputs = {}
                inputs["pts"] = data["pts"][0][start_idx:end_idx].contiguous()
                inputs["rgb"] = data["rgb"][0][start_idx:end_idx].contiguous()
                inputs["rgb_choose"] = data["rgb_choose"][0][start_idx:end_idx].contiguous()
                if "fps_idx_m" in data:
                    inputs["fps_idx_m"] = data["fps_idx_m"][0][start_idx:end_idx].contiguous()

                inputs["tem1_rgb"] = data["tem1_rgb"][0][start_idx:end_idx].contiguous()
                inputs["tem1_choose"] = data["tem1_choose"][0][start_idx:end_idx].contiguous()
                inputs["tem1_pts"] = data["tem1_pts"][0][start_idx:end_idx].contiguous()
                if "fps_idx_o" in data:
                    inputs["fps_idx_o"] = data["fps_idx_o"][0][start_idx:end_idx].contiguous()

                # tem_id = data["tem_id"][0][start_idx:end_idx].reshape(-1)
                # tem_rgb = [all_tem[tem_id[k]][obj[k]] for k in range(len(tem_id))]
                # tem_pts = [all_tem_pts[tem_id[k]][obj[k]] for k in range(len(tem_id))]
                # tem_choose = [all_tem_choose[tem_id[k]][obj[k]] for k in range(len(tem_id))]
                # inputs["tem1_rgb"] = torch.stack(tem_rgb).contiguous()
                # inputs["tem1_pts"] = torch.stack(tem_pts).contiguous()
                # inputs["tem1_choose"] = torch.stack(tem_choose).contiguous()

                # inputs["model"] = data["model"][0][start_idx:end_idx].contiguous()
                # inputs["dense_po"] = dense_po[obj].contiguous()
                # inputs["dense_fo"] = dense_fo[obj].contiguous()

                # make predictions
                with torch.no_grad():
                    end_points = model(inputs)
                if "tem1_pose" in data:
                    pose_ref_obj = data["tem1_pose"][0][start_idx:end_idx].contiguous()
                    predpose_tgt_ref = torch.zeros_like(pose_ref_obj)
                    predpose_tgt_ref[:, 3, 3] = 1.0
                    predpose_tgt_ref[:, :3, :3] = end_points["pred_R"]
                    predpose_tgt_ref[:, :3, 3] = end_points["pred_t"]
                    predpose_tgt_obj = predpose_tgt_ref @ pose_ref_obj
                    pred_Rs.append(predpose_tgt_obj[:, :3, :3])
                    pred_Ts.append(predpose_tgt_obj[:, :3, 3])
                else:
                    pred_Rs.append(end_points["pred_R"])
                    pred_Ts.append(end_points["pred_t"])
                pred_scores.append(end_points["pred_pose_score"])

            pred_Rs = torch.cat(pred_Rs, dim=0).reshape(-1, 9).detach().cpu().numpy()
            pred_Ts = torch.cat(pred_Ts, dim=0).detach().cpu().numpy() * 1000
            pred_scores = torch.cat(pred_scores, dim=0) * data["score"][0, :, 0]
            pred_scores = pred_scores.detach().cpu().numpy()
            image_time = time.perf_counter() - end

            # write results
            scene_id = data["scene_id"].item()
            img_id = data["img_id"].item()
            det_key = f"{scene_id:06d}_{img_id:06d}"
            inst_ids = data["inst_ids"][0].cpu().numpy()
            image_time += data["seg_time"].item()
            # image_time = data["seg_time"].item()
            for k in range(n_instance):
                inst_i = int(inst_ids[k])
                dets[det_key][inst_i]["pred_R"] = pred_Rs[k].tolist()
                dets[det_key][inst_i]["pred_t"] = pred_Ts[k].tolist()
                line = ",".join(
                    (
                        str(scene_id),
                        str(img_id),
                        str(data["obj_id"][0][k].item()),
                        str(pred_scores[k]),
                        " ".join((str(v) for v in pred_Rs[k])),
                        " ".join((str(v) for v in pred_Ts[k])),
                        f"{image_time}\n",
                    )
                )
                lines.append(line)

            pbar.set_description("Test [{}/{}]".format(i + 1, len(data_loader)))
            pbar.update(1)

    with open(save_path, "w+") as f:
        f.writelines(lines)
    logger.info(f"saved to {save_path}")

    save_json_path = save_path.replace(".csv", ".json")
    Path(save_json_path).write_bytes(json.dumps(dets))
    logger.info(f"json saved to {save_json_path}")
