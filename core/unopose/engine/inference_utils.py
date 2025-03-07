from tqdm import tqdm
import torch
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


# example, no use
def inference_and_save(
    model,
    data_loader,
    save_path,
    instance_batch_size=16,
):
    model.eval()

    # prepare for target objects
    all_tem, all_tem_pts, all_tem_choose = data_loader.dataset.get_templates()
    with torch.no_grad():
        dense_po, dense_fo = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

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
                obj = data["obj"][0][start_idx:end_idx].reshape(-1)

                # process inputs
                inputs = {}
                inputs["pts"] = data["pts"][0][start_idx:end_idx].contiguous()
                inputs["rgb"] = data["rgb"][0][start_idx:end_idx].contiguous()
                inputs["rgb_choose"] = data["rgb_choose"][0][start_idx:end_idx].contiguous()
                inputs["model"] = data["model"][0][start_idx:end_idx].contiguous()
                inputs["dense_po"] = dense_po[obj].contiguous()
                inputs["dense_fo"] = dense_fo[obj].contiguous()

                # make predictions
                with torch.no_grad():
                    end_points = model(inputs)
                pred_Rs.append(end_points["pred_R"])
                pred_Ts.append(end_points["pred_t"])
                pred_scores.append(end_points["pred_pose_score"])

            pred_Rs = torch.cat(pred_Rs, dim=0).reshape(-1, 9).detach().cpu().numpy()
            pred_Ts = torch.cat(pred_Ts, dim=0).detach().cpu().numpy() * 1000
            pred_scores = torch.cat(pred_scores, dim=0) * data["score"][0, :, 0]
            pred_scores = pred_scores.detach().cpu().numpy()
            image_time = time.time() - end

            # write results
            scene_id = data["scene_id"].item()
            img_id = data["img_id"].item()
            image_time += data["seg_time"].item()
            for k in range(n_instance):
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
