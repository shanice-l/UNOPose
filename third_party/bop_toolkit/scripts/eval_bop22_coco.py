import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from bop_toolkit_lib import pycoco_utils
import argparse

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    # Names of files with detection results for which to calculate the Average Precisions
    # (assumed to be stored in folder p['results_path']).
    "result_filenames": [
        "json/file/with/coco/results",
    ],
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
    # Folder with BOP datasets.
    "datasets_path": config.datasets_path,
    # Annotation type that should be evaluated. Can be 'segm' or 'bbox'.
    "ann_type": "segm",
    # bbox type. Options: 'modal', 'amodal'.
    "bbox_type": "amodal",
    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    "targets_filename": "test_targets_bop19.json",
}
################################################################################

# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--result_filenames", default=",".join(p["result_filenames"]), help="Comma-separated names of files with results."
)
parser.add_argument("--results_path", default=p["results_path"])
parser.add_argument("--eval_path", default=p["eval_path"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--ann_type", default=p["ann_type"])
parser.add_argument("--bbox_type", default=p["bbox_type"])
args = parser.parse_args()

p["result_filenames"] = args.result_filenames.split(",")
p["results_path"] = str(args.results_path)
p["eval_path"] = str(args.eval_path)
p["targets_filename"] = str(args.targets_filename)
p["ann_type"] = str(args.ann_type)
p["bbox_type"] = str(args.bbox_type)

# Evaluation.
# ------------------------------------------------------------------------------
for result_filename in p["result_filenames"]:
    misc.log("===========")
    misc.log("EVALUATING: {}".format(result_filename))
    misc.log("===========")

    # Parse info about the method and the dataset from the filename.
    result_name = os.path.splitext(os.path.basename(result_filename))[0]
    result_info = result_name.split("_")
    method = str(result_info[0])
    dataset_info = result_info[1].split("-")
    dataset = str(dataset_info[0])
    split = str(dataset_info[1])
    split_type = str(dataset_info[2]) if len(dataset_info) > 2 else None

    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(p["datasets_path"], dataset, split, split_type)

    model_type = "eval"
    dp_model = dataset_params.get_model_params(p["datasets_path"], dataset, model_type)

    # Checking coco result file
    check_passed, _ = inout.check_coco_results(os.path.join(p["results_path"], result_filename), ann_type=p["ann_type"])
    if not check_passed:
        misc.log("Please correct the coco result format of {}".format(result_filename))
        exit()

    # Load coco resultsZ
    misc.log("Loading coco results...")
    coco_results = inout.load_json(os.path.join(p["results_path"], result_filename), keys_to_int=True)

    # Load the estimation targets.
    targets = inout.load_json(os.path.join(dp_split["base_path"], p["targets_filename"]))

    # Organize the targets by scene and image.
    misc.log("Organizing estimation targets...")
    targets_org = {}
    for target in targets:
        targets_org.setdefault(target["scene_id"], {}).setdefault(target["im_id"], {})

    # Organize the results by scene.
    misc.log("Organizing estimation results...")
    results_org = {}
    for result in coco_results:
        if (p["ann_type"] == "bbox" and result["bbox"]) or (p["ann_type"] == "segm" and result["segmentation"]):
            results_org.setdefault(result["scene_id"], []).append(result)

    if not results_org:
        misc.log("No valid coco results for annotation type: {}".format(p["ann_type"]))

    misc.log("Merging coco annotations and predictions...")
    # Merge coco scene annotations and results
    for i, scene_id in enumerate(targets_org):
        scene_coco_ann_path = dp_split["scene_gt_coco_tpath"].format(scene_id=scene_id)
        if p["ann_type"] == "bbox" and p["bbox_type"] == "modal":
            scene_coco_ann_path = scene_coco_ann_path.replace("scene_gt_coco", "scene_gt_coco_modal")
        scene_coco_ann = inout.load_json(scene_coco_ann_path, keys_to_int=True)
        scene_coco_results = results_org[scene_id] if scene_id in results_org else []

        # filter target image ids
        target_img_ids = targets_org[scene_id].keys()
        scene_coco_ann["images"] = [img for img in scene_coco_ann["images"] if img["id"] in target_img_ids]
        scene_coco_ann["annotations"] = [
            ann for ann in scene_coco_ann["annotations"] if ann["image_id"] in target_img_ids
        ]
        scene_coco_results = [res for res in scene_coco_results if res["image_id"] in target_img_ids]

        if i == 0:
            dataset_coco_ann = scene_coco_ann
            dataset_coco_results = scene_coco_results
        else:
            dataset_coco_ann, image_id_offset = pycoco_utils.merge_coco_annotations(dataset_coco_ann, scene_coco_ann)
            dataset_coco_results = pycoco_utils.merge_coco_results(
                dataset_coco_results, scene_coco_results, image_id_offset
            )

    # initialize COCO ground truth api
    cocoGt = COCO(dataset_coco_ann)
    cocoDt = cocoGt.loadRes(dataset_coco_results)

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, p["ann_type"])
    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    res_type = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    coco_scores = {res_type[i]: stat for i, stat in enumerate(cocoEval.stats)}

    # Calculate the average estimation time per image.
    times = {}
    times_available = True
    for result in coco_results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["image_id"])
        if result["time"] < 0:
            # All estimation times must be provided.
            times_available = False
            break
        elif result_key in times:
            if abs(times[result_key] - result["time"]) > 0.001:
                raise ValueError(
                    "The running time for scene {} and image {} is not the same for "
                    "all estimates.".format(result["scene_id"], result["image_id"])
                )
        else:
            times[result_key] = result["time"]

    if times_available:
        coco_scores["average_time_per_image"] = np.mean(list(times.values()))
    else:
        coco_scores["average_time_per_image"] = -1.0

    # Save the final scores.
    os.makedirs(os.path.join(p["eval_path"], result_name), exist_ok=True)
    final_scores_path = os.path.join(p["eval_path"], result_name, "scores_bop22_coco_{}.json".format(p["ann_type"]))
    if p["ann_type"] == "bbox" and p["bbox_type"] == "modal":
        final_scores_path = final_scores_path.replace(".json", "_modal.json")
    inout.save_json(final_scores_path, coco_scores)
