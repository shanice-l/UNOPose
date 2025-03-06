import logging
import os
import os.path as osp
import sys
import subprocess
import time

import mmengine
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

cur_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))
sys.path.insert(0, osp.join(cur_dir, "../../../third_party/bop_toolkit/"))
import ref
from bop_toolkit_lib import misc


logger = logging.getLogger(__name__)

SPLIT_STR = "="


def _to_str(item):
    if isinstance(item, (list, tuple)):
        return " ".join(["{}".format(e) for e in item])
    else:
        return "{}".format(item)


def to_list(array):
    return array.flatten().tolist()


def save_and_eval_results(val_cfg, results_all, output_dir, obj_ids=None, exp_id=""):
    save_root = output_dir  # eval_path
    split_type_str = f"-{val_cfg.split_type}" if val_cfg.split_type != "" else ""
    mmengine.mkdir_or_exist(save_root)
    header = "scene_id,im_id,obj_id,score,R,t,time"
    keys = header.split(",")
    result_names = []
    for name, result_list in results_all.items():
        method_name = f"{exp_id.replace('_', '-')}-{name}"
        result_name = f"{method_name}_{val_cfg.dataset_name}-{val_cfg.split}{split_type_str}.csv"
        res_path = osp.join(save_root, result_name)
        result_names.append(result_name)
        with open(res_path, "w") as f:
            f.write(header + "\n")
            for line_i, result in enumerate(result_list):
                items = []
                for res_k in keys:
                    items.append(_to_str(result[res_k]))
                f.write(",".join(items) + "\n")
        logger.info("wrote results to: {}".format(res_path))

    if not val_cfg.save_bop_csv_only:
        result_names_str = ",".join(result_names)
        eval_cmd = [
            "python",
            val_cfg.script_path,
            "--results_path={}".format(save_root),
            "--result_filenames={}".format(result_names_str),
            "--renderer_type={}".format(val_cfg.renderer_type),
            "--error_types={}".format(val_cfg.error_types),
            "--eval_path={}".format(save_root),
            "--targets_filename={}".format(val_cfg.targets_filename),
            "--n_top={}".format(val_cfg.n_top),
        ]
        if val_cfg.score_only:
            eval_cmd += ["--score_only"]
        eval_time = time.perf_counter()
        if subprocess.call(eval_cmd) != 0:
            logger.warning("evaluation failed.")

        load_and_print_val_scores_tab(
            val_cfg,
            eval_root=save_root,
            result_names=result_names,
            error_types=val_cfg.error_types.split(","),
            obj_ids=obj_ids,
        )
        logger.info("eval time: {}s".format(time.perf_counter() - eval_time))


def eval_cached_results(val_cfg, output_dir, obj_ids=None, exp_id="", n_iter_test=4):
    logger.info("eval cached results")
    split_type_str = f"-{val_cfg.split_type}" if val_cfg.split_type != "" else ""
    save_root = output_dir  # eval_path
    assert osp.exists(save_root), save_root
    result_names = []
    names = ["iter{}".format(i) for i in range(n_iter_test + 1)]
    # print('exp_id', exp_id)
    for name in names:
        method_name = "{}-{}".format(exp_id.replace("_", "-"), name)
        result_name = f"{method_name}_{val_cfg.dataset_name}-{val_cfg.split}{split_type_str}.csv"
        res_path = osp.join(save_root, result_name)
        if not osp.exists(res_path):
            if exp_id.endswith("_test"):
                method_name = "{}-{}".format(exp_id.replace("_test", "").replace("_", "-"), name)
                result_name = f"{method_name}_{val_cfg.dataset_name}-{val_cfg.split}{split_type_str}.csv"
                res_path = osp.join(save_root, result_name)
        assert osp.exists(res_path), res_path
        result_names.append(result_name)
    try:
        if not val_cfg.eval_print_only:
            raise RuntimeError()
        load_and_print_val_scores_tab(
            cfg,
            eval_root=save_root,
            result_names=result_names,
            error_types=val_cfg.error_types.split(","),
            obj_ids=obj_ids,
        )
    except:
        result_names_str = ",".join(result_names)
        eval_cmd = [
            "python",
            val_cfg.script_path,
            "--results_path={}".format(save_root),
            "--result_filenames={}".format(result_names_str),
            "--renderer_type={}".format(val_cfg.renderer_type),
            "--error_types={}".format(val_cfg.error_types),
            "--eval_path={}".format(save_root),
            "--targets_filename={}".format(val_cfg.targets_filename),
            "--n_top={}".format(val_cfg.N_TOP),
        ]
        if val_cfg.score_only:
            eval_cmd += ["--score_only"]
        eval_time = time.perf_counter()
        if subprocess.call(eval_cmd) != 0:
            logger.warning("evaluation failed.")

        load_and_print_val_scores_tab(
            cfg,
            eval_root=save_root,
            result_names=result_names,
            error_types=val_cfg.error_types.split(","),
            obj_ids=obj_ids,
        )
        logger.info("eval time: {}s".format(time.perf_counter() - eval_time))
    exit(0)


def get_data_ref(dataset_name):
    ref_key_dict = {
        "lm": "lm_full",
        "lmo": "lmo",
        "lmo_full": "lmo",
        "ycbv": "ycbv",
        "ycbvposecnn": "ycbv",
        "tless": "tless",
        "tudl": "tudl",
    }
    ref_key = ref_key_dict[dataset_name]
    return ref.__dict__[ref_key]()


def get_thr(score_path):
    # used for sorting score json files
    # scores_th:2.000_min-visib:0.100.json
    # rete: scores_th:10.000-10.000_min-visib:-1.000.json
    # NOTE: assume the same threshold (currently can deal with rete, rete_s)
    return float(score_path.split("/")[-1].replace(f"scores_th{SPLIT_STR}", "").split("_")[0].split("-")[0])


def simplify_float_str(float_str):
    value = float(float_str)
    if value == int(value):
        return str(int(value))
    return float_str


def get_thr_str(score_path):
    # path/to/scores_th:2.000_min-visib:0.100.json  --> 2
    # rete: path/to/scores_th:10.000-10.000_min-visib:-1.000.json --> 10
    thr_str = score_path.split("/")[-1].split("_")[1]
    thr_str = thr_str.split(SPLIT_STR)[-1]
    if "-" in thr_str:
        thr_str_split = thr_str.split("-")
        simple_str_list = [simplify_float_str(_thr) for _thr in thr_str_split]
        if len(set(simple_str_list)) == 1:
            res_thr_str = simple_str_list[0]
        else:
            res_thr_str = "-".join(simple_str_list)
    else:
        res_thr_str = simplify_float_str(thr_str)
    return res_thr_str


def is_auc_metric(error_type):
    if error_type in ["AUCadd", "AUCadi", "AUCad", "vsd", "mssd", "mspd"]:
        return True
    return False


def is_weighted_average_metric(error_type):
    if error_type in ["mspd", "mssd", "vsd"]:
        return True
    return False


def get_object_nums_from_targets(targets_path):
    """stat the number of each object given a targets json file in BOP
    format."""
    assert osp.exists(targets_path), targets_path
    targets = mmengine.load(targets_path)

    obj_nums_dict = {}
    for target in targets:
        obj_id = target["obj_id"]
        if obj_id not in obj_nums_dict:
            obj_nums_dict[obj_id] = 0
        obj_nums_dict[obj_id] += target["inst_count"]
    res_obj_nums_dict = {str(key): obj_nums_dict[key] for key in sorted(obj_nums_dict.keys())}
    return res_obj_nums_dict


def summary_scores(
    score_paths,
    error_type,
    val_dataset_name,
    print_all_objs=False,
    obj_ids=None,
    obj_nums_dict=None,
):
    data_ref = get_data_ref(val_dataset_name)

    sorted_score_paths = sorted(score_paths.keys(), key=get_thr)

    min_max_thr_str = None
    obj_recalls_dict = {}
    if is_auc_metric(error_type):
        min_thr_str = get_thr_str(sorted_score_paths[0])
        max_thr_str = get_thr_str(sorted_score_paths[-1])
        min_max_thr_str = f"{min_thr_str}:{max_thr_str}"

    tabs_col2 = []
    for score_path in sorted_score_paths:
        score_dict = mmengine.load(score_path)
        if obj_ids is None:
            sel_obj_ids = [int(_id) for _id in score_dict["obj_recalls"].keys()]
        else:
            sel_obj_ids = obj_ids

        thr_str = get_thr_str(score_path)
        # logging the results with tabulate
        # tab_header = ["objects", "{}[{}](%)".format(error_type, thr_str)]
        tab_header = [
            "objects",
            "{}_{}".format(error_type, thr_str),
        ]  # 2 columns, objs in col
        cur_tab_col2 = [tab_header]
        for _id, _recall in score_dict["obj_recalls"].items():
            obj_name = data_ref.id2obj[int(_id)]
            if int(_id) in sel_obj_ids:
                cur_tab_col2.append([obj_name, f"{_recall * 100:.2f}"])
                if min_max_thr_str is not None:  # for AUC metrics
                    if obj_name not in obj_recalls_dict:
                        obj_recalls_dict[obj_name] = []
                    obj_recalls_dict[obj_name].append(_recall)
            else:
                if print_all_objs:
                    cur_tab_col2.append([obj_name, "-"])

        # mean of selected objs
        num_objs = len(sel_obj_ids)
        if num_objs > 1:
            sel_obj_recalls = [_recall for _id, _recall in score_dict["obj_recalls"].items() if int(_id) in sel_obj_ids]
            if not is_weighted_average_metric(error_type):
                mean_obj_recall = np.mean(sel_obj_recalls)
            else:
                assert obj_nums_dict is not None
                sel_obj_nums = np.array([_v for _k, _v in obj_nums_dict.items() if int(_k) in sel_obj_ids])
                sel_obj_weights = sel_obj_nums / sum(sel_obj_nums)
                mean_obj_recall = sum(sel_obj_weights * np.array(sel_obj_recalls))
            cur_tab_col2.append(["Avg({})".format(num_objs), f"{mean_obj_recall * 100:.2f}"])

        cur_tab_col2 = np.array(cur_tab_col2)
        tabs_col2.append(cur_tab_col2)

    if len(tabs_col2) == 1:
        return tabs_col2[0]
    else:
        if min_max_thr_str is None:  # not AUC metrics, concat
            res_tab = np.concatenate(
                [tabs_col2[0]] + [_tab[:, 1:2] for _tab in tabs_col2[1:]],
                axis=1,
            )
        else:  # AUC metrics, mean
            auc_header = [
                "objects",
                "{}_{}".format(error_type, min_max_thr_str),
            ]  # 2 columns, objs in col
            res_tab = [auc_header]
            obj_aucs = []
            obj_nums = []
            for obj_name in tabs_col2[0][1:-1, 0].tolist():
                if obj_name in obj_recalls_dict:
                    cur_auc = np.mean(obj_recalls_dict[obj_name])
                    obj_aucs.append(cur_auc)
                    if obj_nums_dict is not None:
                        obj_nums.append(obj_nums_dict[str(data_ref.obj2id[obj_name])])
                    res_tab.append([obj_name, f"{cur_auc * 100:.2f}"])
            if is_weighted_average_metric(error_type):
                assert len(obj_nums) == len(obj_aucs), f"{len(obj_nums)} != {len(obj_aucs)}"
                obj_weights = np.array(obj_nums) / sum(obj_nums)
                mean_obj_auc = sum(np.array(obj_aucs) * obj_weights)
            else:
                mean_obj_auc = np.mean(obj_aucs)
            res_tab.append(["Avg({})".format(len(obj_aucs)), f"{mean_obj_auc * 100:.2f}"])
            res_tab = np.array(res_tab)
        return res_tab


def maybe_average_vsd_scores(res_log_tab):
    # obj in row, scores in col
    if "vsd_0.050:0.500" in res_log_tab[:, 0]:
        vsd_rows = [_r for _r in range(res_log_tab.shape[0]) if res_log_tab[_r, 0] == "vsd_0.050:0.500"]
        vsd_mean = np.mean(res_log_tab[vsd_rows, 1:].astype("float32"), 0)
        vsd_mean_row = np.array(
            ["vsd_0.050:0.500"] + [f"{_v:.2f}" for _v in vsd_mean],
            dtype=res_log_tab.dtype,
        )
        new_res_log_tab = []
        vsd_cnt = 0
        for row_i, log_row in enumerate(res_log_tab):
            if row_i not in vsd_rows:
                new_res_log_tab.append(log_row)
            else:
                if vsd_cnt == 0:
                    new_res_log_tab.append(vsd_mean_row)
                vsd_cnt += 1
        new_res_log_tab = np.array(new_res_log_tab)
    else:
        new_res_log_tab = res_log_tab
    return new_res_log_tab


def load_and_print_val_scores_tab(
    val_cfg,
    eval_root,
    result_names,
    error_types=["proj", "ad", "rete"],
    obj_ids=None,
    print_all_objs=False,
):
    vsd_deltas = {
        "hb": 15,
        "hbs": 15,
        "icbin": 15,
        "icmi": 15,
        "itodd": 5,
        "lm": 15,
        "lmo": 15,
        "ruapc": 15,
        "tless": 15,
        "tudl": 15,
        "tyol": 15,
        "ycbv": 15,
        "ycbvposecnn": 15,
    }
    ntop = val_cfg.n_top
    val_dataset_name = val_cfg.dataset_name
    vsd_delta = vsd_deltas[val_dataset_name]
    data_ref = get_data_ref(val_dataset_name)

    if any(is_weighted_average_metric(err_type) for err_type in error_types):
        obj_nums_dict = get_object_nums_from_targets(osp.join(data_ref.dataset_root, val_cfg.targets_filename))
    else:
        obj_nums_dict = None

    vsd_taus = list(np.arange(0.05, 0.51, 0.05))
    # visib_gt_min = 0.1

    for result_name in tqdm(result_names):
        logger.info("=====================================================================")
        big_tab_row = []
        for error_type in error_types:
            result_name = result_name.replace(".csv", "")
            # logger.info(f"************{result_name} *** [{error_type}]*******************")
            if error_type == "vsd":
                error_signs = [
                    misc.get_error_signature(error_type, ntop, vsd_delta=vsd_delta, vsd_tau=vsd_tau)
                    for vsd_tau in vsd_taus
                ]
            else:
                error_signs = [misc.get_error_signature(error_type, ntop)]
            score_roots = [osp.join(eval_root, result_name, error_sign) for error_sign in error_signs]

            for score_root in score_roots:
                if osp.exists(score_root):
                    # get all score json files for this metric under this threshold
                    score_paths = {
                        osp.join(score_root, fn.name): None
                        for fn in os.scandir(score_root)
                        if ".json" in fn.name and "scores" in fn.name
                    }

                    tab_obj_col = summary_scores(
                        score_paths,
                        error_type,
                        val_dataset_name=val_dataset_name,
                        print_all_objs=print_all_objs,
                        obj_ids=obj_ids,
                        obj_nums_dict=obj_nums_dict,
                    )
                    # print single metric with obj in col here
                    logger.info(f"************{result_name} *********************")
                    tab_obj_col_log_str = tabulate(
                        tab_obj_col,
                        tablefmt="plain",
                        # floatfmt=floatfmt
                    )
                    logger.info("\n{}".format(tab_obj_col_log_str))
                    #####
                    big_tab_row.append(tab_obj_col.T)  # objs in row

                else:
                    logger.warning("{} does not exist.".format(score_root))
                    raise RuntimeError("{} does not exist.".format(score_root))

        if len(big_tab_row) > 0:
            # row: obj in row
            # col: obj in col
            logger.info(f"************{result_name} *********************")
            if len(big_tab_row) == 1:
                res_log_tab = big_tab_row[0]
            else:
                res_log_tab = np.concatenate(
                    [big_tab_row[0]] + [_tab[1:, :] for _tab in big_tab_row[1:]],
                    axis=0,
                )

            new_res_log_tab = maybe_average_vsd_scores(res_log_tab)
            new_res_log_tab_col = new_res_log_tab.T

            if len(new_res_log_tab) < len(new_res_log_tab_col):  # print the table with more rows later
                log_tabs = [new_res_log_tab, new_res_log_tab_col]
                suffixes = ["row", "col"]
            else:
                log_tabs = [new_res_log_tab_col, new_res_log_tab]
                suffixes = ["col", "row"]
            for log_tab_i, suffix in zip(log_tabs, suffixes):
                dump_tab_name = osp.join(eval_root, f"{result_name}_tab_obj_{suffix}.txt")
                log_tab_i_str = tabulate(
                    log_tab_i,
                    tablefmt="plain",
                    # floatfmt=floatfmt
                )
                logger.info("\n{}".format(log_tab_i_str))
                with open(dump_tab_name, "w") as f:
                    f.write("{}\n".format(log_tab_i_str))
    logger.info("{}".format(eval_root))


if __name__ == "__main__":
    import argparse
    from detectron2.config import LazyConfig
    from omegaconf import OmegaConf
    from lib.utils.setup_logger import setup_my_logger

    """
    python core/unopose/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types "vsd,mssd,mspd" \
        --split test \
        --dataset ycbv \
        --result_names result_ycbv-test.csv \
        --result_dir output/unopose/base/inference_model_final/ycbv/
    """
    parser = argparse.ArgumentParser(description="wrapper functions to evaluate with bop toolkit")
    parser.add_argument(
        "--script-path",
        default="third_party/bop_toolkit/scripts/eval_bop19_pose.py",
        help="script path to run bop evaluation",
    )

    parser.add_argument("--result_dir", default="", help="result dir")
    # f"{method_name}_{val_cfg.DATASET_NAME}-{val_cfg.SPLIT}{split_type_str}_{other}-{description}.csv"
    parser.add_argument("--result_names", default="", help="result names: a.csv,b.csv,c.csv")

    parser.add_argument("--dataset", default="lmo", help="dataset name")
    parser.add_argument("--split", default="test", help="split")
    parser.add_argument("--split-type", default="", help="split type")  # bb8

    parser.add_argument(
        "--targets_name",
        default="test_targets_bop19.json",
        help="targets filename",
    )
    parser.add_argument("--obj_ids", default=None, help="obj ids to evaluate: 1,2,3,4")
    # "vsd,mssd,mspd"
    parser.add_argument(
        "--n_top",
        default=-1,
        type=int,
        help="top n to be evaluated, VIVO: -1, SISO: 1",
    )
    # parser.add_argument("--error_types", default="ad,reteS,reS,teS,projS", help="error types")
    parser.add_argument("--error_types", default="ad,re,te,proj", help="error types")
    parser.add_argument("--render_type", default="vispy", help="render type: python | cpp | vispy")
    parser.add_argument("--score_only", default=False, action="store_true", help="score only")
    parser.add_argument("--print_only", default=False, action="store_true", help="print only")
    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.obj_ids is not None:
        obj_ids = [int(_e) for _e in args.obj_ids.split(",")]
    else:
        obj_ids = args.obj_ids
    result_dir = args.result_dir
    setup_my_logger(name="core")
    setup_my_logger(name="__main__")
    result_names_str = args.result_names
    if "," not in result_names_str:
        result_names = [result_names_str]
    else:
        result_names = result_names_str.split(",")

    cfg = dict(
        val=dict(
            dataset_name=args.dataset,
            script_path=args.script_path,
            results_path=result_dir,
            targets_filename=args.targets_name,  # 'lm_test_targets_bb8.json'
            error_types=args.error_types,
            renderer_type=args.render_type,  # cpp, python, egl
            split=args.split,
            split_type=args.split_type,
            n_top=args.n_top,  # SISO: 1, VIVO: -1 (for LINEMOD, 1/-1 are the same)
            score_only=args.score_only,  # if the errors have been calculated
            eval_print_only=args.print_only,  # if the scores/recalls have been saved
        )
    )
    cfg = OmegaConf.create(cfg)
    LazyConfig.apply_overrides(cfg, args.opts)
    val_cfg = cfg.val

    eval_time = time.perf_counter()
    if not args.print_only:
        eval_cmd = [
            "python",
            val_cfg.script_path,
            "--results_path={}".format(result_dir),
            "--result_filenames={}".format(result_names_str),
            "--renderer_type={}".format(val_cfg.renderer_type),
            "--error_types={}".format(val_cfg.error_types),
            "--eval_path={}".format(result_dir),
            "--targets_filename={}".format(val_cfg.targets_filename),
            "--n_top={}".format(val_cfg.n_top),
        ]
        if val_cfg.score_only:
            eval_cmd += ["--score_only"]
        if subprocess.call(eval_cmd) != 0:
            logger.warning("evaluation failed.")

    print("print scores")
    load_and_print_val_scores_tab(
        val_cfg,
        eval_root=result_dir,
        result_names=result_names,
        error_types=val_cfg.error_types.split(","),
        obj_ids=obj_ids,
    )
    logger.info("eval time: {}s".format(time.perf_counter() - eval_time))
