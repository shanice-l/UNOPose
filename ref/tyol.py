# encoding: utf-8
"""This file includes necessary params, info."""
import mmengine
import os.path as osp

import numpy as np
from dataclasses import dataclass
from core.unopose.utils.data_utils import load_json
from lib.utils.utils import lazy_property

from plyfile import PlyData
import os
from bop_toolkit_lib import transform

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
# directory storing experiment data (result, model checkpoints, etc).
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")


def get_obj_rendering(model_path) -> dict:
    """
    returns object usable for vispy rendering
    argument obj_model is expected to have the following fields:
      - pts : (N,3) xyz points in mm
      - normals: (N,3) normals
      - faces: (M,3) polygon faces needed for rendering
    """

    pcd = PlyData.read(model_path)
    # these are already in mm
    xs = pcd["vertex"]["x"]
    ys = pcd["vertex"]["y"]
    zs = pcd["vertex"]["z"]
    nxs = pcd["vertex"]["nx"]
    nys = pcd["vertex"]["ny"]
    nzs = pcd["vertex"]["nz"]

    raw_vertexs = np.asarray(pcd["face"]["vertex_indices"])
    faces = np.stack([vert for vert in raw_vertexs], axis=0)

    xyz = np.stack((xs, ys, zs), axis=1)
    normals = np.stack((nxs, nys, nzs), axis=1)

    return {"pts": xyz, "normals": normals, "faces": faces}


def get_symmetry_transformations(model_info, max_sym_disc_step=0.01):
    """Returns a set of symmetry transformations for an object model.

    :param model_info: See files models_info.json provided with the datasets.
    :param max_sym_disc_step: The maximum fraction of the object diameter which
      the vertex that is the furthest from the axis of continuous rotational
      symmetry travels between consecutive discretized rotations.
    :return: The set of symmetry transformations.
    """
    # Discrete symmetries.
    trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
    if "symmetries_discrete" in model_info:
        for sym in model_info["symmetries_discrete"]:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({"R": R, "t": t})

    # Discretized continuous symmetries.
    trans_cont = []
    if "symmetries_continuous" in model_info:
        for sym in model_info["symmetries_continuous"]:
            axis = np.array(sym["axis"])
            offset = np.array(sym["offset"]).reshape((3, 1))

            # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
            discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

            # Discrete step in radians.
            discrete_step = 2.0 * np.pi / discrete_steps_count

            for i in range(0, discrete_steps_count):
                R = transform.rotation_matrix(i * discrete_step, axis)[:3, :3]
                t = -R.dot(offset) + offset
                trans_cont.append({"R": R, "t": t})

    # Combine the discrete and the discretized continuous symmetries.
    trans = []
    for tran_disc in trans_disc:
        if len(trans_cont):
            for tran_cont in trans_cont:
                R = tran_cont["R"].dot(tran_disc["R"])
                t = tran_cont["R"].dot(tran_disc["t"]) + tran_cont["t"]
                trans.append({"R": R, "t": t})
        else:
            trans.append(tran_disc)

    return trans


def format_sym_set(syms: dict) -> np.ndarray:
    """
    Format a symmetry set provided by BOP into a nd array od shape [N,3,4]
    """

    syms_r = np.stack([np.asarray(sym["R"]) for sym in syms], axis=0)
    syms_t = np.stack([np.asarray(sym["t"]) for sym in syms], axis=0)
    sym_poses = np.concatenate([syms_r, syms_t], axis=2)

    return sym_poses


# ---------------------------------------------------------------- #
# HB DATASET
# ---------------------------------------------------------------- #
@dataclass
class tyol:
    dataset_root = osp.join(bop_root, "tyol")
    train_dir = osp.join(dataset_root, "train")
    test_dir = osp.join(dataset_root, "test")
    model_dir = osp.join(dataset_root, "models")
    vertex_scale = 0.001

    # object info
    id2obj = {i: str(i) for i in range(1, 22)}
    objects = [str(obj) for obj in id2obj.values()]
    obj_num = len(id2obj)
    obj2id = {_name: _id for _id, _name in id2obj.items()}

    @lazy_property
    def model_paths(self):
        return [osp.join(self.model_dir, "obj_{:06d}.ply").format(_id) for _id in self.id2obj]

    def get_obj_full_data(self):
        model_infos = load_json(osp.join(self.model_dir, "models_info.json"))
        obj_models, obj_diams, obj_symms = dict(), dict(), dict()
        for obj_id in self.id2obj:
            model_path = osp.join(self.model_dir, "obj_{:06d}.ply").format(obj_id)
            obj_models[obj_id] = get_obj_rendering(model_path)
            obj_diams[obj_id] = model_infos[str(obj_id)]["diameter"]
            obj_symm = get_symmetry_transformations(model_infos[str(obj_id)], max_sym_disc_step=0.05)
            obj_symms[obj_id] = format_sym_set(obj_symm)
        return obj_models, obj_diams, obj_symms

    def get_obj_data(self, obj_id):
        model_info = load_json(osp.join(self.model_dir, "models_info.json"))[str(obj_id)]
        model_path = osp.join(self.model_dir, "obj_{:06d}.ply").format(obj_id)
        obj_model = get_obj_rendering(model_path)
        obj_diam = model_info["diameter"]
        obj_symm = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
        obj_symm = format_sym_set(obj_symm)
        return obj_model, obj_diam, obj_symm


if __name__ == "__main__":
    data_ref = tyol()
    import ipdb

    ipdb.set_trace()
