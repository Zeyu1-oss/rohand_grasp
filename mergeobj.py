#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse
import numpy as np
import trimesh
import mujoco
import transforms3d.quaternions as tq
from collections import defaultdict

# ===== 常量 =====
FIXED_SCALE = 0.0004
HAND_JOINTS5   = ["th_root_link","th_proximal_link","th_distal_link","if_proximal_link","if_distal_link"]
HAND_FREEJOINT = "root"
_LEGO_EXTS = (".stl", ".obj", ".ply", ".STL", ".OBJ", ".PLY")

# ===== 基础工具 =====
def qnorm(q): 
    return float(np.linalg.norm(np.asarray(q, float).ravel()[:4]))

def split_pose7(p7):
    p7 = np.asarray(p7, float).ravel()
    return p7[:3], p7[3:7]

def xyzw_to_wxyz(q):
    q = np.asarray(q, float).ravel()
    return np.array([q[3], q[0], q[1], q[2]], float)

def wxyz_to_xyzw(q):
    q = np.asarray(q, float).ravel()
    return np.array([q[1], q[2], q[3], q[0]], float)

def compose_AB_wxyz(pA, qA_wxyz, pB, qB_wxyz):
    """ T_A * T_B （四元数均为 wxyz），返回 (p, q_wxyz) """
    RA = tq.quat2mat(qA_wxyz)
    RB = tq.quat2mat(qB_wxyz)
    p = np.asarray(pA, float) + RA @ np.asarray(pB, float)
    R = RA @ RB
    q = tq.mat2quat(R)  # wxyz
    return p, q

def load_payload(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, dict): 
        return arr
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try: 
            return arr.item()
        except Exception:
            pass
    raise RuntimeError(f"{path} 不是包含 dict 的 .npy")

def pick(d: dict, keys):
    for k in keys:
        if k in d: 
            return d[k]
    raise KeyError(keys[0])

# ===== 按 pair 分组与选择 =====
_PAIR_RE = re.compile(r"pair(\d+)_ctrl(\d+)\.succ\.npy$")

def collect_succ_files(pattern_or_dir: str) -> list:
    p = os.path.abspath(pattern_or_dir)
    if os.path.isdir(p):
        paths = glob.glob(os.path.join(p, "**", "*.succ.npy"), recursive=True)
    else:
        tmp = glob.glob(pattern_or_dir, recursive=True)
        files = [t for t in tmp if os.path.isfile(t) and t.endswith(".succ.npy")]
        dirs  = [t for t in tmp if os.path.isdir(t)]
        paths = files
        for d in dirs:
            paths += glob.glob(os.path.join(d, "**", "*.succ.npy"), recursive=True)
    return sorted(set(paths))

def extract_pair_id(path) -> int | None:
    m = _PAIR_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    # 备选：从 payload 读 pair_index
    try:
        payload = load_payload(path)
        if "pair_index" in payload:
            return int(payload["pair_index"])
    except Exception:
        pass
    return None

def choose_paths_per_pair(paths: list, per_pair: int = 3, max_pairs: int = -1) -> list:
    """从 paths 中按 pair000000 开始依序挑选，每个 pair 取最多 per_pair 条。"""
    buckets = defaultdict(list)
    min_id, max_id = None, None
    for p in paths:
        pid = extract_pair_id(p)
        if pid is None:
            continue
        buckets[pid].append(p)
        min_id = pid if min_id is None else min(min_id, pid)
        max_id = pid if max_id is None else max(max_id, pid)
    if min_id is None:
        return []

    # 规范化：每个 pair 内部排序（保证可重复性）
    for pid in buckets:
        buckets[pid] = sorted(buckets[pid])

    selected = []
    pairs_picked = 0
    for pid in range(min_id, max_id + 1):
        if pid not in buckets:
            continue
        selected += buckets[pid][:per_pair]
        pairs_picked += 1
        if max_pairs > 0 and pairs_picked >= max_pairs:
            break
    return selected

# ===== LEGO 网格 =====
def resolve_lego_path(root_or_file: str, lego_name: str | None):
    if not root_or_file: 
        return None
    p = os.path.abspath(root_or_file)
    if os.path.isfile(p) and os.path.splitext(p)[1] in _LEGO_EXTS: 
        return p
    if not os.path.isdir(p): 
        return None
    if not lego_name: 
        return p
    # 1) 精确子目录
    for cand in (lego_name, lego_name.lower(), lego_name.upper()):
        d = os.path.join(p, cand)
        if os.path.isdir(d): 
            return d
    # 2) 根目录下同名文件
    for ext in _LEGO_EXTS:
        f = os.path.join(p, lego_name + ext)
        if os.path.isfile(f): 
            return f
    # 3) 递归模糊
    lego_lower = lego_name.lower()
    for d in glob.glob(os.path.join(p, "**", "*"), recursive=True):
        if os.path.isdir(d) and lego_lower in os.path.basename(d).lower(): 
            return d
    for f in glob.glob(os.path.join(p, "**", "*"), recursive=True):
        if os.path.isfile(f):
            stem, ext = os.path.splitext(os.path.basename(f))
            if ext in _LEGO_EXTS and lego_lower in stem.lower(): 
                return f
    return p

def load_lego_meshes_scaled(mesh_root_or_file, lego_name: str | None = None):
    target = resolve_lego_path(mesh_root_or_file, lego_name)
    if target is None: 
        raise FileNotFoundError(f"LEGO 根路径无效：{mesh_root_or_file}")
    if os.path.isfile(target) and os.path.splitext(target)[1].lower() in (".stl",".obj",".ply"):
        paths = [target]
    else:
        paths = []
        for ext in ("*.stl","*.obj","*.ply","*.STL","*.OBJ","*.PLY"):
            paths += glob.glob(os.path.join(target, "**", ext), recursive=True)
        paths = sorted(paths)
    if not paths: 
        raise FileNotFoundError(f"未找到网格: {target}")
    meshes = []
    for fp in paths:
        m = trimesh.load(fp, force="mesh")
        if isinstance(m, trimesh.Scene): 
            m = m.dump().sum()
        if m.is_empty: 
            continue
        m.apply_scale(FIXED_SCALE)
        meshes.append(m)
    if not meshes: 
        raise RuntimeError(f"读取 lego 网格失败或为空: {target}")
    return meshes

def find_joint_qposadr(model, jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0: 
        raise ValueError(f"joint not found: {jname}")
    return int(model.jnt_qposadr[jid])

def set_freejoint_pose_xyzw(model, data, freejoint_name, pose7_xyzw):
    adr = find_joint_qposadr(model, freejoint_name)
    p, q = split_pose7(pose7_xyzw)
    data.qpos[adr:adr+3]   = p
    data.qpos[adr+3:adr+7] = q
    mujoco.mj_forward(model, data)

def set_named_joint_scalar(model, data, jname, value):
    adr = find_joint_qposadr(model, jname)
    data.qpos[adr] = float(value)

def get_hand_geoms(model):
    out = []
    for gid in range(model.ngeom):
        mid = int(model.geom_dataid[gid])
        if mid >= 0:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
            out.append((gname, gid, mid))
    return out

def place_mesh(vertices, faces, p_world, q_wxyz):
    if abs(qnorm(q_wxyz) - 1.0) > 1e-2:
        print("[warn] quaternion not normalized, norm=%.4f" % qnorm(q_wxyz))
    R = tq.quat2mat(q_wxyz)
    v = (np.asarray(vertices, float) @ R.T) + np.asarray(p_world, float)
    return trimesh.Trimesh(vertices=v, faces=faces, process=False)

def hand_world_pose_xyzw_from_payload(payload):

    # 1) 直接 world(root)
    keys_world = ("root_pose7_in_world(xyzw)", "root_pose7_in_world")
    if any(k in payload for k in keys_world):
        r7 = np.asarray(pick(payload, keys_world), float)
        return r7  # 已是 xyzw

    # 2) 组合
    r7_L = np.asarray(pick(payload, ("root_pose7_in_lego(xyzw)", "root_pose7_in_lego")), float)
    p_Lr, q_Lr_xyzw = split_pose7(r7_L)
    q_Lr_wxyz = xyzw_to_wxyz(q_Lr_xyzw)

    L7_w = np.asarray(pick(payload, ("lego_pose7_in_world(xyzw)", "lego_pose7_in_world")), float)
    p_wL, q_wL_wxyz = split_pose7(L7_w)  # 注意：这里按 wxyz 解释（虽然名字写 xyzw）

    p_wr, q_wr_wxyz = compose_AB_wxyz(p_wL, q_wL_wxyz, p_Lr, q_Lr_wxyz)
    q_wr_xyzw = wxyz_to_xyzw(q_wr_wxyz)
    return np.hstack([p_wr, q_wr_xyzw])

def lego_world_pose_wxyz_from_payload(payload):
    L7_w = np.asarray(pick(payload, ("lego_pose7_in_world(xyzw)", "lego_pose7_in_world")), float)
    p, q_wxyz = split_pose7(L7_w)  # 后四位实际是 wxyz
    return p, q_wxyz

def build_one_pose_mesh(succ_path, hand_xml, lego_mesh_root, include_hand=True, include_lego=True):
    payload = load_payload(succ_path)

    root7_xyzw_world = hand_world_pose_xyzw_from_payload(payload)
    pL, qL_wxyz      = lego_world_pose_wxyz_from_payload(payload)

    qpos5 = np.asarray(pick(payload, ("qpos5",)), float).ravel()[:5]
    lego_name = str(payload.get("lego", "")).strip() or None

    spec = mujoco.MjSpec.from_file(hand_xml)
    model = spec.compile()
    data  = mujoco.MjData(model)
    for nm, v in zip(HAND_JOINTS5, qpos5):
        set_named_joint_scalar(model, data, nm, float(v))
    set_freejoint_pose_xyzw(model, data, HAND_FREEJOINT, root7_xyzw_world)
    mujoco.mj_forward(model, data)

    meshes = []
    if include_hand:
        parts = []
        for _, gid, mid in get_hand_geoms(model):
            mid = int(mid)
            va = int(model.mesh_vertadr[mid]); vn = int(model.mesh_vertnum[mid])
            fa = int(model.mesh_faceadr[mid]); fn = int(model.mesh_facenum[mid])
            vert = model.mesh_vert[va:va+vn].copy()
            face = model.mesh_face[fa:fa+fn].copy()
            Rg = data.geom_xmat[gid].reshape(3,3).copy()
            tg = data.geom_xpos[gid].copy()
            vw = (vert @ Rg.T) + tg
            parts.append(trimesh.Trimesh(vertices=vw, faces=face, process=False))
        if parts: 
            meshes.append(trimesh.util.concatenate(parts))

    if include_lego:
        lego_parts = load_lego_meshes_scaled(lego_mesh_root, lego_name)
        lego_world = [place_mesh(m.vertices, m.faces, pL, qL_wxyz) for m in lego_parts]
        meshes.append(trimesh.util.concatenate(lego_world))

    if not meshes: 
        raise RuntimeError("既未包含手也未包含物体")
    return trimesh.util.concatenate(meshes)

def offset_mesh(m, offset):
    off = np.asarray(offset, float).reshape(3,)
    return trimesh.Trimesh(vertices=m.vertices + off, faces=m.faces, process=False)

def layout_offset(index, layout, spacing, grid_cols):
    """
    生成偏移，仅用于可视化排版。
    - XY 平面：改变 x,y，不动 z
    - XZ 平面：改变 x,z，不动 y
    - 线性：沿 x/y/z 单轴排列
    """
    if layout in ("none", None):
        return np.zeros(3)

    if layout in ("grid_xy", "grid"):  # 兼容 "grid" -> grid_xy
        c = max(1, int(grid_cols))
        r = index // c
        k = index %  c
        return np.array([k*spacing, -r*spacing, 0.0])  # 只改 x,y；z 不变

    if layout == "grid_xz":
        c = max(1, int(grid_cols))
        r = index // c
        k = index %  c
        return np.array([k*spacing, 0.0, -r*spacing])  # x,z 排布

    if layout == "line_x":
        return np.array([index*spacing, 0.0, 0.0])
    if layout == "line_y":
        return np.array([0.0, index*spacing, 0.0])
    if layout == "line_z":
        return np.array([0.0, 0.0, index*spacing])

    return np.zeros(3)

def export_many_to_single_obj(
    succ_glob_or_dir, hand_xml, lego_mesh_root, out_obj,
    include_hand=True, include_lego=True,
    per_pair=3, max_pairs=-1,
    layout="grid_xy", spacing=0.2, grid_cols=10, merge_into_single=False
):
    all_paths = collect_succ_files(succ_glob_or_dir)
    paths = choose_paths_per_pair(all_paths, per_pair=per_pair, max_pairs=max_pairs)
    if not paths: 
        raise FileNotFoundError(f"没有匹配到 .succ.npy：{succ_glob_or_dir}")

    per_pose = []
    for i, p in enumerate(paths):
        print(f"[{i+1}/{len(paths)}] {p}")
        try:
            m = build_one_pose_mesh(p, hand_xml, lego_mesh_root, include_hand, include_lego)
        except Exception as e:
            print(f"  -> 跳过（构建失败）：{e}")
            continue
        off = layout_offset(i, layout, spacing, grid_cols)
        m2 = offset_mesh(m, off); m2.visual = None
        per_pose.append((f"pose_{i:04d}", m2))

    if not per_pose: 
        raise RuntimeError("所有条目都构建失败")

    os.makedirs(os.path.dirname(os.path.abspath(out_obj)), exist_ok=True)
    if merge_into_single:
        trimesh.util.concatenate([m for _, m in per_pose]).export(out_obj)
    else:
        scene = trimesh.Scene()
        for name, m in per_pose: 
            scene.add_geometry(m, node_name=name, geom_name=name)
        scene.export(out_obj)
    print(f"[OK] 导出 {len(per_pose)} 个姿态 → {out_obj}")

def main():
    ap = argparse.ArgumentParser("按 pair 顺序、每个 pair 取 3 个抓取导出 OBJ")
    ap.add_argument("--succ_glob", default="results/eval_rohand/4668_reddish_brown/pair*_ctrl*.succ.npy",
                    help="匹配 .succ.npy 的通配符 或 目录")
    ap.add_argument("--hand_xml", default="assets/left_hand1.xml", help="手的 XML")
    ap.add_argument("--lego_mesh", default="assets/object", help="LEGO 网格根路径或文件")
    ap.add_argument("--out", default="outputs/grasps_all.obj", help="输出 OBJ 路径")
    ap.add_argument("--no_hand", action="store_true", help="不包含手")
    ap.add_argument("--no_lego", action="store_true", help="不包含物体")
    ap.add_argument("--per_pair", type=int, default=3, help="每个 pair 选取的抓取数")
    ap.add_argument("--max_pairs", type=int, default=30, help="最多处理多少个 pai")
    ap.add_argument("--layout",
                    choices=["none","grid","grid_xy","grid_xz","line_x","line_y","line_z"],
                    default="grid_xy",
                    help="姿态布局")
    ap.add_argument("--spacing", type=float, default=0.2, help="布局间距（米）")
    ap.add_argument("--grid_cols", type=int, default=10, help="网格布局列数（每行个数）")
    ap.add_argument("--merge", action="store_true", help="将所有姿态焊接为单一网格")
    args = ap.parse_args()

    export_many_to_single_obj(
        succ_glob_or_dir=args.succ_glob,
        hand_xml=args.hand_xml,
        lego_mesh_root=args.lego_mesh,
        out_obj=args.out,
        include_hand=(not args.no_hand),
        include_lego=(not args.no_lego),
        per_pair=args.per_pair,
        max_pairs=args.max_pairs,
        layout=args.layout,
        spacing=args.spacing,
        grid_cols=args.grid_cols,
        merge_into_single=args.merge,
    )

if __name__ == "__main__":
    main()
