#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import mujoco as mj

SITE1_DEFAULT = "if_distal_site_ball"
SITE2_DEFAULT = "tf_distal_site_ball"

TH_ROOT_JOINT_DEFAULT      = "th_root_link"
TH_PROXIMAL_JOINT_DEFAULT  = "th_proximal_link"
TH_DISTAL_JOINT_DEFAULT    = "th_distal_link"

IF_PROXIMAL_JOINT_DEFAULT  = "if_proximal_link"
IF_DISTAL_JOINT_DEFAULT    = "if_distal_link"

def _jid(model, name):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"joint '{name}' not found")
    return int(jid)

def _aid_for_joint(model, jid):
    for ai in range(model.nu):
        if int(model.actuator_trnid[ai,0]) == int(jid):
            return int(ai)
    return -1

def _qadr(model, jid):
    return int(model.jnt_qposadr[jid])

def _ctrl_range_intersection(model, jid, aid):
    lo, hi = model.actuator_ctrlrange[aid]
    if int(model.jnt_limited[jid]):
        jlo, jhi = model.jnt_range[jid]
        lo = max(lo, jlo); hi = min(hi, jhi)
    if lo > hi: raise RuntimeError(f"invalid range intersection for joint id {jid}")
    return float(lo), float(hi)

def _linspace_inclusive(lo, hi, n):
    return np.linspace(float(lo), float(hi), int(n), dtype=np.float64)

def _load_csv_mapping(path, deg=False):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 2: continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            xs.append(x); ys.append(y)
    if not xs:
        raise ValueError(f"empty or invalid csv: {path}")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if deg:
        x = np.deg2rad(x); y = np.deg2rad(y)
    idx = np.argsort(x)
    return x[idx], y[idx]

def _interp_distal(x_src, y_src, x_query):
    return np.interp(x_query, x_src, y_src)

def _set_hinge_or_slide_qpos(model, data, joint_name, value):
    jid = _jid(model, joint_name)
    if model.jnt_type[jid] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
        raise RuntimeError(f"joint '{joint_name}' must be hinge/slide")
    adr = _qadr(model, jid)
    data.qpos[adr] = float(value)

def build_grid_with_csv(xml_path,
                        site1=SITE1_DEFAULT, site2=SITE2_DEFAULT,
                        th_root_name=TH_ROOT_JOINT_DEFAULT,
                        th_prox_name=TH_PROXIMAL_JOINT_DEFAULT,
                        th_dist_name=TH_DISTAL_JOINT_DEFAULT,
                        if_prox_name=IF_PROXIMAL_JOINT_DEFAULT,
                        if_dist_name=IF_DISTAL_JOINT_DEFAULT,
                        th_csv="", if_csv="",
                        csv_deg=False,
                        steps_per_axis=100,
                        shards=1, shard_id=0,
                        out="scan_from_csv.npy",
                        dtype="float32",
                        checkpoint_every=0):

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(xml_path)
    if not os.path.isfile(th_csv) or not os.path.isfile(if_csv):
        raise FileNotFoundError(f"CSV missing: th_csv={th_csv}, if_csv={if_csv}")

    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    qpos_init = data.qpos.copy()
    qvel_init = data.qvel.copy()

    th_root_jid = _jid(model, th_root_name)
    th_prox_jid = _jid(model, th_prox_name)
    if_prox_jid = _jid(model, if_prox_name)

    th_root_aid = _aid_for_joint(model, th_root_jid)
    th_prox_aid = _aid_for_joint(model, th_prox_jid)
    if_prox_aid = _aid_for_joint(model, if_prox_jid)
    if min(th_root_aid, th_prox_aid, if_prox_aid) < 0:
        raise RuntimeError("some scan joint has no actuator mapped (needed for ctrl range)")

    r_th_root = _ctrl_range_intersection(model, th_root_jid, th_root_aid)
    r_th_prox = _ctrl_range_intersection(model, th_prox_jid, th_prox_aid)
    r_if_prox = _ctrl_range_intersection(model, if_prox_jid, if_prox_aid)

    g_th_root = _linspace_inclusive(*r_th_root, steps_per_axis)
    g_th_prox = _linspace_inclusive(*r_th_prox, steps_per_axis)
    g_if_prox = _linspace_inclusive(*r_if_prox, steps_per_axis)

    th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
    if_x,  if_y  = _load_csv_mapping(if_csv,  deg=csv_deg)

    sid1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site1)
    sid2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site2)
    if sid1 < 0 or sid2 < 0:
        raise RuntimeError(f"site not found: {site1 if sid1<0 else ''} {site2 if sid2<0 else ''}")

    per_dtype = np.float32 if dtype == "float32" else np.float64
    N_total = steps_per_axis**3
    shards = max(1, int(shards)); shard_id = int(shard_id)
    if not (0 <= shard_id < shards):
        raise ValueError(f"shard_id must be in [0,{shards-1}]")
    i0 = (steps_per_axis * shard_id) // shards
    i1 = (steps_per_axis * (shard_id + 1)) // shards
    if i0 >= i1: raise ValueError(f"invalid shard slice: [{i0},{i1})")
    N_shard = (i1 - i0) * steps_per_axis * steps_per_axis

    out_arr = np.empty((N_shard, 6), dtype=per_dtype)

    print(f"[ranges] th_root={r_th_root} | th_prox={r_th_prox} | if_prox={r_if_prox}")
    print(f"[grid] steps_per_axis={steps_per_axis} → total={N_total}, shard={shard_id+1}/{shards} → {N_shard}")

    idx = 0
    for ii, i in enumerate(range(i0, i1), start=1):
        th_root_val = g_th_root[i]
        for j in range(steps_per_axis):
            th_prox_val = g_th_prox[j]
            th_dist_val = _interp_distal(th_x, th_y, th_prox_val)
            for k in range(steps_per_axis):
                if_prox_val = g_if_prox[k]
                if_dist_val = _interp_distal(if_x, if_y, if_prox_val)

                data.qpos[:] = qpos_init
                data.qvel[:] = qvel_init

                _set_hinge_or_slide_qpos(model, data, th_root_name, th_root_val)
                _set_hinge_or_slide_qpos(model, data, th_prox_name, th_prox_val)
                _set_hinge_or_slide_qpos(model, data, th_dist_name, th_dist_val)
                _set_hinge_or_slide_qpos(model, data, if_prox_name, if_prox_val)
                _set_hinge_or_slide_qpos(model, data, if_dist_name, if_dist_val)

                mj.mj_forward(model, data)

                p1 = data.site_xpos[sid1]
                p2 = data.site_xpos[sid2]
                dist = float(np.linalg.norm(p1 - p2))

                out_arr[idx, 0] = th_prox_val
                out_arr[idx, 1] = th_root_val
                out_arr[idx, 2] = th_dist_val
                out_arr[idx, 3] = if_prox_val
                out_arr[idx, 4] = if_dist_val
                out_arr[idx, 5] = dist
                idx += 1

        done = ii / float(i1 - i0)
        print(f"\rprogress: {done:.1%}  (rows={idx}/{N_shard})", end="", flush=True)
        if checkpoint_every and (ii % int(checkpoint_every) == 0):
            ck = out.replace(".npy", f"_checkpoint_i{ii}.npy")
            np.save(ck, out_arr[:idx])
            print(f"\n[checkpoint] saved {idx} rows to {ck}")

    print("\n[done] building array…")
    np.save(out, out_arr)
    print(f"saved: {out}  | shape={out_arr.shape}  dtype={out_arr.dtype}")

    meta = {
        "xml": xml_path,
        "sites": (site1, site2),
        "scan_joints": {
            "th_root": th_root_name,
            "th_prox": th_prox_name,
            "if_prox": if_prox_name,
        },
        "mapped_distal": {
            "th_dist": th_dist_name,
            "if_dist": if_dist_name,
        },
        "ranges": {
            "th_root": r_th_root,
            "th_prox": r_th_prox,
            "if_prox": r_if_prox,
        },
        "steps_per_axis": int(steps_per_axis),
        "csv_deg": bool(csv_deg),
        "csv_files": {"thumb": th_csv, "index": if_csv},
        "shards": int(shards),
        "shard_id": int(shard_id),
        "out": out,
        "columns": ["th_prox", "th_root", "th_dist", "if_prox", "if_dist", "distance"],
    }
    np.save(os.path.splitext(out)[0] + "_meta.npy", meta, allow_pickle=True)
    print("saved meta:", os.path.splitext(out)[0] + "_meta.npy")

def parse_args():
    ap = argparse.ArgumentParser("用 CSV 的 proximal→distal 映射做 100^3 网格并计算 site 距离（纯运动学）")
    ap.add_argument("--xml", default="assets/left_hand.xml")
    ap.add_argument("--site1", default=SITE1_DEFAULT)
    ap.add_argument("--site2", default=SITE2_DEFAULT)
    ap.add_argument("--th-root", default=TH_ROOT_JOINT_DEFAULT)
    ap.add_argument("--th-prox", default=TH_PROXIMAL_JOINT_DEFAULT)
    ap.add_argument("--th-dist", default=TH_DISTAL_JOINT_DEFAULT)
    ap.add_argument("--if-prox", default=IF_PROXIMAL_JOINT_DEFAULT)
    ap.add_argument("--if-dist", default=IF_DISTAL_JOINT_DEFAULT)
    ap.add_argument("--th-csv", default="th_relation.csv", help="拇指: 两列 CSV 映射 th_prox → th_dist")
    ap.add_argument("--if-csv", default="if_curve.csv", help="食指: 两列 CSV 映射 if_prox → if_dist")
    ap.add_argument("--csv-deg", action="store_true", help="CSV 两列为度，自动转弧度")
    ap.add_argument("--steps-per-axis", type=int, default=100)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=0, help="按 th_root 维度每多少步落一次检查点")
    ap.add_argument("--out", default="scan_from_csv.npy")
    ap.add_argument("--dtype", choices=["float32","float64"], default="float32")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_grid_with_csv(
        xml_path=args.xml,
        site1=args.site1, site2=args.site2,
        th_root_name=args.th_root,
        th_prox_name=args.th_prox,
        th_dist_name=args.th_dist,
        if_prox_name=args.if_prox,
        if_dist_name=args.if_dist,
        th_csv=args.th_csv,
        if_csv=args.if_csv,
        csv_deg=bool(args.csv_deg),
        steps_per_axis=int(args.steps_per_axis),
        shards=int(args.shards),
        shard_id=int(args.shard_id),
        out=args.out,
        dtype=args.dtype,
        checkpoint_every=int(args.checkpoint_every),
    )
