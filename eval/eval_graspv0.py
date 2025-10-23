import os, sys, glob, json, time, argparse
from typing import Any, Dict, List, Tuple
import numpy as np
import mujoco as mj

FIXED_SCALE = 0.0004
CONTACT_DUMP_MAX = 32


def _act_id_of_joint(model, jname):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
    for aid in range(model.nu):
        if int(model.actuator_trnid[aid,0]) == jid:
            return aid
    raise RuntimeError(f"no actuator for joint {jname}")

def _ctrl_clamp(model, aid, val):
    lo = float(model.actuator_ctrlrange[aid,0])
    hi = float(model.actuator_ctrlrange[aid,1])
    return float(np.clip(val, lo, hi))

def _simple_yaml_load(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"): continue
        if "#" in s: s = s.split("#",1)[0].rstrip()
        if ":" not in s: continue
        k, v = s.split(":", 1)
        key, val = k.strip(), v.strip()
        if val.startswith(("'", '"')) and val.endswith(("'", '"')) and len(val) >= 2:
            out[key] = val[1:-1]; continue
        low = val.lower()
        if low in ("true","false"): out[key] = (low=="true"); continue
        try:
            if "." in val or "e" in low: out[key] = float(val)
            else: out[key] = int(val)
            continue
        except Exception:
            pass
        out[key] = val
    return out

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if path.endswith((".yaml",".yml")):
        try:
            import yaml
            return yaml.safe_load(txt)
        except Exception:
            return _simple_yaml_load(txt)
    return json.loads(txt)

def _abs_join(cfg_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(cfg_dir, p))

def load_lego_list(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        for key in ("lego_names","names"):
            if key in obj and isinstance(obj[key], list):
                return [str(x).strip() for x in obj[key] if str(x).strip()]
    raise ValueError(f"JSON 格式不识别：{json_path}")


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    mj.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(9))
    return q

def rodrigues(axis, angle):
    a = np.asarray(axis, np.float64); n = np.linalg.norm(a)
    if n < 1e-12: return np.eye(3)
    a = a/n
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]], np.float64)
    c, s = np.cos(angle), np.sin(angle)
    return np.eye(3)+s*K+(1-c)*(K@K)

def resolve_site_name(model, name: str) -> str:
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
    if sid >= 0: return name
    if "_ball" in name:
        alt = name.replace("_ball","_a")
        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, alt) >= 0:
            return alt
    raise RuntimeError(f"site '{name}' not found")

def get_site_pos(model, data, site_name: str):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0: raise RuntimeError(f"site '{site_name}' not found")
    return data.site_xpos[sid].copy()

def _jid(model, name):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0: raise RuntimeError(f"joint '{name}' not found")
    return int(jid)

def lego_pose_from_body(model, data, body_name: str):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0: raise RuntimeError(f"lego body '{body_name}' not found")
    p = data.xpos[bid].copy()
    R = data.xmat[bid].reshape(3,3).copy()
    return p, R, bid

def world_from_local(p_local, p_w, R_w):
    return p_w + R_w @ p_local

def _is_mesh_file(p: str) -> bool:
    return os.path.isfile(p) and os.path.splitext(p)[1].lower() in {".obj",".stl",".ply"}

def _find_mesh_files(root_or_file: str) -> List[str]:
    p = os.path.abspath(root_or_file)
    if _is_mesh_file(p):
        return [p]
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    ret = []
    for ext in ("*.obj","*.stl",".ply","*.PLY","*.OBJ","*.STL"):
        if ext.startswith("."):
            ext = "*" + ext
        ret += glob.glob(os.path.join(p, "**", ext), recursive=True)
    ret = [os.path.abspath(x) for x in ret if _is_mesh_file(x)]
    if not ret:
        raise FileNotFoundError(f"未在目录中找到网格: {p}")
    return sorted(ret)

def _safe_mesh_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
    return base or "mesh"

def _body_name_of_joint(model, jname: str) -> str:
    jid = _jid(model, jname)
    bid = int(model.jnt_bodyid[jid])
    nm  = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
    if not nm:
        raise RuntimeError(f"cannot find body of joint '{jname}'")
    return nm

def rel_pose7_bodyA_wrt_bodyB(model, data, bodyA: str, bodyB: str) -> np.ndarray:
    bidA = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bodyA)
    bidB = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bodyB)
    if bidA < 0 or bidB < 0:
        raise RuntimeError(f"bad body names: A='{bodyA}', B='{bodyB}'")
    pA = data.xpos[bidA].copy()
    RA = data.xmat[bidA].reshape(3,3).copy()
    pB = data.xpos[bidB].copy()
    RB = data.xmat[bidB].reshape(3,3).copy()
    Rrel = RB.T @ RA
    prel = RB.T @ (pA - pB)
    qrel = mat_to_quat_xyzw(Rrel)
    return np.hstack([prel, qrel])

def _ensure_bin_stl(fp: str) -> str:
    if not fp.lower().endswith(".stl"):
        return fp
    try:
        import trimesh
        m = trimesh.load(fp, force="mesh")
        if isinstance(m, trimesh.Scene): m = m.dump().sum()
        outp = fp[:-4] + "_bin.stl"
        m.export(outp, file_type="stl")
        return outp
    except Exception:
        return fp

def _safe_sync(viewer, every: int, i: int, sleep_s: float):
    if viewer is not None and (i % max(1, every) == 0):
        viewer.sync()
        if sleep_s > 0:
            time.sleep(sleep_s)

def _get_joint_range(model, jname: str) -> Tuple[float, float, bool]:
    jid = _jid(model, jname)
    limited = bool(int(model.jnt_limited[jid])) if hasattr(model, "jnt_limited") else True
    if getattr(model.jnt_range, "ndim", 1) == 2:
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
    else:
        lo = float(model.jnt_range[2*jid + 0])
        hi = float(model.jnt_range[2*jid + 1])
    return lo, hi, limited

def _set_joint_qpos_clamped(model, data, jname: str, value: float) -> float:
    jid = _jid(model, jname)
    adr = int(model.jnt_qposadr[jid])
    lo, hi, limited = _get_joint_range(model, jname)
    val = float(value)
    if limited and (lo < hi):
        val = min(max(val, lo), hi)
    data.qpos[adr] = val
    return val

def min_hand_lego_contact_dist(model, data, hand_substrs=("if_","th_","tf_"), lego_body_prefix="lego") -> float:
    hand_bids = set(); lego_bids = set()
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if any(s in name for s in hand_substrs): hand_bids.add(bid)
        if name.startswith(lego_body_prefix): lego_bids.add(bid)
    if not hand_bids or not lego_bids:
        return float("+inf")
    mn = float("+inf")
    for i in range(int(data.ncon)):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        if (b1 in hand_bids and b2 in lego_bids) or (b2 in hand_bids and b1 in lego_bids):
            mn = min(mn, float(c.dist))
    return mn

def _collect_bodyids_by_substrs(model, substrs: Tuple[str, ...]) -> set:
    out = set()
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if any(s in name for s in substrs):
            out.add(bid)
    return out

def sum_two_finger_normal_forces(model, data,
                                 if_substrs=("if_",),
                                 th_substrs=("th_", "tf_"),
                                 lego_body_prefix="lego"):
    if_bids = _collect_bodyids_by_substrs(model, if_substrs)
    th_bids = _collect_bodyids_by_substrs(model, th_substrs)
    lego_bids = set()
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if name.startswith(lego_body_prefix):
            lego_bids.add(bid)
    fn_if = 0.0
    fn_th = 0.0
    result = np.zeros(6, dtype=np.float64)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1 = int(c.geom1); g2 = int(c.geom2)
        b1 = int(model.geom_bodyid[g1]); b2 = int(model.geom_bodyid[g2])
        pair_if = ((b1 in if_bids and b2 in lego_bids) or (b2 in if_bids and b1 in lego_bids))
        pair_th = ((b1 in th_bids and b2 in lego_bids) or (b2 in th_bids and b1 in lego_bids))
        if not (pair_if or pair_th):
            continue
        mj.mj_contactForce(model, data, i, result)
        f_n = float(result[0])
        if f_n < 0.0:
            f_n = -f_n
        if f_n == 0.0:
            addr = int(c.efc_address)
            if addr >= 0:
                f_n = abs(float(data.efc_force[addr]))
        if pair_if:
            fn_if += f_n
        if pair_th:
            fn_th += f_n
    return float(fn_if), float(fn_th)

def min_contact_dist_by_finger(model, data,
                               if_substrs=("if_",),
                               th_substrs=("th_","tf_"),
                               lego_body_prefix="lego"):
    if_bids = _collect_bodyids_by_substrs(model, tuple(if_substrs))
    th_bids = _collect_bodyids_by_substrs(model, tuple(th_substrs))
    lego_bids = set()
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if name.startswith(lego_body_prefix):
            lego_bids.add(bid)
    d_if = float("+inf"); d_th = float("+inf")
    if not lego_bids or (not if_bids and not th_bids):
        return d_if, d_th
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1 = int(c.geom1); g2 = int(c.geom2)
        b1 = int(model.geom_bodyid[g1]); b2 = int(model.geom_bodyid[g2])
        if (b1 in if_bids and b2 in lego_bids) or (b2 in if_bids and b1 in lego_bids):
            d_if = min(d_if, float(c.dist))
        if (b1 in th_bids and b2 in lego_bids) or (b2 in th_bids and b1 in lego_bids):
            d_th = min(d_th, float(c.dist))
    return d_if, d_th

def any_contact_between_bodies(model, data, bodyA_names: List[str], bodyB_names: List[str]) -> bool:
    A = set(); B = set()
    for nm in bodyA_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0: A.add(int(bid))
    for nm in bodyB_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0: B.add(int(bid))
    if not A or not B:
        return False
    for i in range(int(data.ncon)):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        if ((b1 in A and b2 in B) or (b1 in B and b2 in A)) and float(c.dist) <= 0.0:
            return True
    return False

def pose7_of_body(model, data, body_name: str) -> np.ndarray:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    p = data.xpos[bid].copy(); R = data.xmat[bid].reshape(3,3).copy()
    return np.hstack([p, mat_to_quat_xyzw(R)])

def delta_pose(a7, b7):
    pa = np.asarray(a7[:3]); qa = np.asarray(a7[3:]); qa/= (np.linalg.norm(qa)+1e-12)
    pb = np.asarray(b7[:3]); qb = np.asarray(b7[3:]); qb/= (np.linalg.norm(qb)+1e-12)
    dp = float(np.linalg.norm(pb-pa))
    qc = np.empty(4); mj.mju_mulQuat(qc, qb, np.array([qa[0],-qa[1],-qa[2],-qa[3]]))
    da = 2.0*float(np.arccos(np.clip(qc[0], -1.0, 1.0)))
    if da > np.pi: da = 2*np.pi - da
    return dp, da

def _free_qadr(model, name="root"):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0 or model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"freejoint '{name}' 不存在或不是 freejoint")
    return int(model.jnt_qposadr[jid])

def _free_dadr(model, name="root"):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0 or model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"freejoint '{name}' 不存在或不是 freejoint")
    return int(model.jnt_dofadr[jid])

def _set_free_about_point(model, data, qadr, base_pos, base_quat, R_world, pivot_world):
    q_rot = mat_to_quat_xyzw(R_world)
    q_new = np.empty(4); mj.mju_mulQuat(q_new, q_rot, base_quat)
    p0 = np.asarray(pivot_world, float)
    p_new = p0 + R_world @ (base_pos - p0)
    data.qpos[qadr:qadr+3]   = p_new
    data.qpos[qadr+3:qadr+7] = q_new
    mj.mj_forward(model, data)
    return p_new, q_new

def _set_body_pose7(model, data, freejoint_name: str, pose7: np.ndarray):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, freejoint_name)
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr:adr+7] = np.asarray(pose7, float).reshape(7,)
    mj.mj_forward(model, data)

def _print_state_debug(mjmod, model, data, joint_names5, site1, site2, qadr_free, extra=None):
    try:
        print("  [debug] qpos (first 10):", np.array2string(data.qpos[:10], precision=6))
        for jn in joint_names5:
            jid = mjmod.mj_name2id(model, mjmod.mjtObj.mjOBJ_JOINT, jn)
            adr = int(model.jnt_qposadr[jid])
            print(f"  [debug] joint {jn:>18s}: qpos={float(data.qpos[adr]):+.6f}")
        root_p = data.qpos[qadr_free:qadr_free+3]
        root_q = data.qpos[qadr_free+3:qadr_free+7]
        print(f"  [debug] freejoint root pos={root_p}, quat={root_q}")
        sid1 = mjmod.mj_name2id(model, mjmod.mjtObj.mjOBJ_SITE, site1)
        sid2 = mjmod.mj_name2id(model, mjmod.mjtObj.mjOBJ_SITE, site2)
        s1w = data.site_xpos[sid1]; s2w = data.site_xpos[sid2]
        print(f"  [debug] site1={site1} world={s1w}")
        print(f"  [debug] site2={site2} world={s2w}")
        n = int(data.ncon)
        print(f"  [debug] ncon={n} (dump first {min(n, CONTACT_DUMP_MAX)})")
        for i in range(min(n, CONTACT_DUMP_MAX)):
            c = data.contact[i]
            g1 = int(c.geom1); g2 = int(c.geom2)
            b1 = int(model.geom_bodyid[g1]); b2 = int(model.geom_bodyid[g2])
            nm1 = mjmod.mj_id2name(model, mjmod.mjtObj.mjOBJ_BODY, b1) or ""
            nm2 = mjmod.mj_id2name(model, mjmod.mjtObj.mjOBJ_BODY, b2) or ""
            print(f"    - con#{i:02d}: bodies=({nm1},{nm2}) dist={float(c.dist):+.6e} "
                  f"frame={np.array2string(c.frame, precision=3)}")
        if extra:
            print("  [debug] extra:", extra)
    except Exception as e:
        print("[debug] state dump error:", e)


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
    y = np.asarray(ys, np.float64)
    if deg:
        x = np.deg2rad(x); y = np.deg2rad(y)
    idx = np.argsort(x)
    return x[idx], y[idx]

def _interp_csv_saturated(x_src: np.ndarray, y_src: np.ndarray, xq: float) -> float:
    lo, hi = float(np.min(x_src)), float(np.max(x_src))
    xq = float(np.clip(xq, lo, hi))
    return float(np.interp(xq, x_src, y_src))

def _interp_distal(x_src, y_src, x_query):
    return np.interp(x_query, x_src, y_src)

def load_scan_db(path: str,
                 th_csv: str = "", if_csv: str = "", csv_deg: bool = False):
    arr = np.load(path, allow_pickle=True)
    # 新格式：(N, >=6)
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 6:
        cols = arr.astype(np.float32)
        qpos5 = cols[:, [1, 0, 2, 3, 4]]
        dists = cols[:, 5].astype(np.float32)
        return qpos5, dists
    # 旧格式：(N,4) + CSV 反推 distal
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 4:
        th_prox = arr[:, 0].astype(np.float64)
        th_root = arr[:, 1].astype(np.float64)
        if_prox = arr[:, 2].astype(np.float64)
        dists   = arr[:, 3].astype(np.float32)
        if not (os.path.isfile(th_csv) and os.path.isfile(if_csv)):
            raise RuntimeError("旧格式 scan_db 需要提供 th_csv/if_csv 以补 distal。")
        th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
        if_x,  if_y = _load_csv_mapping(if_csv,  deg=csv_deg)
        th_dist = _interp_distal(th_x, th_y, th_prox)
        if_dist = _interp_distal(if_x,  if_y,  if_prox)
        qpos5 = np.stack([th_root, th_prox, th_dist, if_prox, if_dist], axis=1).astype(np.float32)
        return qpos5, dists
    # 字典数组
    try:
        tmp_ctrl, tmp_dist = [], []
        for row in np.atleast_1d(arr):
            if isinstance(row, dict) and "ctrl" in row and "dist" in row:
                c = np.array(row["ctrl"], float).ravel()
                d = float(row["dist"])
                tmp_ctrl.append(c); tmp_dist.append(d)
        if tmp_ctrl:
            C = np.vstack(tmp_ctrl)
            D = np.asarray(tmp_dist, np.float32)
            if C.shape[1] >= 5:
                return C[:, :5].astype(np.float32), D
            elif C.shape[1] >= 3:
                if not (os.path.isfile(th_csv) and os.path.isfile(if_csv)):
                    raise RuntimeError("旧格式 scan_db(dict) 需要 th_csv/if_csv 以补 distal。")
                th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
                if_x,  if_y = _load_csv_mapping(if_csv,  deg=csv_deg)
                th_prox = C[:, 0]
                th_root = C[:, 1]
                if_prox = C[:, 2]
                th_dist = _interp_distal(th_x, th_y, th_prox)
                if_dist = _interp_distal(if_x,  if_y,  if_prox)
                qpos5 = np.stack([th_root, th_prox, th_dist, if_prox, if_dist], axis=1).astype(np.float32)
                return qpos5, D
    except Exception:
        pass
    raise RuntimeError(f"无法解析 scan_db: {path}（期待 (N,6) 或 (N,4) 或 dict数组带 ctrl/dist）")


def load_antipodal_pairs(path: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    obj = raw
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        try:
            obj = raw.item()
        except Exception:
            obj = raw
    if isinstance(obj, dict):
        for key in ("pairs", "data", "payload"):
            if key in obj:
                v = obj[key]
                if isinstance(v, dict) and "pairs" in v:
                    v = v["pairs"]
                arr = np.asarray(v, dtype=object)
                try:
                    arr = np.asarray(arr, np.float32)
                except Exception:
                    pass
                return _coerce_pairs_array(arr, path)
        vals = list(obj.values())
        if vals and isinstance(vals[0], (list, tuple, np.ndarray)):
            return _coerce_pairs_array(np.asarray(vals, dtype=object), path)
    if isinstance(obj, np.ndarray):
        return _coerce_pairs_array(obj, path)
    raise RuntimeError(f"无法解析 antipodal 文件: {path}")

def _coerce_pairs_array(arr: np.ndarray, path: str) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[1:] == (2,3) and arr.dtype != object:
        return arr.astype(np.float32)
    if arr.dtype == object:
        items = np.atleast_1d(arr).tolist()
        P = []
        for it in items:
            if isinstance(it, dict):
                if "p1" in it and "p2" in it:
                    p = np.stack([np.array(it["p1"], float), np.array(it["p2"], float)], axis=0)
                    P.append(p)
                elif "pair" in it:
                    p = np.array(it["pair"], float)
                    P.append(p)
                elif "pairs" in it:
                    p = np.array(it["pairs"], float)
                    P.append(p)
            elif isinstance(it, (list, tuple, np.ndarray)):
                a = np.array(it, float)
                P.append(a)
        if P:
            P = np.asarray(P, np.float32)
            if P.ndim == 3 and P.shape[1:] == (2,3):
                return P
    try:
        A = np.asarray(arr, np.float32)
    except Exception as e:
        raise TypeError(f"{path} 解析失败：{e}")
    return A


def find_all_matching_ctrls(qpos5: np.ndarray, dists: np.ndarray, L: float,
                            tol: float, tol_fb: float, max_ctrls: int = -1):
    idx = np.where(np.abs(dists - L) < tol)[0]
    used_tol = tol; fallback_used = False
    if len(idx) == 0 and tol_fb > tol:
        idx = np.where(np.abs(dists - L) < tol_fb)[0]
        used_tol = tol_fb; fallback_used = True
    if len(idx) == 0:
        j = int(np.argmin(np.abs(dists - L)))
        idx = np.array([j])
        used_tol = float(abs(dists[j]-L))
        print(f"[warn] 未找到满足容差的ctrl，使用最接近项 idx={j}, |dist-L|={used_tol:.3e}")
    if max_ctrls > 0 and len(idx) > max_ctrls:
        err = np.abs(dists[idx] - L)
        idx = idx[np.argsort(err)][:max_ctrls]
        print(f"[info] 限制每个 pair 只测试 {max_ctrls} 条（从 {len(err)} 候选中挑最接近的）")
    return idx, used_tol, fallback_used


class CombinedScene:
    def __init__(self, hand_xml: str, zero_gravity: bool):
        os.environ.setdefault("MUJOCO_GL", "glfw")
        self.mj = mj
        if not hasattr(self.mj, "MjSpec"):
            raise RuntimeError("需要 mujoco>=3.2")
        self.spec = self.mj.MjSpec.from_file(hand_xml)
        if zero_gravity:
            self.spec.option.disableflags = self.mj.mjtDisableBit.mjDSBL_GRAVITY
        try:
            self.spec.add_key()
        except Exception:
            pass
        self.model = None
        self.data  = None

    def compile(self):
        try:
            self.model = self.mj.MjModel.from_spec(self.spec)
        except Exception:
            self.model = self.spec.compile()
        self.data  = self.mj.MjData(self.model)
        try:
            self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
        except Exception:
            self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)

    def add_lego_object(self, mesh_root_or_file: str,
                        friction=(0.6,0.02,0.001), density: float=1000.0,
                        body_name="lego", freejoint_name="lego_freejoint"):
        files = _find_mesh_files(mesh_root_or_file)
        files = [ _ensure_bin_stl(fp) for fp in files ]
        body = self.spec.worldbody.add_body(name=body_name)
        body.add_freejoint(name=freejoint_name)
        for fp in files:
            fp_abs = os.path.abspath(fp)
            mname  = _safe_mesh_name(fp_abs)
            self.spec.add_mesh(name=mname, file=fp_abs,
                               scale=[FIXED_SCALE, FIXED_SCALE, FIXED_SCALE])
            body.add_geom(name=f"{body_name}_visual_{mname}",
                          type=self.mj.mjtGeom.mjGEOM_MESH, meshname=mname,
                          density=0.0, contype=0, conaffinity=0, friction=friction)
            # collision
            body.add_geom(name=f"{body_name}_collision_{mname}",
                          type=self.mj.mjtGeom.mjGEOM_MESH, meshname=mname,
                          density=density, friction=friction, contype=1, conaffinity=1)
        self.compile()

    def set_lego_collision_enabled(self, enabled: bool, body_prefix="lego"):
        gids = []
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            bname = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(body_prefix):
                gids.append(gid)
        if not gids: return
        gids = np.asarray(gids, dtype=np.int32)
        if enabled:
            self.model.geom_contype[gids]     = 1
            self.model.geom_conaffinity[gids] = 1
        else:
            self.model.geom_contype[gids]     = 0
            self.model.geom_conaffinity[gids] = 0
        self.mj.mj_forward(self.model, self.data)


def eval_one_lego(cfg: Dict[str,Any], lego_name_raw: str, show=False, debug=False):
    cfg_dir = os.path.dirname(os.path.abspath(cfg["_config_path"]))
    xml_path       = _abs_join(cfg_dir, str(cfg["xml"]))
    obj_mesh_root  = _abs_join(cfg_dir, str(cfg["obj_mesh_root"]))
    antipodal_dir  = _abs_join(cfg_dir, str(cfg["antipodal_dir"]))
    scan_db_path   = _abs_join(cfg_dir, str(cfg["scan_db"]))
    out_root       = _abs_join(cfg_dir, str(cfg["out_dir"]))
    antipodal_prefix = str(cfg.get("antipodal_prefix","")).strip()

    # joints (config 可覆盖)
    th_root_name = str(cfg.get("th_root_joint", "th_root_link"))
    th_prox_name = str(cfg.get("th_prox_joint", "th_proximal_link"))
    th_dist_name = str(cfg.get("th_dist_joint", "th_distal_link"))
    if_prox_name = str(cfg.get("if_prox_joint", "if_proximal_link"))
    if_dist_name = str(cfg.get("if_dist_joint", "if_distal_link"))
    joint_names5 = [th_root_name, th_prox_name, th_dist_name, if_prox_name, if_dist_name]

    # CSV 关系（若给出则可用于 prox→dist 联动）
    th_csv = _abs_join(cfg_dir, str(cfg.get("th_csv",""))) if "th_csv" in cfg else ""
    if_csv = _abs_join(cfg_dir, str(cfg.get("if_csv",""))) if "if_csv" in cfg else ""
    csv_deg = bool(cfg.get("csv_deg", False))
    th_x = th_y = if_x = if_y = None
    if th_csv and os.path.isfile(th_csv): th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
    if if_csv and os.path.isfile(if_csv): if_x, if_y = _load_csv_mapping(if_csv,  deg=csv_deg)

    # —— 解析 tighten_joints（可任意关节），若未给出则按开关退化
    tighten_joints = cfg.get("tighten_joints", [])
    if isinstance(tighten_joints, str):
        tighten_joints = [s.strip() for s in tighten_joints.replace(";",",").split(",") if s.strip()]
    if not tighten_joints:
        use_if = bool(cfg.get("tighten_use_if_prox", True))
        use_th = bool(cfg.get("tighten_use_th_prox", True))
        tj = []
        if use_if: tj.append(if_prox_name)
        if use_th: tj.append(th_prox_name)
        if not tj:
            tj = [if_prox_name, th_prox_name]
        tighten_joints = tj

    # —— 每个 tighten 关节的符号列表
    tighten_signs = cfg.get("tighten_signs", [])
    if isinstance(tighten_signs, (int, float, str)):
        tighten_signs = [float(x) for x in (str(tighten_signs).replace(";",",").split(","))]
    elif isinstance(tighten_signs, (list, tuple)):
        tighten_signs = [float(x) for x in tighten_signs]
    else:
        tighten_signs = [-1.0]
    while len(tighten_signs) < len(tighten_joints):
        tighten_signs.append(tighten_signs[-1])

    tighten_total_cmd = float(cfg.get("tighten_cmd", 0.05))
    tighten_steps     = int(cfg.get("tighten_steps", 120))
    contact_force_eps = float(cfg.get("contact_force_eps", 1e-4))
    tighten_consecutive = int(cfg.get("tighten_consecutive_steps", 3))
    detect_force_eps = float(cfg.get("contact_detect_force_eps", max(1e-5, 0.05*contact_force_eps)))
    detect_dist_eps  = float(cfg.get("contact_detect_dist_eps", 0.0))
    sim_per_step     = int(cfg.get("tighten_sim_per_step", 3))

    # 探碰阶段参数（可选）
    precontact_force_eps = float(cfg.get("precontact_force_eps", max(1e-5, 0.25*contact_force_eps)))
    precontact_dist_eps  = float(cfg.get("precontact_dist_eps", 2.0e-4))
    precontact_max_steps = int(cfg.get("precontact_max_steps", 400))
    probe_total_cmd      = float(cfg.get("probe_cmd", 0.3 * tighten_total_cmd))
    probe_steps          = int(cfg.get("probe_steps", min(400, tighten_steps if tighten_steps>0 else 400)))

    show_interval = int(cfg.get("show_interval", 2))
    viewer_sleep  = float(cfg.get("viewer_sleep", 0.0))
    visualize_force_phase = bool(cfg.get("visualize_force_phase", True))

    if_substrs = tuple(cfg.get("if_substrs", ["if_distal_link"]))
    th_substrs = tuple(cfg.get("th_substrs", ["th_distal_link"]))

    lego_name = lego_name_raw.strip()
    name_stem = os.path.splitext(lego_name)[0]

    scene = CombinedScene(xml_path, zero_gravity=bool(cfg.get("zero_gravity", True)))

    mesh_target = obj_mesh_root if _is_mesh_file(obj_mesh_root) else os.path.join(obj_mesh_root, lego_name)
    density   = float(cfg.get("default_density", 1000.0))
    fric_cfg  = cfg.get("friction", [0.6,0.02,0.001])
    friction  = (float(fric_cfg[0]), float(fric_cfg[1]), float(fric_cfg[2] if len(fric_cfg)>=3 else 0.001))
    scene.add_lego_object(mesh_target, friction=friction, density=density,
                          body_name="lego", freejoint_name="lego_freejoint")
    model, data = scene.model, scene.data
    lego_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "lego")
    if lego_bid < 0:
        raise RuntimeError("body 'lego' not found")

    mass = float(getattr(model, "body_subtreemass", model.body_mass)[lego_bid])

    g_cfg = float(cfg.get("g_override", 0.0))
    g_val = float(np.linalg.norm(model.opt.gravity))
    if g_cfg > 0:
        g_val = g_cfg
    elif g_val == 0.0:
        g_val = 9.81

    volume_est = mass / density if density > 0 else float("nan")

    use_auto_force = bool(cfg.get("force_use_mass_g", True))
    force_scale   = float(cfg.get("force_scale", 1.0))
    auto_force_N  = mass * g_val * force_scale
    force_cfg_N   = float(cfg.get("force_N", 0.05))
    force_N = auto_force_N if use_auto_force else force_cfg_N

    print(f"[force] mass={mass:.6f} kg | g={g_val:.6f} m/s^2 | volume_est={volume_est:.6e} m^3 | "
          f"scale={force_scale:.3f} | use_auto={use_auto_force} | force_N={force_N:.6f} (cfg={force_cfg_N:.6f})")

    try:
        act_ids5 = [_act_id_of_joint(model, nm) for nm in joint_names5]
    except Exception:
        raise RuntimeError(f"解析执行器失败")

    qpos_init = data.qpos.copy()
    qvel_init = data.qvel.copy()
    ctrl_init = data.ctrl.copy()

    def reset_to_initial():
        data.qpos[:] = qpos_init
        data.qvel[:] = qvel_init
        data.ctrl[:] = ctrl_init
        mj.mj_forward(model, data)

    # viewer
    viewer = None
    if show:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
        except Exception as e:
            print("[warn] 打不开 viewer：", e)
            viewer = None

    site1 = resolve_site_name(model, cfg.get("site1","if_distal_site_ball"))
    site2 = resolve_site_name(model, cfg.get("site2","tf_distal_site_ball"))
    qadr_free = _free_qadr(model, cfg.get("freejoint","root"))
    dadr_free = _free_dadr(model, cfg.get("freejoint","root"))

    # ===== Root-lock
    root_lock_enabled = False
    root_lock_pose7 = None
    root_vel_slice = slice(dadr_free, dadr_free+6)

    def lock_root_pose():
        nonlocal root_lock_enabled, root_lock_pose7
        root_lock_pose7 = data.qpos[qadr_free:qadr_free+7].copy()
        root_lock_enabled = True
        print("[lock] root locked at pose7 =", np.array2string(root_lock_pose7, precision=6))

    def _enforce_root_lock():
        if not root_lock_enabled:
            return
        data.qpos[qadr_free:qadr_free+7] = root_lock_pose7
        data.qvel[root_vel_slice] = 0.0

    def _step_with_root_lock(n=1):
        for _ in range(int(n)):
            _enforce_root_lock()
            mj.mj_step(model, data)
            _enforce_root_lock()

    distal_a_body = str(cfg.get("distal_a_body", "if_distal_link"))
    distal_b_body = str(cfg.get("distal_b_body", "th_distal_link"))

    r1 = float(cfg.get("r1",0.004)); r2 = float(cfg.get("r2",0.0055))
    tol = float(cfg.get("tol",1e-4)); tol_fb = float(cfg.get("tol_fallback",5e-4))
    trans_steps=int(cfg.get("trans_steps",10)); rot_steps=int(cfg.get("rot_steps",10))
    penetration_check_steps = int(cfg.get("penetration_check_steps",10))
    force_steps=int(cfg.get("force_steps",300))
    force_ramp=float(cfg.get("force_ramp_ratio",0.2))
    trans_thre=float(cfg.get("trans_thre",0.005))
    angle_thre=float(cfg.get("angle_thre",0.2))
    max_ctrls_per_pair = int(cfg.get("max_ctrls_per_pair", -1))
    approach_gap_if = float(cfg.get("approach_gap_if", cfg.get("approach_gap", 2e-4)))
    approach_gap_th = float(cfg.get("approach_gap_th", 2e-4))

    # —— 工具：判断关节属于哪根手指（if / th）
    def finger_of_joint(jn: str) -> str:
        nm = jn or ""
        if nm in (if_prox_name, if_dist_name) or "if_" in nm:
            return "if"
        if nm in (th_root_name, th_prox_name, th_dist_name) or "th_" in nm or "tf_" in nm:
            return "th"
        return ""

    # —— 阶段A：各指独立“探碰”，先测力与距离，再推进一点（仅用 prox 名称推进，便于靠近）
    def _read_joint(model, data, jname: str) -> float:
        jid = _jid(model, jname)
        adr = int(model.jnt_qposadr[jid])
        return float(data.qpos[adr])

    def probe_until_both_ready() -> Dict[str, Any]:
        info = {
            "used": True,
            "if_ready_step": -1,
            "th_ready_step": -1,
            "both_ready_at": -1,
            "per_step": []
        }
        if probe_steps <= 0 or abs(probe_total_cmd) <= 0:
            info["used"] = False
            return info
        dp_probe = probe_total_cmd / max(1, probe_steps)
        ready_if = False
        ready_th = False
        step_reached = 0
        for s in range(min(precontact_max_steps, probe_steps)):
            fn_if, fn_th = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
            d_if, d_th = min_contact_dist_by_finger(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
            info["per_step"].append({"step": s+1, "fn_if": float(fn_if), "fn_th": float(fn_th),
                                     "d_if": float(d_if), "d_th": float(d_th)})

            # 判定就位
            if (not ready_if) and (fn_if >= precontact_force_eps or d_if <= precontact_dist_eps):
                ready_if = True; info["if_ready_step"] = s+1
            if (not ready_th) and (fn_th >= precontact_force_eps or d_th <= precontact_dist_eps):
                ready_th = True; info["th_ready_step"] = s+1

            if ready_if and ready_th:
                info["both_ready_at"] = s+1
                print(f"    [probe] both fingers ready at step {info['both_ready_at']} "
                      f"(if: fn={fn_if:.3e}, d={d_if:.3e}; th: fn={fn_th:.3e}, d={d_th:.3e})")
                break

            # 推进尚未就位的 prox
            for prox_name, sign in zip([if_prox_name, th_prox_name], [-1.0, -1.0]):
                need_this = ((prox_name == if_prox_name) and (not ready_if)) or \
                            ((prox_name == th_prox_name) and (not ready_th))
                if not need_this:
                    continue
                curp = _read_joint(model, data, prox_name)
                newp = _set_joint_qpos_clamped(model, data, prox_name, curp + sign*dp_probe)
                # CSV 联动 distal
                if prox_name == if_prox_name and (if_x is not None and if_y is not None):
                    newd = _interp_csv_saturated(if_x, if_y, newp)
                    _set_joint_qpos_clamped(model, data, if_dist_name, newd)
                if prox_name == th_prox_name and (th_x is not None and th_y is not None):
                    newd = _interp_csv_saturated(th_x, th_y, newp)
                    _set_joint_qpos_clamped(model, data, th_dist_name, newd)

            _step_with_root_lock(1)
            if viewer is not None:
                _safe_sync(viewer, show_interval, s, viewer_sleep)
            step_reached = s+1

        if info["both_ready_at"] < 0:
            print(f"    [probe] not both ready after {step_reached} steps "
                  f"(epsF={precontact_force_eps:.3e}, epsD={precontact_dist_eps:.3e})")
        return info

    def tighten_by_actuators_freeze_then_together() -> Dict[str, Any]:

        log = {
            "used": True,
            "established": False,
            "both_over_threshold": False,   # 兼容旧字段名
            "established_at_step": -1,
            "consecutive_required": int(tighten_consecutive),
            "per_step": [],
            "final_forces": {"fn_if": 0.0, "fn_th": 0.0},
            "frozen_at": {"if": -1, "th": -1},
            "active": {"if": False, "th": False},
            "detect_force_eps": float(detect_force_eps),
            "detect_dist_eps": float(detect_dist_eps),
            "rounds": []
        }
        if not tighten_joints or tighten_steps <= 0 or abs(tighten_total_cmd) <= 0:
            log["used"] = False
            return log

        extend_enable      = bool(cfg.get("tighten_auto_extend", True))
        extend_cmd_each    = float(cfg.get("tighten_extend_cmd", 0.05))    
        extend_steps_each  = int(cfg.get("tighten_extend_steps", 80))      
        extend_max_rounds  = int(cfg.get("tighten_extend_max_rounds", 1))  
        target_force_if    = float(cfg.get("tighten_target_force_if", contact_force_eps))
        target_force_th    = float(cfg.get("tighten_target_force_th", contact_force_eps))

        # ---- actuator id 映射
        joint2aid = {}
        for nm in list(set(tighten_joints + [if_dist_name, th_dist_name])):  # CSV 可能会用到 distal
            try:
                joint2aid[nm] = _act_id_of_joint(model, nm)
            except Exception:
                pass

        def _set_ctrl_for_joint(jname: str, val: float):
            if jname in joint2aid:
                aid = joint2aid[jname]
                data.ctrl[aid] = _ctrl_clamp(model, aid, float(val))

        # 哪些“手指”参与（任一被选中的关节属于该指即可视为参与）
        def finger_of_joint(jn: str) -> str:
            nm = jn or ""
            if nm in (if_prox_name, if_dist_name) or "if_" in nm: return "if"
            if nm in (th_root_name, th_prox_name, th_dist_name) or "th_" in nm or "tf_" in nm: return "th"
            return ""

        active_if = any(finger_of_joint(jn) == "if" for jn in tighten_joints)
        active_th = any(finger_of_joint(jn) == "th" for jn in tighten_joints)
        log["active"]["if"] = bool(active_if)
        log["active"]["th"] = bool(active_th)

        # 初始目标 ctrl（每关节独立：当前 + sign * tighten_cmd）
        start_ctrl  = {}
        target_ctrl = {}
        per_step_abs = abs(tighten_total_cmd) / max(1, tighten_steps)

        for jn, sgn in zip(tighten_joints, tighten_signs):
            if jn not in joint2aid: continue
            aid = joint2aid[jn]
            cur = float(data.ctrl[aid])
            start_ctrl[jn]  = cur
            target_ctrl[jn] = _ctrl_clamp(model, aid, cur + float(sgn) * float(tighten_total_cmd))

        # 允许 CSV（prox 被选中、distal 未被选中、且有 CSV 曲线）
        allow_if_csv = (if_prox_name in tighten_joints) and (if_dist_name not in tighten_joints) and (if_x is not None and if_y is not None) and (if_dist_name in joint2aid)
        allow_th_csv = (th_prox_name in tighten_joints) and (th_dist_name not in tighten_joints) and (th_x is not None and th_y is not None) and (th_dist_name in joint2aid)

        def one_round(max_steps: int, tag: str):
            """执行一轮向 target_ctrl 逼近；返回 (established: bool)"""
            nonlocal log
            contacted_if = (not active_if)
            contacted_th = (not active_th)
            consecutive_ok = 0
            steps_taken = 0

            for s in range(1, max_steps+1):
                fn_if, fn_th = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
                d_if,  d_th  = min_contact_dist_by_finger(model, data, if_substrs=if_substrs, th_substrs=th_substrs)

                log["per_step"].append({
                    "step": s, "round": tag, "fn_if": float(fn_if), "fn_th": float(fn_th),
                    "d_if": float(d_if), "d_th": float(d_th),
                    "ctrl": {jn: float(data.ctrl[joint2aid[jn]]) for jn in target_ctrl if jn in joint2aid}
                })

                # 冻结早接触的一侧（只对参与的手指）
                if active_if and (not contacted_if) and (fn_if >= detect_force_eps or d_if <= detect_dist_eps):
                    contacted_if = True
                    for jn in tighten_joints:
                        if finger_of_joint(jn) == "if" and jn in target_ctrl and jn in joint2aid:
                            target_ctrl[jn] = float(data.ctrl[joint2aid[jn]])
                    log["frozen_at"]["if"] = log["frozen_at"]["if"] if log["frozen_at"]["if"] > 0 else len(log["per_step"])

                if active_th and (not contacted_th) and (fn_th >= detect_force_eps or d_th <= detect_dist_eps):
                    contacted_th = True
                    for jn in tighten_joints:
                        if finger_of_joint(jn) == "th" and jn in target_ctrl and jn in joint2aid:
                            target_ctrl[jn] = float(data.ctrl[joint2aid[jn]])
                    log["frozen_at"]["th"] = log["frozen_at"]["th"] if log["frozen_at"]["th"] > 0 else len(log["per_step"])

                # 两侧都接触后：判稳（也允许你把目标力设得高于 eps）
                if contacted_if and contacted_th:
                    ok_if = (fn_if >= target_force_if) if active_if else True
                    ok_th = (fn_th >= target_force_th) if active_th else True
                    consecutive_ok = (consecutive_ok + 1) if (ok_if and ok_th) else 0
                    if consecutive_ok >= log["consecutive_required"]:
                        log["established"] = True
                        log["both_over_threshold"] = True
                        log["established_at_step"] = len(log["per_step"])
                        log["final_forces"] = {"fn_if": float(fn_if), "fn_th": float(fn_th)}
                        print(f"    [tighten] ESTABLISHED({tag}) at step {s} (if={fn_if:.3e}, th={fn_th:.3e})")
                        return True

                for jn, sgn in zip(tighten_joints, tighten_signs):
                    if jn not in joint2aid: continue
                    finger = finger_of_joint(jn)
                    if finger == "if" and contacted_if and (active_th and not contacted_th):
                        step = 0.0
                    elif finger == "th" and contacted_th and (active_if and not contacted_if):
                        step = 0.0
                    else:
                        aid = joint2aid[jn]
                        cur = float(data.ctrl[aid])
                        tgt = target_ctrl.get(jn, cur)
                        diff = tgt - cur
                        step = 0.0 if abs(diff) <= 1e-12 else float(np.clip(diff, -per_step_abs, per_step_abs))
                        if abs(step) > 0.0:
                            newv = _ctrl_clamp(model, aid, cur + step)
                            data.ctrl[aid] = newv
                            if jn == if_prox_name and allow_if_csv:
                                newd = _interp_csv_saturated(if_x, if_y, newv)
                                data.ctrl[joint2aid[if_dist_name]] = _ctrl_clamp(model, joint2aid[if_dist_name], newd)
                            if jn == th_prox_name and allow_th_csv:
                                newd = _interp_csv_saturated(th_x, th_y, newv)
                                data.ctrl[joint2aid[th_dist_name]] = _ctrl_clamp(model, joint2aid[th_dist_name], newd)

                _step_with_root_lock(sim_per_step)
                if viewer is not None:
                    _safe_sync(viewer, show_interval, s, viewer_sleep)
                steps_taken += 1

            print(f"    [tighten] round '{tag}' ended (steps={steps_taken}) not established; "
                f"last if/th = {fn_if:.3e}/{fn_th:.3e}, target_if/th={target_force_if:.3e}/{target_force_th:.3e}")
            return False

        ok = one_round(tighten_steps, tag="init")
        log["rounds"].append({"tag":"init","ok":bool(ok)})

        if (not ok) and extend_enable and (extend_cmd_each > 0.0) and (extend_steps_each > 0) and (extend_max_rounds > 0):
            for r in range(1, extend_max_rounds+1):
                for jn, sgn in zip(tighten_joints, tighten_signs):
                    if jn not in joint2aid: continue
                    aid = joint2aid[jn]
                    cur_tgt = target_ctrl.get(jn, float(data.ctrl[aid]))
                    new_tgt = _ctrl_clamp(model, aid, cur_tgt + float(sgn) * float(extend_cmd_each))
                    target_ctrl[jn] = new_tgt
                print(f"    [tighten] auto-extend round {r}: +cmd={extend_cmd_each} (steps={extend_steps_each})")
                ok = one_round(extend_steps_each, tag=f"ext{r}")
                log["rounds"].append({"tag":f"ext{r}","ok":bool(ok)})
                if ok:
                    break

        if not log["established"]:
            last = log["per_step"][-1] if log["per_step"] else {"fn_if":0,"fn_th":0}
            print(f"  [tighten] NOT ESTABLISHED after all rounds "
                f"(last if={last['fn_if']:.3e}, th={last['fn_th']:.3e}, "
                f"need ≥{max(target_force_if,target_force_th):.3e})")
        return log

    def apply_force_with_visuals(body_name: str, f_world3: np.ndarray, steps: int, ramp_ratio: float):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        ft = np.zeros(6, dtype=np.float64); ft[:3] = np.asarray(f_world3, float).reshape(3,)
        ramp = max(1, int(steps*float(np.clip(ramp_ratio,0.0,1.0))))
        # 斜坡阶段
        for i in range(ramp):
            a = (i+1)/float(ramp)
            _enforce_root_lock()
            data.xfrc_applied[bid] = a*ft
            _step_with_root_lock(1)
            if viewer is not None and visualize_force_phase:
                _safe_sync(viewer, show_interval, i, viewer_sleep)
        # 恒定外力阶段
        for j in range(max(0,steps-ramp)):
            _enforce_root_lock()
            data.xfrc_applied[bid] = ft
            _step_with_root_lock(1)
            if viewer is not None and visualize_force_phase:
                _safe_sync(viewer, show_interval, j, viewer_sleep)
        data.xfrc_applied[bid] = np.zeros(6)
        mj.mj_forward(model, data)
        _enforce_root_lock()

    pairs_path = os.path.join(antipodal_dir, f"{antipodal_prefix}{name_stem}.npy")
    if not os.path.isfile(pairs_path):
        raise FileNotFoundError(f"找不到 antipodal 文件：{pairs_path}")
    pairs_b = load_antipodal_pairs(pairs_path)
    if pairs_b.ndim != 3 or pairs_b.shape[1:] != (2,3):
        raise RuntimeError(f"{pairs_path} 不是 (K,2,3) 的数组")
    if not bool(cfg.get("pairs_already_scaled", False)):
        pairs_b = pairs_b * float(FIXED_SCALE)

    K = pairs_b.shape[0]
    topk = int(cfg.get("topk",-1))
    eval_ids = list(range(min(K,topk) if (topk and topk>0) else K))
    qpos5_db, dists_db = load_scan_db(scan_db_path, th_csv=th_csv, if_csv=if_csv, csv_deg=csv_deg)

    out_dir = os.path.join(out_root, name_stem); os.makedirs(out_dir, exist_ok=True)

    total_tests = 0
    total_success = 0
    pair_stats = []

    print(f"\n[{name_stem}] 开始评估 {len(eval_ids)} 个 antipodal pairs...")

    for pair_idx, k in enumerate(eval_ids, start=1):
        print(f"\n{'='*60}\n[Pair {pair_idx}/{len(eval_ids)}] pair_index={k}\n{'='*60}")

        p1_b = pairs_b[k,0].astype(np.float64)
        p2_b = pairs_b[k,1].astype(np.float64)
        seg  = float(np.linalg.norm(p2_b - p1_b))
        vhat = (p2_b - p1_b) / (seg + 1e-12)

        c1_b = p1_b - r1*vhat          # IF 球心
        c2_b = p2_b + r2*vhat          # TH/TF 球心
        L_nominal = seg + r1 + r2

        c1_b_gap = c1_b - approach_gap_if * vhat
        c2_b_gap = c2_b + approach_gap_th * vhat
        L_match  = L_nominal + approach_gap_if + approach_gap_th

        matching_indices, used_tol, fallback = find_all_matching_ctrls(
            qpos5_db, dists_db, L_match, tol, tol_fb, max_ctrls_per_pair
        )
        print(f"[Pair {k}] seg={seg:.6f}  L_nominal={L_nominal:.6f}  "
              f"L_match={L_match:.6f} (gaps: IF={approach_gap_if:.6f}, TH={approach_gap_th:.6f})")

        n_match = len(matching_indices)
        print(f"[Pair {k}] 用 L_match={L_match:.6f} 匹配到 {n_match} 条 "
              f"(tol={used_tol:.2e}, fallback={fallback})")

        pair_success = 0; pair_fail = 0; pair_skip = 0

        for idx_local, db_idx in enumerate(matching_indices, start=1):
            total_tests += 1
            dist_db = float(dists_db[db_idx])
            qpos5   = np.asarray(qpos5_db[db_idx], np.float64).ravel()[:5]
            print(f"\n  [{idx_local}/{n_match}] db_idx={db_idx}, dist={dist_db:.6f}")

            reset_to_initial()
            scene.set_lego_collision_enabled(False, body_prefix="lego")

            try:
                for aid, val in zip(act_ids5, qpos5):
                    data.ctrl[aid] = _ctrl_clamp(model, aid, float(val))
            except Exception as e:
                print(f"  → SKIP: 设置 ctrl 失败: {e}")
                pair_skip += 1
                continue

            for _ in range(150):
                mj.mj_step(model, data)
                _safe_sync(viewer, show_interval, _, viewer_sleep)

            lego_p, lego_R, _ = lego_pose_from_body(model, data, "lego")
            c1_w = world_from_local(c1_b, lego_p, lego_R)
            c2_w = world_from_local(c2_b, lego_p, lego_R)
            c1_w_gap = world_from_local(c1_b_gap, lego_p, lego_R)
            c2_w_gap = world_from_local(c2_b_gap, lego_p, lego_R)

            s2 = get_site_pos(model, data, site2)
            base_pos = data.qpos[qadr_free:qadr_free+3].copy()
            base_quat = data.qpos[qadr_free+3:qadr_free+7].copy()
            T = max(1, trans_steps if viewer else 1)
            delta2 = c2_w_gap - s2
            for i in range(T):
                a = (i+1)/T
                data.qpos[qadr_free:qadr_free+3]   = base_pos + a*delta2
                data.qpos[qadr_free+3:qadr_free+7] = base_quat
                mj.mj_forward(model, data)
                _safe_sync(viewer, show_interval, i, viewer_sleep)
            base_pos = data.qpos[qadr_free:qadr_free+3].copy()

            s1a = get_site_pos(model, data, site1)
            v = s1a - c2_w_gap
            w = c1_w_gap - c2_w_gap
            vn = v/(np.linalg.norm(v)+1e-12)
            wn = w/(np.linalg.norm(w)+1e-12)
            axis = np.cross(vn, wn)
            l = np.linalg.norm(axis)
            dot = float(np.clip(np.dot(vn, wn), -1.0, 1.0))
            ang = float(np.arctan2(l, dot))
            if l < 1e-9:
                tmp = np.array([1,0,0], np.float64)
                if abs(np.dot(tmp, vn)) > 0.9: tmp = np.array([0,1,0], np.float64)
                axis = np.cross(vn, tmp); axis /= (np.linalg.norm(axis)+1e-12)

            if viewer:
                rotN = max(1, rot_steps)
                dR = rodrigues(axis, ang/rotN)
                for rr in range(rotN):
                    base_pos, base_quat = _set_free_about_point(model, data, qadr_free, base_pos, base_quat, dR, c2_w_gap)
                    _safe_sync(viewer, show_interval, rr, viewer_sleep)
            else:
                R_all = rodrigues(axis, ang)
                base_pos, base_quat = _set_free_about_point(model, data, qadr_free, base_pos, base_quat, R_all, c2_w_gap)

            s1f = get_site_pos(model, data, site1)
            s2f = get_site_pos(model, data, site2)
            e1 = float(np.linalg.norm(s1f - c1_w_gap))
            e2 = float(np.linalg.norm(s2f - c2_w_gap))

            if any_contact_between_bodies(model, data, [distal_a_body], [distal_b_body]):
                print(f"  → SKIP: distal self collision")
                pair_skip += 1
                continue

            scene.set_lego_collision_enabled(True, body_prefix="lego")

            lock_root_pose()

            for pp in range(penetration_check_steps):
                _step_with_root_lock(1)
                _safe_sync(viewer, show_interval, pp, viewer_sleep)
            min_dist_now = min_hand_lego_contact_dist(model, data, ("if_","th_","tf_"), "lego")

            if min_dist_now <= -0.001:
                print(f"  → SKIP: hand-lego collision right after align (min_dist={min_dist_now:.6f})")
                pair_skip += 1
                continue

            # —— 探碰
            print(f"  [probe] start (epsF={precontact_force_eps:.3e}, epsD={precontact_dist_eps:.3e})")
            probe_info = probe_until_both_ready()

            # —— 加紧
            print(f"  [tighten] actuator-based start (steps={tighten_steps}, cmd={tighten_total_cmd}, "
                  f"signs={tighten_signs}, force_eps={contact_force_eps:.3e}, "
                  f"detect_epsF={detect_force_eps:.3e}, detect_epsD={detect_dist_eps:.3e})")
            tighten_info = tighten_by_actuators_freeze_then_together()

            fn_if1, fn_th1 = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
            if not (tighten_info.get("both_over_threshold", False) and
                    (fn_if1 >= contact_force_eps) and (fn_th1 >= contact_force_eps)):
                print(f"  → SKIP: no both-finger contact after tighten; if={fn_if1:.3e}, th={fn_th1:.3e}, eps={contact_force_eps:.3e}")
                pair_skip += 1
                continue

            dirs = np.array([
                [-1,0,0], [1,0,0],
                [0,-1,0], [0,1,0],
                [0,0,-1], [0,0,1]
            ], dtype=float)

            base_pose = pose7_of_body(model, data, "lego")
            all_ok   = True
            per_dir  = []
            worst = {"idx": -1, "dir": None, "dp": -1.0, "da": -1.0}

            print(f"  [stability] align errors: e1={e1:.6f}, e2={e2:.6f} | "
                  f"thresholds: trans={trans_thre:.6f}, angle={angle_thre:.6f}")

            for di, d in enumerate(dirs):
                _set_body_pose7(model, data, "lego_freejoint", base_pose)
                pre = pose7_of_body(model, data, "lego")

                apply_force_with_visuals("lego", d*force_N, steps=force_steps, ramp_ratio=force_ramp)

                lat = pose7_of_body(model, data, "lego")
                dp, da = delta_pose(pre, lat)
                violated = []
                if dp >= trans_thre: violated.append("translation")
                if da >= angle_thre: violated.append("rotation")
                ok = (len(violated) == 0)

                if dp > worst["dp"] or (abs(dp-worst["dp"])<1e-12 and da>worst["da"]):
                    worst.update({"idx": di, "dir": d.tolist(), "dp": dp, "da": da})

                per_dir.append({"dir": d.tolist(), "dp": dp, "da": da, "pass": bool(ok)})

                if not ok:
                    blown = (dp >= 10.0*trans_thre) or (da >= 10.0*angle_thre)
                    classification = "blown_away" if blown else "drifted_over_threshold"
                    print(f"  → FAIL(stability) at dir#{di}={d.tolist()} | "
                          f"violated={violated} | dp={dp:.6f} (th={trans_thre:.6f}), "
                          f"da={da:.4f}rad (th={angle_thre:.4f}) | class={classification}")
                    all_ok = False
                    break

            status = "success" if all_ok else "fail"
            if status == "success":
                print(f"  → SUCCESS | worst_dir#{worst['idx']}={worst['dir']} "
                      f"| max_dp={worst['dp']:.6f}, max_da={worst['da']:.4f}rad")
                pair_success += 1
                total_success += 1
            else:
                print(f"  → FAIL    | worst_dir#{worst['idx']}={worst['dir']} "
                      f"| max_dp={worst['dp']:.6f}, max_da={worst['da']:.4f}rad")
                pair_fail += 1

            root7_world = data.qpos[qadr_free:qadr_free+7].copy()
            root_body = _body_name_of_joint(model, cfg.get("freejoint", "root"))
            root7_in_lego = rel_pose7_bodyA_wrt_bodyB(model, data, root_body, "lego")
            lego7_world = pose7_of_body(model, data, "lego")

            if status == "success":
                out = {
                    "status": status,
                    "lego": name_stem,
                    "pair_index": int(k),
                    "ctrl_db_index": int(db_idx),
                    "qpos5": qpos5.tolist(),
                    "qpos_names": joint_names5,
                    # "ctrl_dist": dist_db,
                    # "ctrl_error": float(abs(dist_db - L_match)),
                    "errors_align": {"e1": e1, "e2": e2},
                    # "probe": probe_info,
                    # "tighten": tighten_info,
                    "stability_per_dir": per_dir,
                    "root_pose7_in_world(xyzw)": root7_world.tolist(),
                    "root_pose7_in_lego(xyzw)": root7_in_lego.tolist(),
                    "lego_pose7_in_world(xyzw)": lego7_world.tolist()

                }
                np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.succ.npy"),
                        out, allow_pickle=True)

        pair_total = pair_success + pair_fail + pair_skip
        pair_stats.append({
            "pair_index": int(k),
            "success": pair_success,
            "fail": pair_fail,
            "skip": pair_skip,
            "total": pair_total,
            "success_rate": (pair_success/pair_total) if pair_total>0 else 0.0
        })
        print(f"\n[Pair {k} Summary] success={pair_success}, fail={pair_fail}, skip={pair_skip}, total={pair_total}")

    overall_rate = total_success / total_tests if total_tests > 0 else 0.0
    summary = {
        "lego": name_stem,
        "total_pairs": len(eval_ids),
        "total_tests": total_tests,
        "total_success": total_success,
        "success_rate": overall_rate,
        "per_pair_stats": pair_stats,
    }
    with open(os.path.join(out_dir, f"{name_stem}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{name_stem}] 总体统计: pairs={len(eval_ids)} | tests={total_tests} | success={total_success} ({100*overall_rate:.2f}%)")

    if viewer is not None:
        print("[info] 关闭窗口以继续下一个 LEGO")
        try:
            while True: time.sleep(0.2)
        except KeyboardInterrupt:
            pass


def main():
    ap = argparse.ArgumentParser(
        description="评估: 探碰(独立) + 冻结→同步加紧(连续判稳) + 任意关节集合 + CSV 联动 + root锁定 + 外力扰动检验"
    )
    ap.add_argument("--config", default="configs/lego.yaml")
    ap.add_argument("--show", action="store_true", help="每个 LEGO 启动独立 viewer")
    ap.add_argument("--debug", action="store_true", help="失败时打印关节/位姿/site/接触等详细信息")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["_config_path"] = os.path.abspath(args.config)

    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    for k in ["xml","obj_mesh_root","antipodal_dir","scan_db","out_dir","lego_list_json","th_csv","if_csv"]:
        if k in cfg:
            cfg[k] = _abs_join(cfg_dir, str(cfg[k]))

    if "lego_list_json" not in cfg:
        print("配置中缺少 lego_list_json"); sys.exit(1)
    lego_names = load_lego_list(cfg["lego_list_json"])
    if not lego_names:
        print("lego_list_json 为空"); sys.exit(1)

    print("评估模式: 探碰(独立) → 冻结→同步加紧(连续判稳) → 外力扰动检验")
    print(f"FIXED_SCALE = {FIXED_SCALE}")

    for i, name in enumerate(lego_names, start=1):
        print(f"\n{'#'*60}\n# [{i}/{len(lego_names)}] {name}\n{'#'*60}")
        try:
            eval_one_lego(cfg, name, show=args.show, debug=args.debug)
        except Exception as e:
            print(f"\n[ERROR] {name} 评估失败: {e}")
            import traceback; traceback.print_exc()
            continue

    print("\n[DONE] 全部 LEGO 测试完成。")

if __name__ == "__main__":
    main()