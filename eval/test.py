#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, json, time, argparse
from typing import Any, Dict, List, Tuple
import numpy as np
import mujoco as mj

FIXED_SCALE = 0.0004

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
            print(f"[info] site '{name}' 不存在，自动改 '{alt}'")
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

def _aid_for_joint(model, joint_id):
    for ai in range(model.nu):
        if int(model.actuator_trnid[ai,0]) == int(joint_id):
            return int(ai)
    return -1

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
    for ext in ("*.obj","*.stl","*.ply"):
        ret += glob.glob(os.path.join(p, "**", ext), recursive=True)
    ret = [os.path.abspath(x) for x in ret if _is_mesh_file(x)]
    if not ret:
        raise FileNotFoundError(f"未在目录中找到网格: {p}")
    return sorted(ret)

def _safe_mesh_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
    return base or "mesh"

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
            body.add_geom(name=f"{body_name}_collision_{mname}",
                          type=self.mj.mjtGeom.mjGEOM_MESH, meshname=mname,
                          density=density, friction=friction, contype=1, conaffinity=1)
        print(f"[lego] added {len(files)} mesh(es), fixed scale={FIXED_SCALE}")
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

def _load_csv_pairs(path, deg=False):
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
    if not xs: raise ValueError(f"empty or invalid csv: {path}")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if deg:
        x = np.deg2rad(x); y = np.deg2rad(y)
    return x, y

def _free_qadr(model, name="root"):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0 or model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"freejoint '{name}' 不存在或不是 freejoint")
    return int(model.jnt_qposadr[jid])

def _set_body_pose7(model, data, freejoint_name: str, pose7: np.ndarray):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, freejoint_name)
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr:adr+7] = np.asarray(pose7, float).reshape(7,)
    mj.mj_forward(model, data)

def min_hand_lego_contact_dist(model, data, hand_substrs=("if_","th_"), lego_body_prefix="lego") -> float:
    hand_bids = set()
    lego_bids = set()
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

def any_contact_between_bodies(model, data, bodyA_names: List[str], bodyB_names: List[str]) -> bool:
    A = set()
    B = set()
    for nm in bodyA_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0: A.add(int(bid))
    for nm in bodyB_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0: B.add(int(bid))
    if not A or not B:
        raise RuntimeError(f"distal body 名不存在: A={bodyA_names}, B={bodyB_names}")
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

def apply_pure_force_on_body(model, data, body_name: str, f_world3: np.ndarray, steps: int, ramp_ratio: float):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    ft = np.zeros(6, dtype=np.float64); ft[:3] = np.asarray(f_world3, float).reshape(3,)
    ramp = max(1, int(steps*float(np.clip(ramp_ratio,0.0,1.0))))
    for i in range(ramp):
        a = (i+1)/float(ramp); data.xfrc_applied[bid] = a*ft; mj.mj_step(model, data)
    for _ in range(max(0,steps-ramp)):
        data.xfrc_applied[bid] = ft; mj.mj_step(model, data)
    data.xfrc_applied[bid] = np.zeros(6); mj.mj_forward(model, data)

def delta_pose(a7, b7):
    pa = np.asarray(a7[:3]); qa = np.asarray(a7[3:]); qa/= (np.linalg.norm(qa)+1e-12)
    pb = np.asarray(b7[:3]); qb = np.asarray(b7[3:]); qb/= (np.linalg.norm(qb)+1e-12)
    dp = float(np.linalg.norm(pb-pa))
    qc = np.empty(4); mj.mju_mulQuat(qc, qb, np.array([qa[0],-qa[1],-qa[2],-qa[3]]))
    da = 2.0*float(np.arccos(np.clip(qc[0], -1.0, 1.0)))
    if da > np.pi: da = 2*np.pi - da
    return dp, da

def _set_hinge_qpos(model, data, joint_name, value):
    jid = _jid(model, joint_name)
    if model.jnt_type[jid] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
        raise RuntimeError(f"joint '{joint_name}' must be hinge/slide")
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr] = float(value)

def _ctrl_range(model, joint_name):
    jid = _jid(model, joint_name)
    aid = _aid_for_joint(model, jid)
    if aid < 0: raise RuntimeError(f"no actuator for joint '{joint_name}'")
    lo, hi = model.actuator_ctrlrange[aid]
    if int(model.jnt_limited[jid]):
        jlo, jhi = model.jnt_range[jid]
        lo = max(lo, jlo); hi = min(hi, jhi)
    return float(lo), float(hi)

def _pick_from_curves_for_L(model, data, site1, site2, th_root_name, th_prox_name, th_dist_name, if_prox_name, if_dist_name, th_curve, if_curve, th_root_steps, L, tol, tol_fb):
    sid1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site1)
    sid2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site2)
    if sid1 < 0 or sid2 < 0: raise RuntimeError("site not found")
    root_lo, root_hi = _ctrl_range(model, th_root_name)
    roots = np.linspace(root_lo, root_hi, max(1,int(th_root_steps)), dtype=np.float64)
    th_px, th_dx = th_curve
    if_px, if_dx = if_curve
    best = None
    base_qpos = data.qpos.copy(); base_qvel = data.qvel.copy()
    for r in roots:
        for i in range(len(th_px)):
            for j in range(len(if_px)):
                data.qpos[:] = base_qpos; data.qvel[:] = base_qvel
                _set_hinge_qpos(model, data, th_root_name, r)
                _set_hinge_qpos(model, data, th_prox_name, th_px[i])
                _set_hinge_qpos(model, data, th_dist_name, th_dx[i])
                _set_hinge_qpos(model, data, if_prox_name, if_px[j])
                _set_hinge_qpos(model, data, if_dist_name, if_dx[j])
                mj.mj_forward(model, data)
                p1 = data.site_xpos[sid1]; p2 = data.site_xpos[sid2]
                d = float(np.linalg.norm(p1-p2))
                err = abs(d-L)
                if best is None or err < best[0]:
                    best = (err, r, th_px[i], th_dx[i], if_px[j], if_dx[j], d)
                    if err < tol: break
            if best is not None and best[0] < tol: break
        if best is not None and best[0] < tol: break
    if best is None or best[0] >= tol_fb:
        pass
    return best[1], best[2], best[3], best[4], best[5], best[6], best[0]

def eval_one_lego(cfg: Dict[str,Any], lego_name_raw: str, show=False):
    cfg_dir = os.path.dirname(os.path.abspath(cfg["_config_path"]))
    xml_path       = _abs_join(cfg_dir, str(cfg["xml"]))
    obj_mesh_root  = _abs_join(cfg_dir, str(cfg["obj_mesh_root"]))
    antipodal_dir  = _abs_join(cfg_dir, str(cfg["antipodal_dir"]))
    out_root       = _abs_join(cfg_dir, str(cfg["out_dir"]))
    antipodal_prefix = str(cfg.get("antipodal_prefix","")).strip()
    th_csv = _abs_join(cfg_dir, str(cfg.get("th_csv","th_curve.csv")))
    if_csv = _abs_join(cfg_dir, str(cfg.get("if_csv","if_curve.csv")))
    csv_deg = bool(cfg.get("csv_deg", False))
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
    qpos_init = data.qpos.copy()
    qvel_init = data.qvel.copy()
    ctrl_init = data.ctrl.copy()
    def reset_to_initial():
        data.qpos[:] = qpos_init
        data.qvel[:] = qvel_init
        data.ctrl[:] = ctrl_init
        mj.mj_forward(model, data)
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
    distal_a_body = str(cfg.get("distal_a_body", "if_distal_link"))
    distal_b_body = str(cfg.get("distal_b_body", "th_distal_link"))
    r1 = float(cfg.get("r1",0.004)); r2 = float(cfg.get("r2",0.0055))
    tol = float(cfg.get("tol",1e-4)); tol_fb = float(cfg.get("tol_fallback",5e-4))
    trans_steps=int(cfg.get("trans_steps",60)); rot_steps=int(cfg.get("rot_steps",120))
    penetration_check_steps = int(cfg.get("penetration_check_steps",10))
    tighten_joints = list(cfg.get("tighten_joints", ["th_proximal_link"]))
    tighten_signs  = [float(x) for x in str(cfg.get("tighten_signs","-1")).split(",")]
    tighten_cmd    = float(cfg.get("tighten_cmd", 0.05))
    tighten_steps  = int(cfg.get("tighten_steps", 60))
    wait_after_tighten = int(cfg.get("wait_after_tighten", 60))
    force_N=float(cfg.get("force_N",0.1)); force_steps=int(cfg.get("force_steps",300))
    force_ramp=float(cfg.get("force_ramp_ratio",0.2)); trans_thre=float(cfg.get("trans_thre",0.005))
    angle_thre=float(cfg.get("angle_thre",0.2))
    th_root_name = str(cfg.get("th_root_joint","th_root_link"))
    th_prox_name = str(cfg.get("th_prox_joint","th_proximal_link"))
    th_dist_name = str(cfg.get("th_dist_joint","th_distal_link"))
    if_prox_name = str(cfg.get("if_prox_joint","if_proximal_link"))
    if_dist_name = str(cfg.get("if_dist_joint","if_distal_link"))
    th_root_steps = int(cfg.get("th_root_steps", 31))
    pairs_path = os.path.join(antipodal_dir, f"{antipodal_prefix}{name_stem}.npy")
    if not os.path.isfile(pairs_path):
        raise FileNotFoundError(f"找不到 antipodal 文件：{pairs_path}")
    payload = np.load(pairs_path, allow_pickle=True)
    if isinstance(payload, dict) or (hasattr(payload, "item") and isinstance(payload.item(), dict)):
        pairs_b = np.asarray((payload if isinstance(payload, dict) else payload.item())["pairs"], np.float32)
    else:
        pairs_b = np.asarray(payload, np.float32)
    if pairs_b.ndim != 3 or pairs_b.shape[1:] != (2,3):
        raise RuntimeError(f"{pairs_path} 不是 (K,2,3) 的数组")
    if not bool(cfg.get("pairs_already_scaled", False)):
        pairs_b = pairs_b * float(FIXED_SCALE)
    th_curve = _load_csv_pairs(th_csv, deg=csv_deg)
    if_curve  = _load_csv_pairs(if_csv,  deg=csv_deg)
    K = pairs_b.shape[0]
    topk = int(cfg.get("topk",-1))
    eval_ids = list(range(min(K,topk) if (topk and topk>0) else K))
    out_dir = os.path.join(out_root, name_stem); os.makedirs(out_dir, exist_ok=True)
    lego_p, lego_R, _ = lego_pose_from_body(model, data, "lego")
    def draw_spheres(p1_b, p2_b, c1_b, c2_b):
        if viewer is None: return
        pL, RL, _ = lego_pose_from_body(model, data, "lego")
        p1w = world_from_local(p1_b, pL, RL)
        p2w = world_from_local(p2_b, pL, RL)
        c1w = world_from_local(c1_b, pL, RL)
        c2w = world_from_local(c2_b, pL, RL)
        s1w = get_site_pos(model, data, site1)
        s2w = get_site_pos(model, data, site2)
        scn = viewer.user_scn; scn.ngeom = 0
        def _sphere(pos, r, rgba):
            g = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.array([r,r,r],np.float32), pos, np.eye(3,dtype=np.float32).reshape(-1), rgba)
            scn.ngeom += 1
        _sphere(p1w, 0.0025, (1,1,0,0.9))
        _sphere(p2w, 0.0025, (1,1,0,0.9))
        _sphere(c1w, r1, (0,1,0,0.6))
        _sphere(c2w, r2, (0,1,0,0.6))
        _sphere(s1w, max(1e-4,0.6*r1), (1,0,0,0.9))
        _sphere(s2w, max(1e-4,0.6*r2), (1,0,0,0.9))
        viewer.sync()
    succ = 0; total = 0
    for k in eval_ids:
        total += 1
        p1_b = pairs_b[k,0].astype(np.float64)
        p2_b = pairs_b[k,1].astype(np.float64)
        seg  = float(np.linalg.norm(p2_b - p1_b))
        vhat = (p2_b - p1_b) / (seg + 1e-12)
        c1_b = p1_b - r1*vhat
        c2_b = p2_b + r2*vhat
        L    = seg + r1 + r2
        best_th_root, best_th_prox, best_th_dist, best_if_prox, best_if_dist, best_d, best_err = _pick_from_curves_for_L(
            model, data, site1, site2,
            th_root_name, th_prox_name, th_dist_name,
            if_prox_name, if_dist_name,
            th_curve, if_curve, th_root_steps, L, tol, tol_fb
        )
        data.qpos[:] = qpos_init; data.qvel[:] = qvel_init
        _set_hinge_qpos(model, data, th_root_name, best_th_root)
        _set_hinge_qpos(model, data, th_prox_name, best_th_prox)
        _set_hinge_qpos(model, data, th_dist_name, best_th_dist)
        _set_hinge_qpos(model, data, if_prox_name, best_if_prox)
        _set_hinge_qpos(model, data, if_dist_name, best_if_dist)
        mj.mj_forward(model, data)
        c1_w = world_from_local(c1_b, lego_p, lego_R)
        c2_w = world_from_local(c2_b, lego_p, lego_R)
        s1 = get_site_pos(model, data, site1)
        base_pos = data.qpos[qadr_free:qadr_free+3].copy()
        base_quat = data.qpos[qadr_free+3:qadr_free+7].copy()
        T = max(1, trans_steps if viewer else 1)
        delta = c1_w - s1
        for i in range(T):
            a = (i+1)/T
            data.qpos[qadr_free:qadr_free+3]   = base_pos + a*delta
            data.qpos[qadr_free+3:qadr_free+7] = base_quat
            mj.mj_forward(model, data)
            draw_spheres(p1_b,p2_b,c1_b,c2_b)
        base_pos = data.qpos[qadr_free:qadr_free+3].copy()
        s2a = get_site_pos(model, data, site2)
        v = s2a - c1_w; w = c2_w - c1_w
        vn = v/(np.linalg.norm(v)+1e-12); wn = w/(np.linalg.norm(w)+1e-12)
        axis = np.cross(vn, wn); l = np.linalg.norm(axis)
        dot = float(np.clip(np.dot(vn, wn), -1.0, 1.0))
        ang = float(np.arctan2(l, dot))
        if l < 1e-9:
            tmp = np.array([1,0,0], np.float64)
            if abs(np.dot(tmp, vn)) > 0.9: tmp = np.array([0,1,0], np.float64)
            axis = np.cross(vn, tmp); axis /= (np.linalg.norm(axis)+1e-12)
        if viewer:
            rotN = max(1, rot_steps)
            dR = rodrigues(axis, ang/rotN)
            for _ in range(rotN):
                q_rot = mat_to_quat_xyzw(dR)
                q_new = np.empty(4); mj.mju_mulQuat(q_new, q_rot, base_quat)
                p0 = np.asarray(c1_w, float)
                p_new = p0 + dR @ (base_pos - p0)
                data.qpos[qadr_free:qadr_free+3]   = p_new
                data.qpos[qadr_free+3:qadr_free+7] = q_new
                mj.mj_forward(model, data)
                base_pos, base_quat = p_new.copy(), q_new.copy()
                draw_spheres(p1_b,p2_b,c1_b,c2_b)
        else:
            R_all = rodrigues(axis, ang)
            q_rot = mat_to_quat_xyzw(R_all)
            q_new = np.empty(4); mj.mju_mulQuat(q_new, q_rot, base_quat)
            p0 = np.asarray(c1_w, float)
            p_new = p0 + R_all @ (base_pos - p0)
            data.qpos[qadr_free:qadr_free+3]   = p_new
            data.qpos[qadr_free+3:qadr_free+7] = q_new
            mj.mj_forward(model, data)
        s1f = get_site_pos(model, data, site1)
        s2f = get_site_pos(model, data, site2)
        e1 = float(np.linalg.norm(s1f - c1_w)); e2 = float(np.linalg.norm(s2f - c2_w))
        if any_contact_between_bodies(model, data, [distal_a_body], [distal_b_body]):
            out = {
                "status": "skip_distal_self_collision",
                "lego": name_stem,
                "pair_index": int(k),
                "errors_align": {"e1": e1, "e2": e2},
                "best_match": {
                    "th_root": best_th_root,
                    "th_prox": best_th_prox,
                    "th_dist": best_th_dist,
                    "if_prox": best_if_prox,
                    "if_dist": best_if_dist,
                    "dist": best_d,
                    "abs_err": best_err
                }
            }
            np.save(os.path.join(out_dir, f"{k:06d}.skip.npy"), out, allow_pickle=True)
            reset_to_initial()
            continue
        scene.set_lego_collision_enabled(True, body_prefix="lego")
        for _ in range(penetration_check_steps):
            mj.mj_step(model, data)
        min_dist = min_hand_lego_contact_dist(model, data, ("if_","th_"), "lego")
        if min_dist <= 0.0:
            out = {
                "status": "skip_hand_lego_collision",
                "lego": name_stem,
                "pair_index": int(k),
                "errors_align": {"e1": e1, "e2": e2},
                "min_contact_dist_after_align": float(min_dist),
                "best_match": {
                    "th_root": best_th_root,
                    "th_prox": best_th_prox,
                    "th_dist": best_th_dist,
                    "if_prox": best_if_prox,
                    "if_dist": best_if_dist,
                    "dist": best_d,
                    "abs_err": best_err
                }
            }
            np.save(os.path.join(out_dir, f"{k:06d}.skip.npy"), out, allow_pickle=True)
            reset_to_initial()
            continue
        if tighten_joints and tighten_steps>0 and abs(tighten_cmd)>0:
            jids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn) for jn in tighten_joints]
            aids = []
            for j in jids:
                a = -1
                for ai in range(model.nu):
                    if int(model.actuator_trnid[ai,0]) == int(j):
                        a = int(ai); break
                aids.append(a)
            signs = (tighten_signs + [tighten_signs[-1]])[:len(tighten_joints)]
            for t in range(tighten_steps):
                a = (t+1)/float(tighten_steps)
                for si, ai in zip(signs, aids):
                    if ai >= 0:
                        data.ctrl[int(ai)] += float(a * si * (-abs(tighten_cmd)))
                mj.mj_step(model, data)
            for _ in range(wait_after_tighten):
                mj.mj_step(model, data)
        dirs = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]], dtype=float)
        base_pose = pose7_of_body(model, data, "lego")
        all_ok = True; per_dir = []
        for d in dirs:
            _set_body_pose7(model, data, "lego_freejoint", base_pose)
            pre = pose7_of_body(model, data, "lego")
            apply_pure_force_on_body(model, data, "lego", d*force_N, steps=force_steps, ramp_ratio=force_ramp)
            lat = pose7_of_body(model, data, "lego")
            dp, da = delta_pose(pre, lat)
            ok = (dp < trans_thre) and (da < angle_thre)
            per_dir.append({"dp": dp, "da": da, "pass": bool(ok), "dir": d.tolist()})
            if not ok:
                all_ok = False; break
        status = "success" if all_ok else "fail"
        if status == "success": succ += 1
        root7 = data.qpos[qadr_free:qadr_free+7].copy()
        out = {
            "status": status,
            "lego": name_stem,
            "pair_index": int(k),
            "r1": r1, "r2": r2, "green_distance": L,
            "errors_align": {"e1": e1, "e2": e2},
            "min_contact_dist_after_align": float(min_dist),
            "root_pose7": root7.tolist(),
            "best_match": {
                "th_root": best_th_root,
                "th_prox": best_th_prox,
                "th_dist": best_th_dist,
                "if_prox": best_if_prox,
                "if_dist": best_if_dist,
                "dist": best_d,
                "abs_err": best_err
            },
            "stability": {
                "pass_all": bool(all_ok),
                "force_N": force_N, "force_steps": force_steps, "force_ramp_ratio": force_ramp,
                "per_direction": per_dir
            }
        }
        np.save(os.path.join(out_dir, f"{k:06d}.{('succ' if status=='success' else 'fail')}.npy"),
                out, allow_pickle=True)
    with open(os.path.join(out_dir, f"{name_stem}.summary.json"), "w", encoding="utf-8") as f:
        json.dump({"lego": name_stem, "success": succ, "total": total,
                   "rate": (succ/total if total>0 else 0.0)}, f, ensure_ascii=False, indent=2)
    print(f"[{name_stem}] success={succ}/{total} ({(100.0*succ/total if total>0 else 0.0):.2f}%)")
    if viewer is not None:
        print("[info] 关闭当前 LEGO 的窗口以继续…")
        try:
            while True: time.sleep(0.2)
        except KeyboardInterrupt:
            pass

def main():
    ap = argparse.ArgumentParser("Antipodal batch eval with CSV curves for joints (fixed scale=0.0004)")
    ap.add_argument("--config", default="configs/lego_eval.yaml")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cfg["_config_path"] = os.path.abspath(args.config)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    for k in ["xml","obj_mesh_root","antipodal_dir","out_dir","lego_list_json","th_csv","if_csv"]:
        if k in cfg:
            cfg[k] = _abs_join(cfg_dir, str(cfg[k]))
    if "lego_list_json" not in cfg:
        print("配置中缺少 lego_list_json"); sys.exit(1)
    lego_names = load_lego_list(cfg["lego_list_json"])
    if not lego_names:
        print("lego_list_json 为空"); sys.exit(1)
    print(f"将评估 {len(lego_names)} 个 LEGO；固定缩放 FIXED_SCALE = {FIXED_SCALE}")
    for i, name in enumerate(lego_names, start=1):
        print(f"\n===== [{i}/{len(lego_names)}] {name} =====")
        eval_one_lego(cfg, name, show=args.show)
    print("\n[done] 全部 LEGO 测试完成。")

if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os, sys, glob, json, time, argparse
# from typing import Any, Dict, List, Tuple
# import numpy as np
# import mujoco as mj

# FIXED_SCALE = 0.0004
# CONTACT_DUMP_MAX = 32

# def _act_id_of_joint(model, jname):
#     jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
#     for aid in range(model.nu):
#         if int(model.actuator_trnid[aid,0]) == jid:
#             return aid
#     raise RuntimeError(f"no actuator for joint {jname}")

# def _ctrl_clamp(model, aid, val):
#     lo = float(model.actuator_ctrlrange[aid,0])
#     hi = float(model.actuator_ctrlrange[aid,1])
#     return float(np.clip(val, lo, hi))


# def _simple_yaml_load(text: str) -> Dict[str, Any]:
#     out: Dict[str, Any] = {}
#     for raw in text.splitlines():
#         s = raw.strip()
#         if not s or s.startswith("#"): continue
#         if "#" in s: s = s.split("#",1)[0].rstrip()
#         if ":" not in s: continue
#         k, v = s.split(":", 1)
#         key, val = k.strip(), v.strip()
#         if val.startswith(("'", '"')) and val.endswith(("'", '"')) and len(val) >= 2:
#             out[key] = val[1:-1]; continue
#         low = val.lower()
#         if low in ("true","false"): out[key] = (low=="true"); continue
#         try:
#             if "." in val or "e" in low: out[key] = float(val)
#             else: out[key] = int(val)
#             continue
#         except Exception:
#             pass
#         out[key] = val
#     return out

# def load_cfg(path: str) -> Dict[str, Any]:
#     with open(path, "r", encoding="utf-8") as f:
#         txt = f.read()
#     if path.endswith((".yaml",".yml")):
#         try:
#             import yaml
#             return yaml.safe_load(txt)
#         except Exception:
#             return _simple_yaml_load(txt)
#     return json.loads(txt)

# def _abs_join(cfg_dir: str, p: str) -> str:
#     return p if os.path.isabs(p) else os.path.abspath(os.path.join(cfg_dir, p))

# def load_lego_list(json_path: str) -> List[str]:
#     with open(json_path, "r", encoding="utf-8") as f:
#         obj = json.load(f)
#     if isinstance(obj, list):
#         return [str(x).strip() for x in obj if str(x).strip()]
#     if isinstance(obj, dict):
#         for key in ("lego_names","names"):
#             if key in obj and isinstance(obj[key], list):
#                 return [str(x).strip() for x in obj[key] if str(x).strip()]
#     raise ValueError(f"JSON 格式不识别：{json_path}")

# # ====================== 数学/工具 ======================

# def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
#     q = np.empty(4, dtype=np.float64)
#     mj.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(9))
#     return q

# def rodrigues(axis, angle):
#     a = np.asarray(axis, np.float64); n = np.linalg.norm(a)
#     if n < 1e-12: return np.eye(3)
#     a = a/n
#     K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]], np.float64)
#     c, s = np.cos(angle), np.sin(angle)
#     return np.eye(3)+s*K+(1-c)*(K@K)

# def resolve_site_name(model, name: str) -> str:
#     sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
#     if sid >= 0: return name
#     if "_ball" in name:
#         alt = name.replace("_ball","_a")
#         if mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, alt) >= 0:
#             return alt
#     raise RuntimeError(f"site '{name}' not found")

# def get_site_pos(model, data, site_name: str):
#     sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
#     if sid < 0: raise RuntimeError(f"site '{site_name}' not found")
#     return data.site_xpos[sid].copy()

# def _jid(model, name):
#     jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
#     if jid < 0: raise RuntimeError(f"joint '{name}' not found")
#     return int(jid)

# def lego_pose_from_body(model, data, body_name: str):
#     bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
#     if bid < 0: raise RuntimeError(f"lego body '{body_name}' not found")
#     p = data.xpos[bid].copy()
#     R = data.xmat[bid].reshape(3,3).copy()
#     return p, R, bid

# def world_from_local(p_local, p_w, R_w):
#     return p_w + R_w @ p_local

# def _is_mesh_file(p: str) -> bool:
#     return os.path.isfile(p) and os.path.splitext(p)[1].lower() in {".obj",".stl",".ply"}

# def _find_mesh_files(root_or_file: str) -> List[str]:
#     p = os.path.abspath(root_or_file)
#     if _is_mesh_file(p):
#         return [p]
#     if not os.path.isdir(p):
#         raise FileNotFoundError(p)
#     ret = []
#     for ext in ("*.obj","*.stl",".ply","*.PLY","*.OBJ","*.STL"):
#         if ext.startswith("."):  # 容错
#             ext = "*" + ext
#         ret += glob.glob(os.path.join(p, "**", ext), recursive=True)
#     ret = [os.path.abspath(x) for x in ret if _is_mesh_file(x)]
#     if not ret:
#         raise FileNotFoundError(f"未在目录中找到网格: {p}")
#     return sorted(ret)

# def _safe_mesh_name(path: str) -> str:
#     base = os.path.splitext(os.path.basename(path))[0]
#     base = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
#     return base or "mesh"

# def _ensure_bin_stl(fp: str) -> str:
#     if not fp.lower().endswith(".stl"):
#         return fp
#     try:
#         import trimesh
#         m = trimesh.load(fp, force="mesh")
#         if isinstance(m, trimesh.Scene): m = m.dump().sum()
#         outp = fp[:-4] + "_bin.stl"
#         m.export(outp, file_type="stl")
#         return outp
#     except Exception:
#         return fp

# # ====================== 关节/接触工具 ======================

# def _get_joint_range(model, jname: str) -> Tuple[float, float, bool]:
#     jid = _jid(model, jname)
#     limited = bool(int(model.jnt_limited[jid])) if hasattr(model, "jnt_limited") else True
#     # 兼容 MuJoCo 2.x (扁平) 与 3.x (nx2)
#     if hasattr(model.jnt_range, "ndim") and getattr(model.jnt_range, "ndim", 1) == 2:
#         lo = float(model.jnt_range[jid, 0])
#         hi = float(model.jnt_range[jid, 1])
#     else:
#         lo = float(model.jnt_range[2*jid + 0])
#         hi = float(model.jnt_range[2*jid + 1])
#     return lo, hi, limited

# def _set_joint_qpos_clamped(model, data, jname: str, value: float) -> float:
#     jid = _jid(model, jname)
#     adr = int(model.jnt_qposadr[jid])
#     lo, hi, limited = _get_joint_range(model, jname)
#     val = float(value)
#     if limited and (lo < hi):
#         val = min(max(val, lo), hi)
#     data.qpos[adr] = val
#     return val

# def min_hand_lego_contact_dist(model, data, hand_substrs=("if_","th_","tf_"), lego_body_prefix="lego") -> float:
#     hand_bids = set(); lego_bids = set()
#     for bid in range(model.nbody):
#         name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
#         if any(s in name for s in hand_substrs): hand_bids.add(bid)
#         if name.startswith(lego_body_prefix): lego_bids.add(bid)
#     if not hand_bids or not lego_bids:
#         return float("+inf")
#     mn = float("+inf")
#     for i in range(int(data.ncon)):
#         c = data.contact[i]
#         b1 = int(model.geom_bodyid[int(c.geom1)])
#         b2 = int(model.geom_bodyid[int(c.geom2)])
#         if (b1 in hand_bids and b2 in lego_bids) or (b2 in hand_bids and b1 in lego_bids):
#             mn = min(mn, float(c.dist))
#     return mn

# def _collect_bodyids_by_substrs(model, substrs: Tuple[str, ...]) -> set:
#     out = set()
#     for bid in range(model.nbody):
#         name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
#         if any(s in name for s in substrs):
#             out.add(bid)
#     return out

# def sum_two_finger_normal_forces(model, data,
#                                  if_substrs=("if_",),
#                                  th_substrs=("th_","tf_"),
#                                  lego_body_prefix="lego") -> Tuple[float, float]:
#     """
#     统计 if_* 与 LEGO、th_/tf_* 与 LEGO 的法向力（取绝对值相加）。
#     需要在 mj_step 之后调用。
#     """
#     if_bids = _collect_bodyids_by_substrs(model, if_substrs)
#     th_bids = _collect_bodyids_by_substrs(model, th_substrs)
#     lego_bids = set()
#     for bid in range(model.nbody):
#         name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
#         if name.startswith(lego_body_prefix): lego_bids.add(bid)

#     fn_if = 0.0
#     fn_th = 0.0
#     for i in range(int(data.ncon)):
#         c = data.contact[i]
#         g1 = int(c.geom1); g2 = int(c.geom2)
#         b1 = int(model.geom_bodyid[g1]); b2 = int(model.geom_bodyid[g2])
#         addr = int(c.efc_address)
#         if addr < 0:  # 不可约束的碰撞
#             continue
#         f_n = abs(float(data.efc_force[addr]))  # 法向力取绝对值

#         # if_* ↔ lego
#         if ((b1 in if_bids and b2 in lego_bids) or (b2 in if_bids and b1 in lego_bids)):
#             fn_if += f_n
#         # th_/tf_* ↔ lego
#         if ((b1 in th_bids and b2 in lego_bids) or (b2 in th_bids and b1 in lego_bids)):
#             fn_th += f_n

#     return float(fn_if), float(fn_th)

# def any_contact_between_bodies(model, data, bodyA_names: List[str], bodyB_names: List[str]) -> bool:
#     A = set(); B = set()
#     for nm in bodyA_names:
#         bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, nm)
#         if bid >= 0: A.add(int(bid))
#     for nm in bodyB_names:
#         bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, nm)
#         if bid >= 0: B.add(int(bid))
#     if not A or not B:
#         return False
#     for i in range(int(data.ncon)):
#         c = data.contact[i]
#         b1 = int(model.geom_bodyid[int(c.geom1)])
#         b2 = int(model.geom_bodyid[int(c.geom2)])
#         if ((b1 in A and b2 in B) or (b1 in B and b2 in A)) and float(c.dist) <= 0.0:
#             return True
#     return False

# def pose7_of_body(model, data, body_name: str) -> np.ndarray:
#     bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
#     p = data.xpos[bid].copy(); R = data.xmat[bid].reshape(3,3).copy()
#     return np.hstack([p, mat_to_quat_xyzw(R)])

# def apply_pure_force_on_body(model, data, body_name: str, f_world3: np.ndarray, steps: int, ramp_ratio: float):
#     bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
#     ft = np.zeros(6, dtype=np.float64); ft[:3] = np.asarray(f_world3, float).reshape(3,)
#     ramp = max(1, int(steps*float(np.clip(ramp_ratio,0.0,1.0))))
#     for i in range(ramp):
#         a = (i+1)/float(ramp); data.xfrc_applied[bid] = a*ft; mj.mj_step(model, data)
#     for _ in range(max(0,steps-ramp)):
#         data.xfrc_applied[bid] = ft; mj.mj_step(model, data)
#     data.xfrc_applied[bid] = np.zeros(6); mj.mj_forward(model, data)

# def delta_pose(a7, b7):
#     pa = np.asarray(a7[:3]); qa = np.asarray(a7[3:]); qa/= (np.linalg.norm(qa)+1e-12)
#     pb = np.asarray(b7[:3]); qb = np.asarray(b7[3:]); qb/= (np.linalg.norm(qb)+1e-12)
#     dp = float(np.linalg.norm(pb-pa))
#     qc = np.empty(4); mj.mju_mulQuat(qc, qb, np.array([qa[0],-qa[1],-qa[2],-qa[3]]))
#     da = 2.0*float(np.arccos(np.clip(qc[0], -1.0, 1.0)))
#     if da > np.pi: da = 2*np.pi - da
#     return dp, da

# def _free_qadr(model, name="root"):
#     jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
#     if jid < 0 or model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
#         raise RuntimeError(f"freejoint '{name}' 不存在或不是 freejoint")
#     return int(model.jnt_qposadr[jid])

# def _set_free_about_point(model, data, qadr, base_pos, base_quat, R_world, pivot_world):
#     q_rot = mat_to_quat_xyzw(R_world)
#     q_new = np.empty(4); mj.mju_mulQuat(q_new, q_rot, base_quat)
#     p0 = np.asarray(pivot_world, float)
#     p_new = p0 + R_world @ (base_pos - p0)
#     data.qpos[qadr:qadr+3]   = p_new
#     data.qpos[qadr+3:qadr+7] = q_new
#     mj.mj_forward(model, data)
#     return p_new, q_new

# def _set_body_pose7(model, data, freejoint_name: str, pose7: np.ndarray):
#     jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, freejoint_name)
#     adr = int(model.jnt_qposadr[jid])
#     data.qpos[adr:adr+7] = np.asarray(pose7, float).reshape(7,)
#     mj.mj_forward(model, data)

# def _print_state_debug(mjmod, model, data, joint_names5, site1, site2, qadr_free, extra=None):
#     try:
#         print("  [debug] qpos (first 10):", np.array2string(data.qpos[:10], precision=6))
#         for jn in joint_names5:
#             jid = mjmod.mj_name2id(model, mjmod.mjtObj.mjOBJ_JOINT, jn)
#             adr = int(model.jnt_qposadr[jid])
#             print(f"  [debug] joint {jn:>18s}: qpos={float(data.qpos[adr]):+.6f}")

#         root_p = data.qpos[qadr_free:qadr_free+3]
#         root_q = data.qpos[qadr_free+3:qadr_free+7]
#         print(f"  [debug] freejoint root pos={root_p}, quat={root_q}")

#         sid1 = mjmod.mj_name2id(model, mjmod.mjtObj.mjOBJ_SITE, site1)
#         sid2 = mjmod.mj_name2id(model, mjmod.mjtObj.mjOBJ_SITE, site2)
#         s1w = data.site_xpos[sid1]; s2w = data.site_xpos[sid2]
#         print(f"  [debug] site1={site1} world={s1w}")
#         print(f"  [debug] site2={site2} world={s2w}")

#         n = int(data.ncon)
#         print(f"  [debug] ncon={n} (dump first {min(n, CONTACT_DUMP_MAX)})")
#         for i in range(min(n, CONTACT_DUMP_MAX)):
#             c = data.contact[i]
#             g1 = int(c.geom1); g2 = int(c.geom2)
#             b1 = int(model.geom_bodyid[g1]); b2 = int(model.geom_bodyid[g2])
#             nm1 = mjmod.mj_id2name(model, mjmod.mjtObj.mjOBJ_BODY, b1) or ""
#             nm2 = mjmod.mj_id2name(model, mjmod.mjtObj.mjOBJ_BODY, b2) or ""
#             print(f"    - con#{i:02d}: bodies=({nm1},{nm2}) dist={float(c.dist):+.6e} "
#                   f"frame={np.array2string(c.frame, precision=3)}")
#         if extra:
#             print("  [debug] extra:", extra)
#     except Exception as e:
#         print("[debug] state dump error:", e)

# # ====================== 扫描库/CSV ======================

# def _load_csv_mapping(path, deg=False):
#     xs, ys = [], []
#     with open(path, "r", encoding="utf-8") as f:
#         for ln in f:
#             s = ln.strip()
#             if not s: continue
#             parts = [p.strip() for p in s.split(",")]
#             if len(parts) < 2: continue
#             try:
#                 x = float(parts[0]); y = float(parts[1])
#             except ValueError:
#                 continue
#             xs.append(x); ys.append(y)
#     if not xs:
#         raise ValueError(f"empty or invalid csv: {path}")
#     x = np.asarray(xs, dtype=np.float64)
#     y = np.asarray(ys, dtype=np.float64)
#     if deg:
#         x = np.deg2rad(x); y = np.deg2rad(y)
#     idx = np.argsort(x)
#     return x[idx], y[idx]

# def _interp_csv_saturated(x_src: np.ndarray, y_src: np.ndarray, xq: float) -> float:
#     lo, hi = float(np.min(x_src)), float(np.max(x_src))
#     xq = float(np.clip(xq, lo, hi))
#     return float(np.interp(xq, x_src, y_src))

# def _interp_distal(x_src, y_src, x_query):
#     return np.interp(x_query, x_src, y_src)

# def load_scan_db(path: str,
#                  th_csv: str = "", if_csv: str = "", csv_deg: bool = False):
#     arr = np.load(path, allow_pickle=True)

#     if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 6:
#         cols = arr.astype(np.float32)
#         qpos5 = cols[:, [1, 0, 2, 3, 4]]
#         dists = cols[:, 5].astype(np.float32)
#         return qpos5, dists

#     if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 4:
#         th_prox = arr[:, 0].astype(np.float64)
#         th_root = arr[:, 1].astype(np.float64)
#         if_prox = arr[:, 2].astype(np.float64)
#         dists   = arr[:, 3].astype(np.float32)

#         if not (os.path.isfile(th_csv) and os.path.isfile(if_csv)):
#             raise RuntimeError("旧格式 scan_db 需要提供 th_csv/if_csv 以补 distal。")

#         th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
#         if_x,  if_y = _load_csv_mapping(if_csv,  deg=csv_deg)

#         th_dist = _interp_distal(th_x, th_y, th_prox)
#         if_dist = _interp_distal(if_x,  if_y,  if_prox)

#         qpos5 = np.stack([th_root, th_prox, th_dist, if_prox, if_dist], axis=1).astype(np.float32)
#         return qpos5, dists

#     # 字典数组
#     try:
#         tmp_ctrl, tmp_dist = [], []
#         for row in np.atleast_1d(arr):
#             if isinstance(row, dict) and "ctrl" in row and "dist" in row:
#                 c = np.array(row["ctrl"], float).ravel()
#                 d = float(row["dist"])
#                 tmp_ctrl.append(c); tmp_dist.append(d)
#         if tmp_ctrl:
#             C = np.vstack(tmp_ctrl)
#             D = np.asarray(tmp_dist, np.float32)
#             if C.shape[1] >= 5:
#                 return C[:, :5].astype(np.float32), D
#             elif C.shape[1] >= 3:
#                 if not (os.path.isfile(th_csv) and os.path.isfile(if_csv)):
#                     raise RuntimeError("旧格式 scan_db(dict) 需要 th_csv/if_csv 以补 distal。")
#                 th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
#                 if_x,  if_y = _load_csv_mapping(if_csv,  deg=csv_deg)
#                 th_prox = C[:, 0]
#                 th_root = C[:, 1]
#                 if_prox = C[:, 2]
#                 th_dist = _interp_distal(th_x, th_y, th_prox)
#                 if_dist = _interp_distal(if_x,  if_y,  if_prox)
#                 qpos5 = np.stack([th_root, th_prox, th_dist, if_prox, if_dist], axis=1).astype(np.float32)
#                 return qpos5, D
#     except Exception:
#         pass

#     raise RuntimeError(f"无法解析 scan_db: {path}（期待 (N,6) 或 (N,4) 或 dict数组带 ctrl/dist）")

# def find_all_matching_ctrls(qpos5: np.ndarray, dists: np.ndarray, L: float,
#                             tol: float, tol_fb: float, max_ctrls: int = -1):
#     idx = np.where(np.abs(dists - L) < tol)[0]
#     used_tol = tol; fallback_used = False
#     if len(idx) == 0 and tol_fb > tol:
#         idx = np.where(np.abs(dists - L) < tol_fb)[0]
#         used_tol = tol_fb; fallback_used = True
#     if len(idx) == 0:
#         j = int(np.argmin(np.abs(dists - L)))
#         idx = np.array([j])
#         used_tol = float(abs(dists[j]-L))
#         print(f"[warn] 未找到满足容差的ctrl，使用最接近项 idx={j}, |dist-L|={used_tol:.3e}")
#     if max_ctrls > 0 and len(idx) > max_ctrls:
#         err = np.abs(dists[idx] - L)
#         idx = idx[np.argsort(err)][:max_ctrls]
#         print(f"[info] 限制每个 pair 只测试 {max_ctrls} 条（从 {len(err)} 候选中挑最接近的）")
#     return idx, used_tol, fallback_used

# # ====================== 场景类 ======================

# class CombinedScene:
#     def __init__(self, hand_xml: str, zero_gravity: bool):
#         os.environ.setdefault("MUJOCO_GL", "glfw")
#         self.mj = mj
#         if not hasattr(self.mj, "MjSpec"):
#             raise RuntimeError("需要 mujoco>=3.2")
#         self.spec = self.mj.MjSpec.from_file(hand_xml)
#         if zero_gravity:
#             self.spec.option.disableflags = self.mj.mjtDisableBit.mjDSBL_GRAVITY
#         try:
#             self.spec.add_key()
#         except Exception:
#             pass
#         self.model = None
#         self.data  = None

#     def compile(self):
#         try:
#             self.model = self.mj.MjModel.from_spec(self.spec)
#         except Exception:
#             self.model = self.spec.compile()
#         self.data  = self.mj.MjData(self.model)
#         try:
#             self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
#         except Exception:
#             self.mj.mj_resetData(self.model, self.data)
#         self.mj.mj_forward(self.model, self.data)

#     def add_lego_object(self, mesh_root_or_file: str,
#                         friction=(0.6,0.02,0.001), density: float=1000.0,
#                         body_name="lego", freejoint_name="lego_freejoint"):
#         files = _find_mesh_files(mesh_root_or_file)
#         files = [ _ensure_bin_stl(fp) for fp in files ]
#         body = self.spec.worldbody.add_body(name=body_name)
#         body.add_freejoint(name=freejoint_name)
#         for fp in files:
#             fp_abs = os.path.abspath(fp)
#             mname  = _safe_mesh_name(fp_abs)
#             self.spec.add_mesh(name=mname, file=fp_abs,
#                                scale=[FIXED_SCALE, FIXED_SCALE, FIXED_SCALE])
#             # visual
#             body.add_geom(name=f"{body_name}_visual_{mname}",
#                           type=self.mj.mjtGeom.mjGEOM_MESH, meshname=mname,
#                           density=0.0, contype=0, conaffinity=0, friction=friction)
#             # collision
#             body.add_geom(name=f"{body_name}_collision_{mname}",
#                           type=self.mj.mjtGeom.mjGEOM_SDF, meshname=mname,
#                           density=density, friction=friction, contype=1, conaffinity=1)
#         self.compile()

#     def set_lego_collision_enabled(self, enabled: bool, body_prefix="lego"):
#         gids = []
#         for gid in range(self.model.ngeom):
#             bid = int(self.model.geom_bodyid[gid])
#             bname = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, bid) or ""
#             if bname.startswith(body_prefix):
#                 gids.append(gid)
#         if not gids: return
#         gids = np.asarray(gids, dtype=np.int32)
#         if enabled:
#             self.model.geom_contype[gids]     = 1
#             self.model.geom_conaffinity[gids] = 1
#         else:
#             self.model.geom_contype[gids]     = 0
#             self.model.geom_conaffinity[gids] = 0
#         self.mj.mj_forward(self.model, self.data)
# # def eval_one_lego(cfg: Dict[str,Any], lego_name_raw: str, show=False, debug=False):
# #     cfg_dir = os.path.dirname(os.path.abspath(cfg["_config_path"]))
# #     xml_path       = _abs_join(cfg_dir, str(cfg["xml"]))
# #     obj_mesh_root  = _abs_join(cfg_dir, str(cfg["obj_mesh_root"]))
# #     antipodal_dir  = _abs_join(cfg_dir, str(cfg["antipodal_dir"]))
# #     scan_db_path   = _abs_join(cfg_dir, str(cfg["scan_db"]))
# #     out_root       = _abs_join(cfg_dir, str(cfg["out_dir"]))
# #     antipodal_prefix = str(cfg.get("antipodal_prefix","")).strip()

# #     # five joints
# #     th_root_name = str(cfg.get("th_root_joint", "th_root_link"))
# #     th_prox_name = str(cfg.get("th_prox_joint", "th_proximal_link"))
# #     th_dist_name = str(cfg.get("th_dist_joint", "th_distal_link"))
# #     if_prox_name = str(cfg.get("if_prox_joint", "if_proximal_link"))
# #     if_dist_name = str(cfg.get("if_dist_joint", "if_distal_link"))
# #     joint_names5 = [th_root_name, th_prox_name, th_dist_name, if_prox_name, if_dist_name]

# #     # CSV for tighten mapping
# #     th_csv = _abs_join(cfg_dir, str(cfg.get("th_csv",""))) if "th_csv" in cfg else ""
# #     if_csv = _abs_join(cfg_dir, str(cfg.get("if_csv",""))) if "if_csv" in cfg else ""
# #     csv_deg = bool(cfg.get("csv_deg", False))
# #     th_x = th_y = if_x = if_y = None
# #     if th_csv and os.path.isfile(th_csv): th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
# #     if if_csv and os.path.isfile(if_csv): if_x, if_y = _load_csv_mapping(if_csv,  deg=csv_deg)

# #     # tighten cfg
# #     tighten_joints = cfg.get("tighten_joints", [])
# #     if isinstance(tighten_joints, str):
# #         tighten_joints = [s.strip() for s in tighten_joints.split(",") if s.strip()]
# #     tighten_signs = cfg.get("tighten_signs", "")
# #     if isinstance(tighten_signs, str):
# #         parts = [p.strip() for p in tighten_signs.replace(";",",").split(",") if p.strip()]
# #         tighten_signs = [float(p) for p in (parts if parts else ["1"])]
# #     elif isinstance(tighten_signs, (list, tuple)):
# #         tighten_signs = [float(x) for x in tighten_signs]
# #     else:
# #         tighten_signs = [1.0]
# #     while len(tighten_signs) < len(tighten_joints):  # 对齐长度
# #         tighten_signs.append(tighten_signs[-1])
# #     tighten_total_cmd = float(cfg.get("tighten_cmd", 0.05))         # 总角度
# #     tighten_steps     = int(cfg.get("tighten_steps", 120))          # 慢速：默认更细
# #     contact_force_eps = float(cfg.get("contact_force_eps", 1e-4))   # 单指法向力阈值

# #     # 用于识别两根手指
# #     if_substrs = tuple(cfg.get("if_substrs", ["if_"]))
# #     th_substrs = tuple(cfg.get("th_substrs", ["th_", "tf_"]))

# #     lego_name = lego_name_raw.strip()
# #     name_stem = os.path.splitext(lego_name)[0]

# #     scene = CombinedScene(xml_path, zero_gravity=bool(cfg.get("zero_gravity", True)))

# #     mesh_target = obj_mesh_root if _is_mesh_file(obj_mesh_root) else os.path.join(obj_mesh_root, lego_name)
# #     density   = float(cfg.get("default_density", 1000.0))
# #     fric_cfg  = cfg.get("friction", [0.6,0.02,0.001])
# #     friction  = (float(fric_cfg[0]), float(fric_cfg[1]), float(fric_cfg[2] if len(fric_cfg)>=3 else 0.001))
# #     scene.add_lego_object(mesh_target, friction=friction, density=density,
# #                           body_name="lego", freejoint_name="lego_freejoint")
# #     model, data = scene.model, scene.data

# #     # === CHANGED === 预先解析 5 个关节的执行器 ID（用来写 ctrl）
# #     try:
# #         act_ids5 = [_act_id_of_joint(model, nm) for nm in joint_names5]
# #     except Exception as e:
# #         raise RuntimeError(f"解析执行器失败（请检查 5 个关节都挂了 position actuator）：{e}")
# #     # === CHANGED END ===

# #     qpos_init = data.qpos.copy()
# #     qvel_init = data.qvel.copy()
# #     ctrl_init = data.ctrl.copy()

# #     def reset_to_initial():
# #         data.qpos[:] = qpos_init
# #         data.qvel[:] = qvel_init
# #         data.ctrl[:] = ctrl_init
# #         mj.mj_forward(model, data)

# #     # viewer
# #     viewer = None
# #     if show:
# #         try:
# #             import mujoco.viewer as mjv
# #             viewer = mjv.launch_passive(model, data)
# #         except Exception as e:
# #             print("[warn] 打不开 viewer：", e)
# #             viewer = None

# #     site1 = resolve_site_name(model, cfg.get("site1","if_distal_site_ball"))
# #     site2 = resolve_site_name(model, cfg.get("site2","tf_distal_site_ball"))
# #     qadr_free = _free_qadr(model, cfg.get("freejoint","root"))

# #     distal_a_body = str(cfg.get("distal_a_body", "if_distal_link"))
# #     distal_b_body = str(cfg.get("distal_b_body", "th_distal_link"))

# #     r1 = float(cfg.get("r1",0.004)); r2 = float(cfg.get("r2",0.0055))
# #     tol = float(cfg.get("tol",1e-4)); tol_fb = float(cfg.get("tol_fallback",5e-4))
# #     trans_steps=int(cfg.get("trans_steps",10)); rot_steps=int(cfg.get("rot_steps",10))
# #     penetration_check_steps = int(cfg.get("penetration_check_steps",10))
# #     force_N=float(cfg.get("force_N",0.05)); force_steps=int(cfg.get("force_steps",300))
# #     force_ramp=float(cfg.get("force_ramp_ratio",0.2)); trans_thre=float(cfg.get("trans_thre",0.005))
# #     angle_thre=float(cfg.get("angle_thre",0.2))
# #     max_ctrls_per_pair = int(cfg.get("max_ctrls_per_pair", -1))  # -1 = 无限制

# #     pairs_path = os.path.join(antipodal_dir, f"{antipodal_prefix}{name_stem}.npy")
# #     if not os.path.isfile(pairs_path):
# #         raise FileNotFoundError(f"找不到 antipodal 文件：{pairs_path}")
# #     payload = np.load(pairs_path, allow_pickle=True)
# #     if isinstance(payload, dict) or (hasattr(payload, "item") and isinstance(payload.item(), dict)):
# #         pairs_b = np.asarray((payload if isinstance(payload, dict) else payload.item())["pairs"], np.float32)
# #     else:
# #         pairs_b = np.asarray(payload, np.float32)
# #     if pairs_b.ndim != 3 or pairs_b.shape[1:] != (2,3):
# #         raise RuntimeError(f"{pairs_path} 不是 (K,2,3) 的数组")
# #     if not bool(cfg.get("pairs_already_scaled", False)):
# #         pairs_b = pairs_b * float(FIXED_SCALE)

# #     K = pairs_b.shape[0]
# #     topk = int(cfg.get("topk",-1))
# #     eval_ids = list(range(min(K,topk) if (topk and topk>0) else K))
# #     qpos5_db, dists_db = load_scan_db(scan_db_path, th_csv=th_csv, if_csv=if_csv, csv_deg=csv_deg)

# #     out_dir = os.path.join(out_root, name_stem); os.makedirs(out_dir, exist_ok=True)

# #     lego_p, lego_R, _ = lego_pose_from_body(model, data, "lego")

# #     def draw_spheres(p1_b, p2_b, c1_b, c2_b):
# #         if viewer is None: return
# #         pL, RL, _ = lego_pose_from_body(model, data, "lego")
# #         p1w = world_from_local(p1_b, pL, RL)
# #         p2w = world_from_local(p2_b, pL, RL)
# #         c1w = world_from_local(c1_b, pL, RL)
# #         c2w = world_from_local(c2_b, pL, RL)
# #         s1w = get_site_pos(model, data, site1)
# #         s2w = get_site_pos(model, data, site2)
# #         scn = viewer.user_scn; scn.ngeom = 0
# #         def _sphere(pos, r, rgba):
# #             g = scn.geoms[scn.ngeom]
# #             mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.array([r,r,r],np.float32), pos,
# #                             np.eye(3,dtype=np.float32).reshape(-1), rgba)
# #             scn.ngeom += 1
# #         _sphere(p1w, 0.0025, (1,1,0,0.9))
# #         _sphere(p2w, 0.0025, (1,1,0,0.9))
# #         _sphere(c1w, r1, (0,1,0,0.6))
# #         _sphere(c2w, r2, (0,1,0,0.6))
# #         _sphere(s1w, max(1e-4,0.6*r1), (1,0,0,0.9))
# #         _sphere(s2w, max(1e-4,0.6*r2), (1,0,0,0.9))
# #         viewer.sync()

# #     # 读取关节值
# #     def _read_joint(model, data, jname: str) -> float:
# #         jid = _jid(model, jname)
# #         adr = int(model.jnt_qposadr[jid])
# #         return float(data.qpos[adr])

# #     def tighten_until_both_fingers_contact() -> Dict[str, Any]:

# #         log = {
# #             "used": bool(tighten_joints),
# #             "both_over_threshold": False,
# #             "established_at_step": -1,
# #             "per_step_force": [],  # [{step, fn_if, fn_th}]
# #             "eps": float(contact_force_eps),
# #         }
# #         if not tighten_joints:
# #             return log

# #         dp = (tighten_total_cmd / max(1, tighten_steps))
# #         for s in range(tighten_steps):
# #             # 逐个 prox 增量推进，并做 CSV 饱和插值到 distal
# #             for idx, prox in enumerate(tighten_joints):
# #                 sgn = float(tighten_signs[min(idx, len(tighten_signs)-1)])
# #                 curp = _read_joint(model, data, prox)
# #                 newp = _set_joint_qpos_clamped(model, data, prox, curp + sgn*dp)

# #                 if prox == if_prox_name and (if_x is not None and if_y is not None):
# #                     newd = _interp_csv_saturated(if_x, if_y, newp)
# #                     _set_joint_qpos_clamped(model, data, if_dist_name, newd)
# #                 elif prox == th_prox_name and (th_x is not None and th_y is not None):
# #                     newd = _interp_csv_saturated(th_x, th_y, newp)
# #                     _set_joint_qpos_clamped(model, data, th_dist_name, newd)

# #             mj.mj_step(model, data)
# #             fn_if, fn_th = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
# #             log["per_step_force"].append({"step": s+1, "fn_if": fn_if, "fn_th": fn_th})

# #             if (fn_if >= contact_force_eps) and (fn_th >= contact_force_eps):
# #                 log["both_over_threshold"] = True
# #                 log["established_at_step"] = s + 1
# #                 print(f"    [tighten] contact OK at step {s+1} (if={fn_if:.3e} N, th={fn_th:.3e} N, eps={contact_force_eps:.3e})")
# #                 break

# #         if not log["both_over_threshold"]:
# #             print("  [tighten] still not both-over-threshold "
# #                   f"(last if/th = {log['per_step_force'][-1] if log['per_step_force'] else {'na','na'}})")

# #         return log


# #     total_tests = 0
# #     total_success = 0
# #     pair_stats = []

# #     print(f"\n[{name_stem}] 开始评估 {len(eval_ids)} 个 antipodal pairs...")

# #     for pair_idx, k in enumerate(eval_ids, start=1):
# #         print(f"\n{'='*60}\n[Pair {pair_idx}/{len(eval_ids)}] pair_index={k}\n{'='*60}")

# #         p1_b = pairs_b[k,0].astype(np.float64)
# #         p2_b = pairs_b[k,1].astype(np.float64)
# #         seg  = float(np.linalg.norm(p2_b - p1_b))
# #         vhat = (p2_b - p1_b) / (seg + 1e-12)
# #         c1_b = p1_b - r1*vhat
# #         c2_b = p2_b + r2*vhat
# #         L    = seg + r1 + r2

# #         matching_indices, used_tol, fallback = find_all_matching_ctrls(
# #             qpos5_db, dists_db, L, tol, tol_fb, max_ctrls_per_pair
# #         )
# #         n_match = len(matching_indices)
# #         print(f"[Pair {k}] L={L:.6f}, 匹配到 {n_match} 条 (tol={used_tol:.2e}, fallback={fallback})")

# #         pair_success = 0; pair_fail = 0; pair_skip = 0

# #         for idx_local, db_idx in enumerate(matching_indices, start=1):
# #             total_tests += 1
# #             dist_db = float(dists_db[db_idx])
# #             qpos5   = np.asarray(qpos5_db[db_idx], np.float64).ravel()[:5]
# #             print(f"\n  [{idx_local}/{n_match}] db_idx={db_idx}, dist={dist_db:.6f}")

# #             reset_to_initial()
# #             scene.set_lego_collision_enabled(False, body_prefix="lego")

# #             # === CHANGED === 用执行器 ctrl 写目标位（夹到 ctrlrange），并收敛若干步
# #             try:
# #                 for aid, val in zip(act_ids5, qpos5):
# #                     data.ctrl[aid] = _ctrl_clamp(model, aid, float(val))
# #             except Exception as e:
# #                 print(f"  → SKIP: 设置 ctrl 失败: {e}")
# #                 pair_skip += 1
# #                 out = {
# #                     "status": "skip_set_qpos_error",  # 保持原字段名，避免你下游依赖改动
# #                     "lego": name_stem,
# #                     "pair_index": int(k),
# #                     "ctrl_db_index": int(db_idx),
# #                     "qpos5": qpos5.tolist(),
# #                     "qpos_names": joint_names5,
# #                     "error": str(e),
# #                 }
# #                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
# #                 continue

# #             # 给位置控制器一点时间把 qpos 拉到 ctrl 目标
# #             for _ in range(150):
# #                 mj.mj_step(model, data)
# #             # === CHANGED END ===

# #             # 平移/旋转到 c1/c2
# #             c1_w = world_from_local(c1_b, lego_p, lego_R)
# #             c2_w = world_from_local(c2_b, lego_p, lego_R)
# #             s1 = get_site_pos(model, data, site1)
# #             base_pos = data.qpos[qadr_free:qadr_free+3].copy()
# #             base_quat = data.qpos[qadr_free+3:qadr_free+7].copy()
# #             T = max(1, trans_steps if viewer else 1)
# #             delta = c1_w - s1
# #             for i in range(T):
# #                 a = (i+1)/T
# #                 data.qpos[qadr_free:qadr_free+3]   = base_pos + a*delta
# #                 data.qpos[qadr_free+3:qadr_free+7] = base_quat
# #                 mj.mj_forward(model, data)
# #                 if viewer: draw_spheres(p1_b,p2_b,c1_b,c2_b)
# #             base_pos = data.qpos[qadr_free:qadr_free+3].copy()

# #             s2a = get_site_pos(model, data, site2)
# #             v = s2a - c1_w; w = c2_w - c1_w
# #             vn = v/(np.linalg.norm(v)+1e-12); wn = w/(np.linalg.norm(w)+1e-12)
# #             axis = np.cross(vn, wn); l = np.linalg.norm(axis)
# #             dot = float(np.clip(np.dot(vn, wn), -1.0, 1.0))
# #             ang = float(np.arctan2(l, dot))
# #             if l < 1e-9:
# #                 tmp = np.array([1,0,0], np.float64)
# #                 if abs(np.dot(tmp, vn)) > 0.9: tmp = np.array([0,1,0], np.float64)
# #                 axis = np.cross(vn, tmp); axis /= (np.linalg.norm(axis)+1e-12)

# #             if viewer:
# #                 rotN = max(1, rot_steps)
# #                 dR = rodrigues(axis, ang/rotN)
# #                 for _ in range(rotN):
# #                     base_pos, base_quat = _set_free_about_point(
# #                         model, data, qadr_free, base_pos, base_quat, dR, c1_w)
# #                     draw_spheres(p1_b,p2_b,c1_b,c2_b)
# #             else:
# #                 R_all = rodrigues(axis, ang)
# #                 base_pos, base_quat = _set_free_about_point(
# #                     model, data, qadr_free, base_pos, base_quat, R_all, c1_w)

# #             # 对齐误差
# #             s1f = get_site_pos(model, data, site1)
# #             s2f = get_site_pos(model, data, site2)
# #             e1 = float(np.linalg.norm(s1f - c1_w))
# #             e2 = float(np.linalg.norm(s2f - c2_w))

# #             # distal 自碰（手内） -> 跳过
# #             if any_contact_between_bodies(model, data, [distal_a_body], [distal_b_body]):
# #                 print(f"  → SKIP: distal self collision")
# #                 pair_skip += 1
# #                 out = {
# #                     "status": "skip_distal_self_collision",
# #                     "lego": name_stem,
# #                     "pair_index": int(k),
# #                     "ctrl_db_index": int(db_idx),
# #                     "qpos5": qpos5.tolist(),
# #                     "qpos_names": joint_names5,
# #                     "ctrl_dist": dist_db,
# #                     "errors_align": {"e1": e1, "e2": e2},
# #                 }
# #                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
# #                 continue

# #             # 开启 LEGO 碰撞，推进几步检穿透
# #             scene.set_lego_collision_enabled(True, body_prefix="lego")
# #             for _ in range(penetration_check_steps):
# #                 mj.mj_step(model, data)
# #             min_dist_now = min_hand_lego_contact_dist(model, data, ("if_","th_","tf_"), "lego")

# #             if min_dist_now <= 0.0:
# #                 print(f"  → SKIP: hand-lego collision right after align (min_dist={min_dist_now:.6f})")
# #                 pair_skip += 1
# #                 out = {
# #                     "status": "skip_hand_lego_collision_after_align",
# #                     "lego": name_stem,
# #                     "pair_index": int(k),
# #                     "ctrl_db_index": int(db_idx),
# #                     "qpos5": qpos5.tolist(),
# #                     "qpos_names": joint_names5,
# #                     "ctrl_dist": dist_db,
# #                     "errors_align": {"e1": e1, "e2": e2},
# #                     "min_contact_dist": float(min_dist_now),
# #                 }
# #                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
# #                 continue

# #             # 若无足够接触 → 慢速 CSV 联动加紧（直到两指都过阈值）
# #             fn_if0, fn_th0 = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
# #             tighten_info = {"used": False, "both_over_threshold": False,
# #                             "established_at_step": -1, "per_step_force": [], "eps": contact_force_eps}
# #             if (fn_if0 < contact_force_eps) or (fn_th0 < contact_force_eps):
# #                 print(f"  [tighten] start (steps={tighten_steps}, cmd={tighten_total_cmd}, "
# #                       f"signs={tighten_signs}, eps={contact_force_eps:.3e})")
# #                 tighten_info = tighten_until_both_fingers_contact()

# #             # 若 tighten 后仍未达到“两指都过阈值” → 跳过稳定性
# #             fn_if1, fn_th1 = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
# #             if not ((fn_if1 >= contact_force_eps) and (fn_th1 >= contact_force_eps) and tighten_info.get("both_over_threshold", False)):
# #                 print(f"  → SKIP: no both-finger contact after tighten; if={fn_if1:.3e}, th={fn_th1:.3e}, eps={contact_force_eps:.3e}")
# #                 pair_skip += 1
# #                 out = {
# #                     "status": "skip_no_both_contact_after_tighten",
# #                     "lego": name_stem,
# #                     "pair_index": int(k),
# #                     "ctrl_db_index": int(db_idx),
# #                     "qpos5": qpos5.tolist(),
# #                     "qpos_names": joint_names5,
# #                     "ctrl_dist": dist_db,
# #                     "errors_align": {"e1": e1, "e2": e2},
# #                     "tighten": tighten_info
# #                 }
# #                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
# #                 continue

# #             # ========= 稳定性：六方向推力（只有两指都过阈值才进入） =========
# #             dirs = np.array([
# #                 [-1,0,0], [1,0,0],
# #                 [0,-1,0], [0,1,0],
# #                 [0,0,-1], [0,0,1]
# #             ], dtype=float)

# #             base_pose = pose7_of_body(model, data, "lego")
# #             all_ok   = True
# #             per_dir  = []
# #             worst = {"idx": -1, "dir": None, "dp": -1.0, "da": -1.0}
# #             fail_detail = None

# #             print(f"  [stability] align errors: e1={e1:.6f}, e2={e2:.6f} | "
# #                   f"thresholds: trans={trans_thre:.6f}, angle={angle_thre:.6f}")

# #             for di, d in enumerate(dirs):
# #                 _set_body_pose7(model, data, "lego_freejoint", base_pose)
# #                 pre = pose7_of_body(model, data, "lego")

# #                 apply_pure_force_on_body(model, data, "lego", d*force_N,
# #                                          steps=force_steps, ramp_ratio=force_ramp)
# #                 lat = pose7_of_body(model, data, "lego")

# #                 dp, da = delta_pose(pre, lat)
# #                 violated = []
# #                 if dp >= trans_thre: violated.append("translation")
# #                 if da >= angle_thre: violated.append("rotation")
# #                 ok = (len(violated) == 0)

# #                 if dp > worst["dp"] or (abs(dp-worst["dp"])<1e-12 and da>worst["da"]):
# #                     worst.update({"idx": di, "dir": d.tolist(), "dp": dp, "da": da})

# #                 per_dir.append({"dir": d.tolist(), "dp": dp, "da": da, "pass": bool(ok)})

# #                 if not ok:
# #                     blown = (dp >= 10.0*trans_thre) or (da >= 10.0*angle_thre)
# #                     classification = "blown_away" if blown else "drifted_over_threshold"
# #                     print(f"  → FAIL(stability) at dir#{di}={d.tolist()} | "
# #                           f"violated={violated} | dp={dp:.6f} (th={trans_thre:.6f}), "
# #                           f"da={da:.4f}rad (th={angle_thre:.4f}) | class={classification}")

# #                     fail_detail = {
# #                         "type": "stability_over_threshold",
# #                         "dir_index": int(di),
# #                         "dir": d.tolist(),
# #                         "dp": float(dp),
# #                         "da": float(da),
# #                         "thresholds": {"trans_thre": float(trans_thre), "angle_thre": float(angle_thre)},
# #                         "violated": violated,
# #                         "classification": classification,
# #                         "align_errors_before": {"e1": float(e1), "e2": float(e2)}
# #                     }
# #                     if debug:
# #                         _print_state_debug(mj, model, data, joint_names5, site1, site2, qadr_free, extra=fail_detail)
# #                     all_ok = False
# #                     break

# #             status = "success" if all_ok else "fail"
# #             if status == "success":
# #                 print(f"  → SUCCESS | worst_dir#{worst['idx']}={worst['dir']} "
# #                       f"| max_dp={worst['dp']:.6f}, max_da={worst['da']:.4f}rad")
# #                 pair_success += 1
# #                 total_success += 1
# #             else:
# #                 print(f"  → FAIL    | worst_dir#{worst['idx']}={worst['dir']} "
# #                       f"| max_dp={worst['dp']:.6f}, max_da={worst['da']:.4f}rad")
# #                 pair_fail += 1

# #             # 保存
# #             root7 = data.qpos[qadr_free:qadr_free+7].copy()
# #             out = {
# #                 "status": status,
# #                 "lego": name_stem,
# #                 "pair_index": int(k),
# #                 "ctrl_db_index": int(db_idx),
# #                 "qpos5": qpos5.tolist(),
# #                 "qpos_names": joint_names5,
# #                 "ctrl_dist": dist_db,
# #                 "ctrl_error": float(abs(dist_db - L)),
# #                 "r1": r1, "r2": r2, "green_distance": L,
# #                 "errors_align": {"e1": e1, "e2": e2},
# #                 "root_pose7": root7.tolist(),
# #                 "tighten": tighten_info,
# #                 "stability": {
# #                     "pass_all": bool(all_ok),
# #                     "force_N": force_N, "force_steps": force_steps, "force_ramp_ratio": force_ramp,
# #                     "per_direction": per_dir,
# #                     "worst_case": {"dir_index": int(worst["idx"]),
# #                                    "dir": worst["dir"],
# #                                    "max_dp": float(worst["dp"]),
# #                                    "max_da": float(worst["da"])}
# #                 },
# #                 "fail_reason": None if status=="success" else "stability_over_threshold",
# #                 "fail_detail": None if status=="success" else fail_detail
# #             }
# #             suffix = "succ" if status == "success" else "fail"
# #             np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.{suffix}.npy"),
# #                     out, allow_pickle=True)

# #         pair_total = pair_success + pair_fail + pair_skip
# #         pair_stats.append({
# #             "pair_index": int(k),
# #             "success": pair_success,
# #             "fail": pair_fail,
# #             "skip": pair_skip,
# #             "total": pair_total,
# #             "success_rate": (pair_success/pair_total) if pair_total>0 else 0.0
# #         })
# #         print(f"\n[Pair {k} Summary] success={pair_success}, fail={pair_fail}, skip={pair_skip}, total={pair_total}")

# #     overall_rate = total_success / total_tests if total_tests > 0 else 0.0
# #     summary = {
# #         "lego": name_stem,
# #         "total_pairs": len(eval_ids),
# #         "total_tests": total_tests,
# #         "total_success": total_success,
# #         "success_rate": overall_rate,
# #         "per_pair_stats": pair_stats,
# #     }
# #     with open(os.path.join(out_dir, f"{name_stem}.summary.json"), "w", encoding="utf-8") as f:
# #         json.dump(summary, f, ensure_ascii=False, indent=2)

# #     print(f"\n{'='*60}")
# #     print(f"[{name_stem}] 总体统计: pairs={len(eval_ids)} | tests={total_tests} | success={total_success} ({100*overall_rate:.2f}%)")
# #     print(f"{'='*60}")

# #     if viewer is not None:
# #         print("[info] 关闭窗口以继续下一个 LEGO…")
# #         try:
# #             while True: time.sleep(0.2)
# #         except KeyboardInterrupt:
# #             pass

# def eval_one_lego(cfg: Dict[str,Any], lego_name_raw: str, show=False, debug=False):
#     cfg_dir = os.path.dirname(os.path.abspath(cfg["_config_path"]))
#     xml_path       = _abs_join(cfg_dir, str(cfg["xml"]))
#     obj_mesh_root  = _abs_join(cfg_dir, str(cfg["obj_mesh_root"]))
#     antipodal_dir  = _abs_join(cfg_dir, str(cfg["antipodal_dir"]))
#     scan_db_path   = _abs_join(cfg_dir, str(cfg["scan_db"]))
#     out_root       = _abs_join(cfg_dir, str(cfg["out_dir"]))
#     antipodal_prefix = str(cfg.get("antipodal_prefix","")).strip()

#     th_root_name = str(cfg.get("th_root_joint", "th_root_link"))
#     th_prox_name = str(cfg.get("th_prox_joint", "th_proximal_link"))
#     th_dist_name = str(cfg.get("th_dist_joint", "th_distal_link"))
#     if_prox_name = str(cfg.get("if_prox_joint", "if_proximal_link"))
#     if_dist_name = str(cfg.get("if_dist_joint", "if_distal_link"))
#     joint_names5 = [th_root_name, th_prox_name, th_dist_name, if_prox_name, if_dist_name]

#     th_csv = _abs_join(cfg_dir, str(cfg.get("th_csv",""))) if "th_csv" in cfg else ""
#     if_csv = _abs_join(cfg_dir, str(cfg.get("if_csv",""))) if "if_csv" in cfg else ""
#     csv_deg = bool(cfg.get("csv_deg", False))
#     th_x = th_y = if_x = if_y = None
#     if th_csv and os.path.isfile(th_csv): th_x, th_y = _load_csv_mapping(th_csv, deg=csv_deg)
#     if if_csv and os.path.isfile(if_csv): if_x, if_y = _load_csv_mapping(if_csv,  deg=csv_deg)

#     tighten_joints = cfg.get("tighten_joints", [])
#     if isinstance(tighten_joints, str):
#         tighten_joints = [s.strip() for s in tighten_joints.split(",") if s.strip()]
#     tighten_signs = cfg.get("tighten_signs", "")
#     if isinstance(tighten_signs, str):
#         parts = [p.strip() for p in tighten_signs.replace(";",",").split(",") if p.strip()]
#         tighten_signs = [float(p) for p in (parts if parts else ["1"])]
#     elif isinstance(tighten_signs, (list, tuple)):
#         tighten_signs = [float(x) for x in tighten_signs]
#     else:
#         tighten_signs = [1.0]
#     while len(tighten_signs) < len(tighten_joints):
#         tighten_signs.append(tighten_signs[-1])
#     tighten_total_cmd = float(cfg.get("tighten_cmd", 0.05))
#     tighten_steps     = int(cfg.get("tighten_steps", 120))
#     contact_force_eps = float(cfg.get("contact_force_eps", 1e-4))

#     if_substrs = tuple(cfg.get("if_substrs", ["if_"]))
#     th_substrs = tuple(cfg.get("th_substrs", ["th_", "tf_"]))

#     lego_name = lego_name_raw.strip()
#     name_stem = os.path.splitext(lego_name)[0]

#     scene = CombinedScene(xml_path, zero_gravity=bool(cfg.get("zero_gravity", True)))

#     mesh_target = obj_mesh_root if _is_mesh_file(obj_mesh_root) else os.path.join(obj_mesh_root, lego_name)
#     density   = float(cfg.get("default_density", 1000.0))
#     fric_cfg  = cfg.get("friction", [0.6,0.02,0.001])
#     friction  = (float(fric_cfg[0]), float(fric_cfg[1]), float(fric_cfg[2] if len(fric_cfg)>=3 else 0.001))
#     scene.add_lego_object(mesh_target, friction=friction, density=density,
#                           body_name="lego", freejoint_name="lego_freejoint")
#     model, data = scene.model, scene.data

#     try:
#         act_ids5 = [_act_id_of_joint(model, nm) for nm in joint_names5]
#     except Exception as e:
#         raise RuntimeError(f"解析执行器失败（请检查 5 个关节都挂了 position actuator）：{e}")
#     joint_to_aid = {nm: aid for nm, aid in zip(joint_names5, act_ids5)}
#     aid_if_prox, aid_if_dist = joint_to_aid[if_prox_name], joint_to_aid[if_dist_name]

#     qpos_init = data.qpos.copy()
#     qvel_init = data.qvel.copy()
#     ctrl_init = data.ctrl.copy()

#     def reset_to_initial():
#         data.qpos[:] = qpos_init
#         data.qvel[:] = qvel_init
#         data.ctrl[:] = ctrl_init
#         mj.mj_forward(model, data)

#     viewer = None
#     if show:
#         try:
#             import mujoco.viewer as mjv
#             viewer = mjv.launch_passive(model, data)
#         except Exception as e:
#             print("[warn] 打不开 viewer：", e)
#             viewer = None

#     site1 = resolve_site_name(model, cfg.get("site1","if_distal_site_ball"))
#     site2 = resolve_site_name(model, cfg.get("site2","tf_distal_site_ball"))
#     qadr_free = _free_qadr(model, cfg.get("freejoint","root"))

#     distal_a_body = str(cfg.get("distal_a_body", "if_distal_link"))
#     distal_b_body = str(cfg.get("distal_b_body", "th_distal_link"))

#     r1 = float(cfg.get("r1",0.004)); r2 = float(cfg.get("r2",0.0055))
#     tol = float(cfg.get("tol",1e-4)); tol_fb = float(cfg.get("tol_fallback",5e-4))
#     trans_steps=int(cfg.get("trans_steps",10)); rot_steps=int(cfg.get("rot_steps",10))
#     penetration_check_steps = int(cfg.get("penetration_check_steps",10))
#     force_N=float(cfg.get("force_N",0.05)); force_steps=int(cfg.get("force_steps",300))
#     force_ramp=float(cfg.get("force_ramp_ratio",0.2)); trans_thre=float(cfg.get("trans_thre",0.005))
#     angle_thre=float(cfg.get("angle_thre",0.2))
#     max_ctrls_per_pair = int(cfg.get("max_ctrls_per_pair", -1))

#     approach_gap = float(cfg.get("approach_gap", 2e-4))

#     def draw_spheres(p1_b, p2_b, c1_b, c2_b):
#         if viewer is None: return
#         pL, RL, _ = lego_pose_from_body(model, data, "lego")
#         p1w = world_from_local(p1_b, pL, RL)
#         p2w = world_from_local(p2_b, pL, RL)
#         c1w = world_from_local(c1_b, pL, RL)
#         c2w = world_from_local(c2_b, pL, RL)
#         s1w = get_site_pos(model, data, site1)
#         s2w = get_site_pos(model, data, site2)
#         scn = viewer.user_scn; scn.ngeom = 0
#         def _sphere(pos, r, rgba):
#             g = scn.geoms[scn.ngeom]
#             mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.array([r,r,r],np.float32), pos,
#                             np.eye(3,dtype=np.float32).reshape(-1), rgba)
#             scn.ngeom += 1
#         _sphere(p1w, 0.0025, (1,1,0,0.9))
#         _sphere(p2w, 0.0025, (1,1,0,0.9))
#         _sphere(c1w, r1, (0,1,0,0.6))
#         _sphere(c2w, r2, (0,1,0,0.6))
#         _sphere(s1w, max(1e-4,0.6*r1), (1,0,0,0.9))
#         _sphere(s2w, max(1e-4,0.6*r2), (1,0,0,0.9))
#         viewer.sync()

#     def _read_joint(model, data, jname: str) -> float:
#         jid = _jid(model, jname)
#         adr = int(model.jnt_qposadr[jid])
#         return float(data.qpos[adr])

#     def tighten_if_until_contact_ctrl(eps: float,
#                                       max_steps: int,
#                                       total_cmd: float,
#                                       sign: float,
#                                       freeze_root=True) -> Dict[str, Any]:
#         log = {"used": True, "established_at_step": -1, "per_step_force": []}
#         if max_steps <= 0 or total_cmd <= 0.0:
#             return log
#         root_pose = data.qpos[qadr_free:qadr_free+7].copy()
#         dp = float(total_cmd) / float(max(1, max_steps))
#         for s in range(max_steps):
#             cur_target = float(data.ctrl[aid_if_prox])
#             new_target = _ctrl_clamp(model, aid_if_prox, cur_target + sign*dp)
#             data.ctrl[aid_if_prox] = new_target
#             if (if_x is not None) and (if_y is not None):
#                 new_dist = _interp_csv_saturated(if_x, if_y, new_target)
#                 data.ctrl[aid_if_dist] = _ctrl_clamp(model, aid_if_dist, new_dist)
#             if freeze_root:
#                 data.qpos[qadr_free:qadr_free+7] = root_pose
#                 mj.mj_forward(model, data)
#             mj.mj_step(model, data)
#             fn_if, fn_th = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)
#             log["per_step_force"].append({"step": s+1, "fn_if": fn_if, "fn_th": fn_th})
#             if fn_if >= eps:
#                 log["established_at_step"] = s + 1
#                 print(f"    [if-tighten-ctrl] contact at step {s+1} (if={fn_if:.3e} N, eps={eps:.3e})")
#                 break
#         return log

#     pairs_path = os.path.join(antipodal_dir, f"{antipodal_prefix}{name_stem}.npy")
#     if not os.path.isfile(pairs_path):
#         raise FileNotFoundError(f"找不到 antipodal 文件：{pairs_path}")
#     payload = np.load(pairs_path, allow_pickle=True)
#     if isinstance(payload, dict) or (hasattr(payload, "item") and isinstance(payload.item(), dict)):
#         pairs_b = np.asarray((payload if isinstance(payload, dict) else payload.item())["pairs"], np.float32)
#     else:
#         pairs_b = np.asarray(payload, np.float32)
#     if pairs_b.ndim != 3 or pairs_b.shape[1:] != (2,3):
#         raise RuntimeError(f"{pairs_path} 不是 (K,2,3) 的数组")
#     if not bool(cfg.get("pairs_already_scaled", False)):
#         pairs_b = pairs_b * float(FIXED_SCALE)

#     K = pairs_b.shape[0]
#     topk = int(cfg.get("topk",-1))
#     eval_ids = list(range(min(K,topk) if (topk and topk>0) else K))
#     qpos5_db, dists_db = load_scan_db(scan_db_path, th_csv=th_csv, if_csv=if_csv, csv_deg=csv_deg)

#     out_dir = os.path.join(out_root, name_stem); os.makedirs(out_dir, exist_ok=True)

#     lego_p, lego_R, _ = lego_pose_from_body(model, data, "lego")

#     total_tests = 0
#     total_success = 0
#     pair_stats = []

#     print(f"\n[{name_stem}] 开始评估 {len(eval_ids)} 个 antipodal pairs...")

#     for pair_idx, k in enumerate(eval_ids, start=1):
#         print(f"\n{'='*60}\n[Pair {pair_idx}/{len(eval_ids)}] pair_index={k}\n{'='*60}")

#         p1_b = pairs_b[k,0].astype(np.float64)
#         p2_b = pairs_b[k,1].astype(np.float64)
#         seg  = float(np.linalg.norm(p2_b - p1_b))
#         vhat = (p2_b - p1_b) / (seg + 1e-12)
#         c1_b = p1_b - r1*vhat
#         c2_b = p2_b + r2*vhat
#         L    = seg + r1 + r2

#         matching_indices, used_tol, fallback = find_all_matching_ctrls(
#             qpos5_db, dists_db, L, tol, tol_fb, max_ctrls_per_pair
#         )
#         n_match = len(matching_indices)
#         print(f"[Pair {k}] L={L:.6f}, 匹配到 {n_match} 条 (tol={used_tol:.2e}, fallback={fallback})")

#         pair_success = 0; pair_fail = 0; pair_skip = 0

#         for idx_local, db_idx in enumerate(matching_indices, start=1):
#             total_tests += 1
#             dist_db = float(dists_db[db_idx])
#             qpos5   = np.asarray(qpos5_db[db_idx], np.float64).ravel()[:5]
#             print(f"\n  [{idx_local}/{n_match}] db_idx={db_idx}, dist={dist_db:.6f}")

#             reset_to_initial()
#             scene.set_lego_collision_enabled(False, body_prefix="lego")

#             try:
#                 for aid, val in zip(act_ids5, qpos5):
#                     data.ctrl[aid] = _ctrl_clamp(model, aid, float(val))
#             except Exception as e:
#                 print(f"  → SKIP: 设置 ctrl 失败: {e}")
#                 pair_skip += 1
#                 out = {
#                     "status": "skip_set_qpos_error",
#                     "lego": name_stem,
#                     "pair_index": int(k),
#                     "ctrl_db_index": int(db_idx),
#                     "qpos5": qpos5.tolist(),
#                     "qpos_names": joint_names5,
#                     "error": str(e),
#                 }
#                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
#                 continue

#             for _ in range(150):
#                 mj.mj_step(model, data)

#             lego_p, lego_R, _ = lego_pose_from_body(model, data, "lego")
#             c1_w = world_from_local(c1_b, lego_p, lego_R)
#             c2_w = world_from_local(c2_b, lego_p, lego_R)

#             s2 = get_site_pos(model, data, site2)
#             base_pos = data.qpos[qadr_free:qadr_free+3].copy()
#             base_quat = data.qpos[qadr_free+3:qadr_free+7].copy()
#             T = max(1, trans_steps if viewer else 1)
#             delta2 = c2_w - s2
#             for i in range(T):
#                 a = (i+1)/T
#                 data.qpos[qadr_free:qadr_free+3]   = base_pos + a*delta2
#                 data.qpos[qadr_free+3:qadr_free+7] = base_quat
#                 mj.mj_forward(model, data)
#                 if viewer: draw_spheres(p1_b,p2_b,c1_b,c2_b)
#             base_pos = data.qpos[qadr_free:qadr_free+3].copy()

#             s1a = get_site_pos(model, data, site1)
#             v = s1a - c2_w
#             w = c1_w - c2_w
#             vn = v/(np.linalg.norm(v)+1e-12)
#             wn = w/(np.linalg.norm(w)+1e-12)
#             axis = np.cross(vn, wn)
#             l = np.linalg.norm(axis)
#             dot = float(np.clip(np.dot(vn, wn), -1.0, 1.0))
#             ang = float(np.arctan2(l, dot))
#             if l < 1e-9:
#                 tmp = np.array([1,0,0], np.float64)
#                 if abs(np.dot(tmp, vn)) > 0.9: tmp = np.array([0,1,0], np.float64)
#                 axis = np.cross(vn, tmp); axis /= (np.linalg.norm(axis)+1e-12)

#             if viewer:
#                 rotN = max(1, rot_steps)
#                 dR = rodrigues(axis, ang/rotN)
#                 for _ in range(rotN):
#                     base_pos, base_quat = _set_free_about_point(model, data, qadr_free, base_pos, base_quat, dR, c2_w)
#                     draw_spheres(p1_b,p2_b,c1_b,c2_b)
#             else:
#                 R_all = rodrigues(axis, ang)
#                 base_pos, base_quat = _set_free_about_point(model, data, qadr_free, base_pos, base_quat, R_all, c2_w)

#             s1f = get_site_pos(model, data, site1)
#             s2f = get_site_pos(model, data, site2)
#             e1 = float(np.linalg.norm(s1f - c1_w))
#             e2 = float(np.linalg.norm(s2f - c2_w))

#             if any_contact_between_bodies(model, data, [distal_a_body], [distal_b_body]):
#                 print(f"  → SKIP: distal self collision")
#                 pair_skip += 1
#                 out = {
#                     "status": "skip_distal_self_collision",
#                     "lego": name_stem,
#                     "pair_index": int(k),
#                     "ctrl_db_index": int(db_idx),
#                     "qpos5": qpos5.tolist(),
#                     "qpos_names": joint_names5,
#                     "ctrl_dist": dist_db,
#                     "errors_align": {"e1": e1, "e2": e2},
#                 }
#                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
#                 continue

#             scene.set_lego_collision_enabled(True, body_prefix="lego")
#             for _ in range(penetration_check_steps):
#                 mj.mj_step(model, data)
#             min_dist_now = min_hand_lego_contact_dist(model, data, ("if_","th_","tf_"), "lego")

#             if min_dist_now <= 0.0:
#                 print(f"  → SKIP: hand-lego collision right after align (min_dist={min_dist_now:.6f})")
#                 pair_skip += 1
#                 out = {
#                     "status": "skip_hand_lego_collision_after_align",
#                     "lego": name_stem,
#                     "pair_index": int(k),
#                     "ctrl_db_index": int(db_idx),
#                     "qpos5": qpos5.tolist(),
#                     "qpos_names": joint_names5,
#                     "ctrl_dist": dist_db,
#                     "errors_align": {"e1": e1, "e2": e2},
#                     "min_contact_dist": float(min_dist_now),
#                 }
#                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
#                 continue

#             fn_if0, fn_th0 = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)

#             if_prox_sign = -1.0
#             if tighten_joints:
#                 for idx, jn in enumerate(tighten_joints):
#                     if jn == if_prox_name:
#                         if_prox_sign = float(tighten_signs[min(idx, len(tighten_signs)-1)])
#                         break

#             if_steps = int(cfg.get("if_tighten_steps", tighten_steps))
#             if_cmd   = float(cfg.get("if_tighten_cmd",   tighten_total_cmd))
#             freeze_rt= bool(cfg.get("if_tighten_freeze_root", True))

#             if_tighten_log = {"used": False, "established_at_step": -1, "per_step_force": []}
#             if fn_if0 < contact_force_eps:
#                 if_tighten_log = tighten_if_until_contact_ctrl(
#                     eps=contact_force_eps,
#                     max_steps=if_steps,
#                     total_cmd=if_cmd,
#                     sign=if_prox_sign,
#                     freeze_root=freeze_rt
#                 )
#                 for _ in range(5):
#                     mj.mj_step(model, data)
#                 fn_if0, fn_th0 = sum_two_finger_normal_forces(model, data, if_substrs=if_substrs, th_substrs=th_substrs)

#             if fn_if0 < contact_force_eps:
#                 print(f"  → SKIP: IF still no contact after ctrl-tighten; if={fn_if0:.3e}, eps={contact_force_eps:.3e}")
#                 pair_skip += 1
#                 out = {
#                     "status": "skip_if_no_contact_after_ctrl_tighten",
#                     "lego": name_stem,
#                     "pair_index": int(k),
#                     "ctrl_db_index": int(db_idx),
#                     "qpos5": qpos5.tolist(),
#                     "qpos_names": joint_names5,
#                     "ctrl_dist": dist_db,
#                     "errors_align": {"e1": e1, "e2": e2},
#                     "if_tighten": if_tighten_log
#                 }
#                 np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.skip.npy"), out, allow_pickle=True)
#                 continue

#             dirs = np.array([
#                 [-1,0,0], [1,0,0],
#                 [0,-1,0], [0,1,0],
#                 [0,0,-1], [0,0,1]
#             ], dtype=float)

#             base_pose = pose7_of_body(model, data, "lego")
#             all_ok   = True
#             per_dir  = []
#             worst = {"idx": -1, "dir": None, "dp": -1.0, "da": -1.0}
#             fail_detail = None

#             print(f"  [stability] align errors: e1={e1:.6f}, e2={e2:.6f} | thresholds: trans={trans_thre:.6f}, angle={angle_thre:.6f}")

#             for di, d in enumerate(dirs):
#                 _set_body_pose7(model, data, "lego_freejoint", base_pose)
#                 pre = pose7_of_body(model, data, "lego")

#                 apply_pure_force_on_body(model, data, "lego", d*force_N,
#                                          steps=force_steps, ramp_ratio=force_ramp)
#                 lat = pose7_of_body(model, data, "lego")

#                 dp, da = delta_pose(pre, lat)
#                 violated = []
#                 if dp >= trans_thre: violated.append("translation")
#                 if da >= angle_thre: violated.append("rotation")
#                 ok = (len(violated) == 0)

#                 if dp > worst["dp"] or (abs(dp-worst["dp"])<1e-12 and da>worst["da"]):
#                     worst.update({"idx": di, "dir": d.tolist(), "dp": dp, "da": da})

#                 per_dir.append({"dir": d.tolist(), "dp": dp, "da": da, "pass": bool(ok)})

#                 if not ok:
#                     blown = (dp >= 10.0*trans_thre) or (da >= 10.0*angle_thre)
#                     classification = "blown_away" if blown else "drifted_over_threshold"
#                     print(f"  → FAIL(stability) at dir#{di}={d.tolist()} | violated={violated} | dp={dp:.6f} (th={trans_thre:.6f}), da={da:.4f}rad (th={angle_thre:.4f}) | class={classification}")

#                     fail_detail = {
#                         "type": "stability_over_threshold",
#                         "dir_index": int(di),
#                         "dir": d.tolist(),
#                         "dp": float(dp),
#                         "da": float(da),
#                         "thresholds": {"trans_thre": float(trans_thre), "angle_thre": float(angle_thre)},
#                         "violated": violated,
#                         "classification": classification,
#                         "align_errors_before": {"e1": float(e1), "e2": float(e2)}
#                     }
#                     if debug:
#                         _print_state_debug(mj, model, data, joint_names5, site1, site2, qadr_free, extra=fail_detail)
#                     all_ok = False
#                     break

#             status = "success" if all_ok else "fail"
#             if status == "success":
#                 print(f"  → SUCCESS | worst_dir#{worst['idx']}={worst['dir']} | max_dp={worst['dp']:.6f}, max_da={worst['da']:.4f}rad")
#                 pair_success += 1
#                 total_success += 1
#             else:
#                 print(f"  → FAIL    | worst_dir#{worst['idx']}={worst['dir']} | max_dp={worst['dp']:.6f}, max_da={worst['da']:.4f}rad")
#                 pair_fail += 1

#             root7 = data.qpos[qadr_free:qadr_free+7].copy()
#             out = {
#                 "status": status,
#                 "lego": name_stem,
#                 "pair_index": int(k),
#                 "ctrl_db_index": int(db_idx),
#                 "qpos5": qpos5.tolist(),
#                 "qpos_names": joint_names5,
#                 "ctrl_dist": dist_db,
#                 "ctrl_error": float(abs(dist_db - L)),
#                 "r1": r1, "r2": r2, "green_distance": L,
#                 "errors_align": {"e1": e1, "e2": e2},
#                 "root_pose7": root7.tolist(),
#                 "if_tighten": if_tighten_log,
#                 "stability": {
#                     "pass_all": bool(all_ok),
#                     "force_N": force_N, "force_steps": force_steps, "force_ramp_ratio": force_ramp,
#                     "per_direction": per_dir,
#                     "worst_case": {"dir_index": int(worst["idx"]),
#                                    "dir": worst["dir"],
#                                    "max_dp": float(worst["dp"]),
#                                    "max_da": float(worst["da"])}
#                 },
#                 "fail_reason": None if status=="success" else "stability_over_threshold",
#                 "fail_detail": None if status=="success" else fail_detail
#             }
#             suffix = "succ" if status == "success" else "fail"
#             np.save(os.path.join(out_dir, f"pair{k:06d}_ctrl{db_idx:06d}.{suffix}.npy"),
#                     out, allow_pickle=True)

#         pair_total = pair_success + pair_fail + pair_skip
#         pair_stats.append({
#             "pair_index": int(k),
#             "success": pair_success,
#             "fail": pair_fail,
#             "skip": pair_skip,
#             "total": pair_total,
#             "success_rate": (pair_success/pair_total) if pair_total>0 else 0.0
#         })
#         print(f"\n[Pair {k} Summary] success={pair_success}, fail={pair_fail}, skip={pair_skip}, total={pair_total}")

#     overall_rate = total_success / total_tests if total_tests > 0 else 0.0
#     summary = {
#         "lego": name_stem,
#         "total_pairs": len(eval_ids),
#         "total_tests": total_tests,
#         "total_success": total_success,
#         "success_rate": overall_rate,
#         "per_pair_stats": pair_stats,
#     }
#     with open(os.path.join(out_dir, f"{name_stem}.summary.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)

#     print(f"\n{'='*60}")
#     print(f"[{name_stem}] 总体统计: pairs={len(eval_ids)} | tests={total_tests} | success={total_success} ({100*overall_rate:.2f}%)")
#     print(f"{'='*60}")

#     if viewer is not None:
#         print("[info] 关闭窗口以继续下一个 LEGO…")
#         try:
#             while True: time.sleep(0.2)
#         except KeyboardInterrupt:
#             pass

# def main():
#     ap = argparse.ArgumentParser(
#         description="评估模式：scan结果设定5关节 + 慢速CSV联动加紧（两指力过阈值后做六向扰动）"
#     )
#     ap.add_argument("--config", default="configs/lego_eval5.yaml")
#     ap.add_argument("--show", action="store_true", help="每个 LEGO 启动独立 viewer")
#     ap.add_argument("--debug", action="store_true", help="失败时打印关节/位姿/site/接触等详细信息")
#     args = ap.parse_args()

#     cfg = load_cfg(args.config)
#     cfg["_config_path"] = os.path.abspath(args.config)

#     cfg_dir = os.path.dirname(os.path.abspath(args.config))
#     for k in ["xml","obj_mesh_root","antipodal_dir","scan_db","out_dir","lego_list_json","th_csv","if_csv"]:
#         if k in cfg:
#             cfg[k] = _abs_join(cfg_dir, str(cfg[k]))

#     if "lego_list_json" not in cfg:
#         print("配置中缺少 lego_list_json"); sys.exit(1)
#     lego_names = load_lego_list(cfg["lego_list_json"])
#     if not lego_names:
#         print("lego_list_json 为空"); sys.exit(1)

#     print("评估模式：scan结果设定 5 个关节 qpos + 慢速 CSV 联动加紧（两指力过阈值后才扰动）")
#     print(f"FIXED_SCALE = {FIXED_SCALE}")

#     for i, name in enumerate(lego_names, start=1):
#         print(f"\n{'#'*60}\n# [{i}/{len(lego_names)}] {name}\n{'#'*60}")
#         try:
#             eval_one_lego(cfg, name, show=args.show, debug=args.debug)
#         except Exception as e:
#             print(f"\n[ERROR] {name} 评估失败: {e}")
#             import traceback; traceback.print_exc()
#             continue

#     print("\n[DONE] 全部 LEGO 测试完成。")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
