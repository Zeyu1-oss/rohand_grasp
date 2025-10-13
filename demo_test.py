#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, math, argparse, numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer

# ====================== 基础工具 ======================
def name2id(model, objtype, name):
    return mj.mj_name2id(model, objtype, name)

def quat_to_mat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def yaw_from_quat(q):
    R = quat_to_mat(q)
    return math.atan2(R[1,0], R[0,0])

def print_root_pose(model, data, base_body, lego_geom=None, tag=""):
    """打印 root 在世界系与 LEGO 系的位姿（位置 + yaw）。"""
    pose = get_body_freejoint_pose(model, data, base_body)
    if pose is None:
        print(f"{tag} [ROOT] no freejoint found for {base_body}")
        return
    pos_w, quat_w = pose
    yaw_w = yaw_from_quat(quat_w) * 180.0 / math.pi
    msg = f"{tag} [ROOT@world] p=[{pos_w[0]:+.4f} {pos_w[1]:+.4f} {pos_w[2]:+.4f}] m, yaw={yaw_w:+.2f} deg"
    if lego_geom is not None:
        try:
            Rg, tg = geom_RT(model, data, lego_geom)          # world_T_lego
            p_L = Rg.T @ (pos_w - tg)
            Rb_w = quat_to_mat(quat_w)
            Rb_L = Rg.T @ Rb_w
            yaw_L = math.atan2(Rb_L[1,0], Rb_L[0,0]) * 180.0 / math.pi
            msg += f" | [ROOT@lego] pL=[{p_L[0]:+.4f} {p_L[1]:+.4f} {p_L[2]:+.4f}] m, yawL={yaw_L:+.2f} deg"
        except:
            pass
    print(msg)

def ensure_upright_and_above_floor(model, data, base_body, z_min, yaw_only=True):
    """把 root 约束成：纯yaw（无滚俯仰）且 z 不低于 z_min。"""
    pose = get_body_freejoint_pose(model, data, base_body)
    if pose is None: return False
    pos, quat = pose
    if yaw_only:
        yaw = yaw_from_quat(quat)
        quat = quat_from_axis_angle(np.array([0,0,1.0]), yaw)
    pos[2] = max(float(pos[2]), float(z_min))
    set_body_freejoint_pose(model, data, base_body, pos, quat)
    return True

def mj_step_n(model, data, n, v=None):
    for _ in range(int(n)):
        mj.mj_step(model, data)
        if v is not None: v.sync()

def site_xpos(model, data, site):
    sid = name2id(model, mj.mjtObj.mjOBJ_SITE, site)
    return data.site_xpos[sid].copy()

def body_RT(model, data, body):
    bid = name2id(model, mj.mjtObj.mjOBJ_BODY, body)
    R = data.xmat[bid].reshape(3,3).copy()
    t = data.xpos[bid].copy()
    return R, t

def geom_RT(model, data, geom):
    gid = name2id(model, mj.mjtObj.mjOBJ_GEOM, geom)
    R = data.geom_xmat[gid].reshape(3,3).copy()
    t = data.geom_xpos[gid].copy()
    return R, t

def actuator_ctrl(model, data, actuator, u):
    aid = name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator)
    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(u, lo, hi))

def set_hinge_qpos(model, data, joint, q):
    jid  = name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr] = float(q)
    mj.mj_forward(model, data)

def find_freejoint_of_body(model, body):
    bid = name2id(model, mj.mjtObj.mjOBJ_BODY, body)
    jadr, jnum = model.body_jntadr[bid], model.body_jntnum[bid]
    for k in range(jnum):
        jid = jadr + k
        if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
            return jid, model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    return None, None, None

def get_body_freejoint_pose(model, data, body):
    jid, qadr, _ = find_freejoint_of_body(model, body)
    if qadr is None: return None
    pos = data.qpos[qadr:qadr+3].copy()
    quat= data.qpos[qadr+3:qadr+7].copy()
    return pos, quat

def set_body_freejoint_pose(model, data, body, pos, quat):
    jid, qadr, _ = find_freejoint_of_body(model, body)
    if qadr is None: return False
    pos = np.asarray(pos, float); quat = np.asarray(quat, float)
    quat = quat / (np.linalg.norm(quat) + 1e-12)  # 归一化四元数
    data.qpos[qadr:qadr+7] = np.concatenate([pos, quat])
    mj.mj_forward(model, data)
    return True

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float); n = np.linalg.norm(axis)
    if n < 1e-12: return np.array([1,0,0,0], float)
    axis = axis / n; s = math.sin(angle/2.0)
    return np.array([math.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], float)

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v); return v/(n+eps)

def angle_between(u, v):
    u = normalize(u); v = normalize(v)
    c = float(np.clip(np.dot(u,v), -1.0, 1.0))
    return math.acos(c)

def wait_body_settled(model, data, body, lin_th=0.02, ang_th=0.05, max_steps=150, v=None):
    """等待物体静止，避免无限等待"""
    bid = name2id(model, mj.mjtObj.mjOBJ_BODY, body)
    for _ in range(max_steps):
        v6 = data.cvel[bid]
        if np.linalg.norm(v6[:3]) < ang_th and np.linalg.norm(v6[3:]) < lin_th:
            return True
        mj.mj_step(model, data)
        if v is not None: v.sync()
    print("[WARN] LEGO 未完全静止，但到达最大步数，继续流程")
    return False

def mj_step_n_pin_root(model, data, base_body, n, v=None):
    jid, qadr, dofadr = find_freejoint_of_body(model, base_body)
    if qadr is None:
        mj_step_n(model, data, n, v=v); return
    pos_t = data.qpos[qadr:qadr+3].copy()
    quat_t= data.qpos[qadr+3:qadr+7].copy()
    for _ in range(int(n)):
        data.qpos[qadr:qadr+7] = np.concatenate([pos_t, quat_t])
        data.qvel[dofadr:dofadr+6] = 0.0
        mj.mj_forward(model, data)
        mj.mj_step(model, data)
        if v is not None: v.sync()

# ====================== antipodal 工具 ======================
def _reshape_pairs(raw):
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[-2:] == (2,3):
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 6:
        return arr.reshape(-1,2,3)
    raise ValueError(f"Unsupported pairs shape: {arr.shape}")

def _find_floor_z(model, data, default=-0.02):
    """找地面高度；找不到就用默认 -0.02m。"""
    try:
        zs = []
        for gid in range(model.ngeom):
            gtype = model.geom_type[gid]
            nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid) or ""
            if gtype == mj.mjtGeom.mjGEOM_PLANE or ("floor" in nm or "ground" in nm):
                zs.append(float(data.geom_xpos[gid, 2]))
        if zs:
            return min(zs)
    except Exception:
        pass
    return float(default)

def pick_pair_world(npy_path, model, data, lego_geom,
                    minD_m=0.004,
                    horiz_frac=0.35,
                    keep_above_mm=0.5,
                    loose_horiz_frac=0.60,
                    lift_if_below_mm=1.0):
    """
    一定返回一对点:
      Pass A:  D>=minD 且 两点都高于 floor+keep_above_mm 且 |dz|/D<=horiz_frac
      Pass B:  放宽为 |dz|/D<=loose_horiz_frac
      Pass C:  仍无 → 取 D 最大的一对；若有任一点低于地面，则整体沿世界z抬高 lift_if_below_mm
    返回: (p1_world, p2_world, n_world_or_None, D)
    """
    # 读取
    raw = np.load(npy_path, allow_pickle=True)
    if isinstance(raw, dict) or hasattr(raw, "item"):
        payload   = raw.item()
        pairs     = _reshape_pairs(payload["pairs"])
        normals   = payload.get("normals", None)
        if normals is not None: normals = _reshape_pairs(normals)
        scale     = float(payload.get("scale_applied", 1.0))
        frame_tag = str(payload.get("coord_frame", "lego"))
    else:
        pairs     = _reshape_pairs(raw)
        normals   = None
        scale     = 1.0
        frame_tag = "lego"

    if pairs.size == 0:
        # 返回一个“虚”的很短对，避免崩溃（理论上不会发生）
        z0 = _find_floor_z(model, data, default=-0.02) + 0.010
        p1 = np.array([0.0, 0.0, z0])
        p2 = np.array([0.004, 0.0, z0])
        print("[ANTIPODAL:FALLBACK-EMPTY] fabricated tiny pair above floor")
        return p1, p2, None, np.linalg.norm(p2-p1)

    # world_T_lego
    Rg, tg    = geom_RT(model, data, lego_geom)
    floor_z   = _find_floor_z(model, data, default=-0.02)
    clr       = keep_above_mm * 1e-3
    lift_dz   = lift_if_below_mm * 1e-3

    # LEGO→世界
    pL = pairs * scale
    pW = (Rg @ pL.reshape(-1,3).T).T.reshape(pL.shape) + tg  # (K,2,3)

    K = pW.shape[0]
    Ds = np.linalg.norm(pW[:,1,:] - pW[:,0,:], axis=1)
    dz = np.abs(pW[:,1,2] - pW[:,0,2])
    dz_over_D = dz / (Ds + 1e-12)

    # 统计
    both_above = (pW[:,0,2] > floor_z + clr) & (pW[:,1,2] > floor_z + clr)
    dist_ok    = (Ds >= float(minD_m))
    A_mask     = dist_ok & both_above & (dz_over_D <= float(horiz_frac))
    B_mask     = dist_ok & both_above & (dz_over_D <= float(loose_horiz_frac))
    any_above  = dist_ok & ((pW[:,0,2] > floor_z + clr) | (pW[:,1,2] > floor_z + clr))

    print(f"[ANTIPODAL:STATS] K={K} | floor_z={floor_z:+.4f} "
          f"| both_above={int(both_above.sum())} | dist_ok={int(dist_ok.sum())} "
          f"| A(horiz)={int(A_mask.sum())} | B(loose)={int(B_mask.sum())} | any_above={int(any_above.sum())}")

    def _select_from(mask, tag):
        idxs = np.where(mask)[0]
        if idxs.size == 0: return None
        # 取前 50% 最大 D 范围里随机一个，偏向更大 D
        order = idxs[np.argsort(-Ds[idxs])]
        top   = max(1, order.size//2)
        k     = order[np.random.randint(0, top)]
        p     = pW[k].astype(np.float64)
        nW    = None
        if normals is not None:
            nl = normals[k].astype(np.float64)
            nW = (Rg @ nl.T).T
            nW = nW / (np.linalg.norm(nW, axis=1, keepdims=True) + 1e-12)
        D     = float(Ds[k])
        print(f"[ANTIPODAL:{tag}] D={D*1e3:.2f}mm | "
              f"p1.z={p[0,2]:+.4f} p2.z={p[1,2]:+.4f} (floor={floor_z:+.4f}) | "
              f"|dz|/D={float(dz_over_D[k]):.2f}")
        return p[0], p[1], nW, D

    # Pass A（最严格）
    ret = _select_from(A_mask, "H-PRIME")
    if ret is not None:
        return ret

    # Pass B（放宽横向）
    ret = _select_from(B_mask, "H-LOOSE")
    if ret is not None:
        return ret

    # Pass C：没有任何“两点都在地上”的候选了——选 D 最大的安全候选，然后“整体抬高”
    if np.any(dist_ok):
        k = int(np.argmax(Ds * dist_ok))   # 最大 D 且满足最小间距
        p = pW[k].astype(np.float64)
        D = float(Ds[k])
        need_lift = max(0.0, (floor_z + clr) - min(p[0,2], p[1,2]) + lift_dz)
        if need_lift > 0.0:
            p[:,2] += need_lift
            print(f"[ANTIPODAL:LIFT] raised pair by {need_lift*1e3:.2f}mm "
                  f"to stay above floor (p1.z={p[0,2]:+.4f}, p2.z={p[1,2]:+.4f})")
        nW = None
        if normals is not None:
            nl = normals[k].astype(np.float64)
            nW = (Rg @ nl.T).T
            nW = nW / (np.linalg.norm(nW, axis=1, keepdims=True) + 1e-12)
        print(f"[ANTIPODAL:MAX-D] D={D*1e3:.2f}mm | "
              f"p1.z={p[0,2]:+.4f} p2.z={p[1,2]:+.4f} (floor={floor_z:+.4f}) | "
              f"|dz|/D={float(dz_over_D[k]):.2f}")
        return p[0], p[1], nW, D

    # 极端兜底：放弃 minD_m，只拿全体里最大的 D，并抬高
    k = int(np.argmax(Ds))
    p = pW[k].astype(np.float64)
    D = float(Ds[k])
    need_lift = max(0.0, (floor_z + clr) - min(p[0,2], p[1,2]) + lift_dz)
    if need_lift > 0.0:
        p[:,2] += need_lift
        print(f"[ANTIPODAL:LIFT-ZERO-DISTOK] raised by {need_lift*1e3:.2f}mm")
    print("[WARN] Using largest-D pair without distance filter (extreme fallback).")
    return p[0], p[1], None, D

def query_sdf_dist_pos_normal(model, data, lego_geom_name, distal_geom_name):
    try:
        lego_gid   = name2id(model, mj.mjtObj.mjOBJ_GEOM, lego_geom_name)
        distal_gid = name2id(model, mj.mjtObj.mjOBJ_GEOM, distal_geom_name)
    except:
        return None
        
    best = None  # (dist, pos, normal_world)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        if {g1, g2} != {lego_gid, distal_gid}:
            continue
        n = np.array([con.frame[0], con.frame[3], con.frame[6]], dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)
        d = float(con.dist)  # 负值=穿透；小正值=接近
        p = np.array([con.pos[0], con.pos[1], con.pos[2]], dtype=float)
        n_world = n if g2==lego_gid else -n     # 指向 LEGO 外
        if (best is None) or (d < best[0]): best = (d, p, n_world)
    return best

def get_contact_pair(model, data, lego_geom, if_geom, th_geom):
    ci = query_sdf_dist_pos_normal(model, data, lego_geom, if_geom)
    ct = query_sdf_dist_pos_normal(model, data, lego_geom, th_geom)
    if (ci is None) or (ct is None):
        return None
    d_if, p_if, n_if = ci
    d_th, p_th, n_th = ct
    d_dir = p_th - p_if
    nd = np.linalg.norm(d_dir)
    if nd < 1e-12:
        return None
    d_hat = d_dir / nd
    return (d_if, p_if, n_if), (d_th, p_th, n_th), d_hat

def friction_cost_from_contacts(pair, cos_alpha, dist_tol, lam=10.0):
    (d_if, p_if, n_if), (d_th, p_th, n_th), d_hat = pair
    c_if = max(0.0, cos_alpha - float(np.dot(n_if,  +d_hat)))
    c_th = max(0.0, cos_alpha - float(np.dot(n_th,  -d_hat)))
    c_gap = max(0.0, dist_tol - min(d_if, d_th))
    J = c_if*c_if + c_th*c_th + lam*(c_gap*c_gap)
    return J, (c_if, c_th, c_gap)

# ====================== root 的 SE(3) 更新与梯度（用于终端优化） ======================
def apply_delta_pose_se3(model, data, base_body, delta):
    # delta: [dtx, dty, dtz, dwx, dwy, dwz]
    pose = get_body_freejoint_pose(model, data, base_body)
    if pose is None:
        return False
    pos, quat = pose
    dtrans = np.array(delta[:3], float)
    domega = np.array(delta[3:], float)
    theta  = float(np.linalg.norm(domega))
    if theta > 1e-12:
        axis = domega / theta
        dquat = quat_from_axis_angle(axis, theta)
    else:
        dquat = np.array([1.0, 0.0, 0.0, 0.0], float)
    pos2  = pos + dtrans
    quat2 = quat_mul(dquat, quat)
    quat2 = quat2 / (np.linalg.norm(quat2) + 1e-12)
    set_body_freejoint_pose(model, data, base_body, pos2, quat2)
    return True

def numeric_grad_cost_root(model, data, base_body, lego_geom, if_geom, th_geom,
                           cos_alpha, dist_tol,
                           h_trans=1e-4, h_rot=1e-3, pin_steps=2):
    pose0 = get_body_freejoint_pose(model, data, base_body)
    if pose0 is None: return None, None, None
    pair0 = get_contact_pair(model, data, lego_geom, if_geom, th_geom)
    if pair0 is None: return None, None, None
    J0, parts0 = friction_cost_from_contacts(pair0, cos_alpha, dist_tol)

    g = np.zeros(6, dtype=float)
    eye = np.eye(6)
    for i in range(6):
        step = h_trans if i < 3 else h_rot
        # +h
        set_body_freejoint_pose(model, data, base_body, *pose0)
        mj_step_n_pin_root(model, data, base_body, 1)
        apply_delta_pose_se3(model, data, base_body, eye[i]*step)
        mj_step_n_pin_root(model, data, base_body, pin_steps)
        pair_p = get_contact_pair(model, data, lego_geom, if_geom, th_geom)
        Jp = friction_cost_from_contacts(pair_p, cos_alpha, dist_tol)[0] if pair_p else J0 + 1e3
        # -h
        set_body_freejoint_pose(model, data, base_body, *pose0)
        mj_step_n_pin_root(model, data, base_body, 1)
        apply_delta_pose_se3(model, data, base_body, -eye[i]*step)
        mj_step_n_pin_root(model, data, base_body, pin_steps)
        pair_m = get_contact_pair(model, data, lego_geom, if_geom, th_geom)
        Jm = friction_cost_from_contacts(pair_m, cos_alpha, dist_tol)[0] if pair_m else J0 + 1e3
        # 还原
        set_body_freejoint_pose(model, data, base_body, *pose0)
        mj_step_n_pin_root(model, data, base_body, 1)
        g[i] = (Jp - Jm) / (2.0*step)
    return g, J0, parts0

# ====================== motor：慢而稳的目标夹距闭环 ======================
def motor_track_gap(model, data, siteA, siteB, motor, target_gap,
                    iters=20, k=0.6, v=None, base_body=None,
                    k_far=3.0, k_near=1.2, far_eps=8e-3, near_eps=3e-3, gap_tol=1e-4,
                    z_guard_min=None):
    """PID-样式微调（很保守）"""
    try:
        aid = name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, motor)
        lo, hi = model.actuator_ctrlrange[aid]
    except:
        print(f"[WARN] Cannot find actuator {motor}")
        return
    for i in range(int(iters)):
        try:
            ga = site_xpos(model, data, siteA); gb = site_xpos(model, data, siteB)
            gap = float(np.linalg.norm(ga - gb))
            e   = gap - target_gap
            if abs(e) < gap_tol: 
                break
            if abs(e) > far_eps:   kk = k_far
            elif abs(e) > near_eps:kk = k_near  
            else:                   kk = k
            data.ctrl[aid] = float(np.clip(data.ctrl[aid] - kk*e, lo, hi))
            if base_body:
                mj_step_n_pin_root(model, data, base_body, 2, v=v)
                if z_guard_min is not None:
                    ensure_upright_and_above_floor(model, data, base_body, z_guard_min, yaw_only=True)
            else:
                mj_step_n(model, data, 2, v=v)
        except Exception as e:
            print(f"[WARN] Motor control error at iter {i}: {e}")
            break

# ====================== 温柔下降（SDF门控） ======================
def smart_pregrasp_descent(model, data, base_body, siteA, siteB,
                           p1, p2, outside_extra, gap_margin,
                           dz=2e-4, steps=400, max_xy_step=4e-4, v=None, motor=None,
                           z_guard_min=0.015, lego_geom=None, if_geom=None, th_geom=None):
    """
    温柔下降：保持 site 在外侧目标的“投影”上方缓慢下压；
    1) 永远夹在 z_guard_min 之上；
    2) SDF 距离越小，dz 越小；小于 0 立即回退；
    3) 只做很小的 XY 纠偏和少量 yaw 修正，避免碰砖/地。
    """
    d = normalize(p2 - p1)
    target_A = p1 - outside_extra * d
    target_B = p2 + outside_extra * d
    need_gap = np.linalg.norm(p2 - p1) + 2*outside_extra + gap_margin

    if motor is not None:
        motor_track_gap(model, data, siteA, siteB, motor, need_gap,
                        iters=50, k=0.6, v=v, base_body=base_body, z_guard_min=z_guard_min)

    safety_band = 1.5e-3
    min_dz = 8e-5
    backoff_z = 2e-3

    for it in range(int(steps)):
        a = site_xpos(model, data, siteA)
        b = site_xpos(model, data, siteB)

        # XY 纠偏：把 (A,B) 的中点往 (target_A,target_B) 的中点靠
        mid    = 0.5*(a + b)
        mid_t  = 0.5*(target_A + target_B)
        err_xy = mid - mid_t
        err_xy[2] = 0.0
        if np.linalg.norm(err_xy) > max_xy_step:
            err_xy *= max_xy_step / (np.linalg.norm(err_xy) + 1e-12)

        # yaw 微调：当前 (b-a) 与目标方向 d 的夹角
        cur_dir = normalize(b - a)
        dir_for_yaw = d if np.dot(cur_dir, d) >= 0.0 else -d
        yaw_err = math.atan2(np.cross(cur_dir, dir_for_yaw)[2],
                             np.dot(cur_dir, dir_for_yaw))
        dquat = quat_from_axis_angle(np.array([0,0,1.0]),
                                     float(np.clip(-0.5*yaw_err, -8e-4, 8e-4)))

        # SDF 距离自适应 dz
        dz_eff = dz
        d_min = None
        if lego_geom and if_geom and th_geom:
            ci = query_sdf_dist_pos_normal(model, data, lego_geom, if_geom)
            ct = query_sdf_dist_pos_normal(model, data, lego_geom, th_geom)
            dd = []
            if ci: dd.append(ci[0])
            if ct: dd.append(ct[0])
            if dd:
                d_min = min(dd)
                if d_min > 0.0:
                    dz_eff = min(dz_eff, max(d_min - safety_band, min_dz))
                else:
                    # 穿透就回退并减速
                    pose = get_body_freejoint_pose(model, data, base_body)
                    if pose is not None:
                        pos, quat = pose
                        set_body_freejoint_pose(model, data, base_body, pos + np.array([0,0,backoff_z]), quat)
                        mj_step_n_pin_root(model, data, base_body, 8, v=v)
                        ensure_upright_and_above_floor(model, data, base_body, z_guard_min, yaw_only=True)
                    dz_eff = min_dz

        # 应用位姿
        pose = get_body_freejoint_pose(model, data, base_body)
        if pose is None: break
        pos, quat = pose
        pos_new   = pos - err_xy + np.array([0,0,-dz_eff])

        # 硬防线：不低于 z_guard_min
        if pos_new[2] < z_guard_min:
            pos_new[2] = z_guard_min

        quat_new = quat_mul(dquat, quat)
        set_body_freejoint_pose(model, data, base_body, pos_new, quat_new)
        ensure_upright_and_above_floor(model, data, base_body, z_guard_min, yaw_only=True)
        mj_step_n_pin_root(model, data, base_body, 1, v=v)

        if it % 20 == 0:
            msg = f"[DESC] it={it:03d} | |err_xy|={np.linalg.norm(err_xy)*1e3:.2f}mm | yaw_err={yaw_err*180/np.pi:.2f}deg | dz={dz_eff*1e3:.3f}mm"
            if d_min is not None: msg += f" | d_min={d_min*1e3:.2f}mm"
            print(msg)
            print_root_pose(model, data, base_body, lego_geom=lego_geom, tag=f"[DESC it={it:03d}]")

    print("[DESC] Descent finished (safe mode)")

# ====================== 改进的SDF外侧锁定 ======================  
def improved_sdf_pregrasp_lockroot(model, data, base_body, motor, siteA, siteB,
                                  lego_geom, if_geom, th_geom,
                                  p1, p2, outside_extra, gap_margin,
                                  align_iters=300, max_shift_step=1e-3, v=None,
                                  z_guard_min=0.015):
    """更稳的外侧锁定"""
    d = normalize(p2 - p1)
    D = np.linalg.norm(p2 - p1)
    thr_neg = -0.5*D - outside_extra
    thr_pos =  0.5*D + outside_extra  
    need_para = D + 2*outside_extra + gap_margin
    target_A = p1 - outside_extra * d
    target_B = p2 + outside_extra * d
    m = 0.5 * (p1 + p2)

    motor_track_gap(model, data, siteA, siteB, motor, need_para, 
                    iters=60, k=0.6, v=v, base_body=base_body, 
                    z_guard_min=z_guard_min)

    success_count = 0
    required_success = 5
    for it in range(int(align_iters)):
        ci = query_sdf_dist_pos_normal(model, data, lego_geom, if_geom)
        ct = query_sdf_dist_pos_normal(model, data, lego_geom, th_geom)
        if (ci is None) or (ct is None):
            mj_step_n_pin_root(model, data, base_body, 2, v=v)
            success_count = 0
            continue
        _, pi, _ = ci
        _, pt, _ = ct

        alpha_i = float(np.dot(pi - m, d))
        alpha_t = float(np.dot(pt - m, d))
        lo, hi = min(alpha_i, alpha_t), max(alpha_i, alpha_t)
        gap_para = hi - lo

        if gap_para < need_para * 0.95:
            motor_track_gap(model, data, siteA, siteB, motor, need_para, 
                            iters=8, k=0.8, v=v, base_body=base_body, 
                            z_guard_min=z_guard_min)
            success_count = 0
            continue

        def perp_error(p, target):
            e = p - target
            return e - np.dot(e, d) * d
            
        e_perp_A = perp_error(pi, target_A)
        e_perp_B = perp_error(pt, target_B)
        e_perp = 0.5 * (e_perp_A + e_perp_B)
        en = np.linalg.norm(e_perp)
        if en > max_shift_step:
            e_perp *= (max_shift_step / (en + 1e-12))

        # 匀速沿 d 方向微调（将 [lo,hi] 推到 [thr_neg,thr_pos]）
        s_lo = lo - thr_neg
        s_hi = hi - thr_pos  
        s = 0.5 * (s_lo + s_hi)
        s = float(np.clip(s, -max_shift_step, max_shift_step))

        pose = get_body_freejoint_pose(model, data, base_body)
        if pose is None: break
        pos, quat = pose
        pos_new = pos - s*d - e_perp

        set_body_freejoint_pose(model, data, base_body, pos_new, quat)
        ensure_upright_and_above_floor(model, data, base_body, z_guard_min, yaw_only=True)
        mj_step_n_pin_root(model, data, base_body, 1, v=v)

        in_range = (lo <= thr_neg + 1e-4) and (hi >= thr_pos - 1e-4)
        perp_err = np.linalg.norm(e_perp_A) + np.linalg.norm(e_perp_B)
        aligned = perp_err < 2e-3

        if in_range and aligned:
            success_count += 1
            if success_count >= required_success:
                print(f"[LOCKROOT] Success at iter {it}")
                return True
        else:
            success_count = 0

        if it % 15 == 0:
            print(f"[LOCKROOT] it={it:03d} | alpha=({lo*1e3:.1f},{hi*1e3:.1f})mm -> target=({thr_neg*1e3:.1f},{thr_pos*1e3:.1f})mm | perp_err={perp_err*1e3:.2f}mm | ok={success_count}/{required_success}")
            print_root_pose(model, data, base_body, lego_geom=lego_geom, tag=f"[LOCK it={it:03d}]")
    print(f"[LOCKROOT] Max iterations reached, success_count={success_count}")
    return success_count >= required_success

# ====================== 评估、回退与线搜索（用于终端优化） ======================
def eval_state(model, data, lego_geom, if_geom, th_geom, cos_alpha, dist_tol):
    pair = get_contact_pair(model, data, lego_geom, if_geom, th_geom)
    if pair is None: return None
    J, (c_if, c_th, c_gap) = friction_cost_from_contacts(pair, cos_alpha, dist_tol)
    (d_if, p_if, n_if), (d_th, p_th, n_th), d_hat = pair
    ok_cone = (np.dot(n_if, +d_hat) >= cos_alpha) and (np.dot(n_th, -d_hat) >= cos_alpha)
    ok_dist = (d_if < dist_tol) and (d_th < dist_tol)
    return dict(J=J, c_if=c_if, c_th=c_th, c_gap=c_gap,
                ok_cone=ok_cone, ok_dist=ok_dist,
                d_if=d_if, d_th=d_th, d_hat=d_hat)

def save_root_pose(model, data, base_body):
    return get_body_freejoint_pose(model, data, base_body)

def restore_root_pose(model, data, base_body, pose):
    if pose is not None:
        set_body_freejoint_pose(model, data, base_body, *pose)
        mj.mj_forward(model, data)

def adaptive_gd_linesearch(model, data, args, grad, cos_alpha, dist_tol, v=None):
    """保守线搜索"""
    if grad is None: return False, None
    pose0 = save_root_pose(model, data, args.base_body)
    gt, gr = grad[:3], grad[3:]
    grad_norm_t = np.linalg.norm(gt)
    grad_norm_r = np.linalg.norm(gr)
    eta_t = args.gd_eta_t * min(1.0, 1e-3 / (grad_norm_t + 1e-6))
    eta_r = args.gd_eta_r * min(1.0, 1e-2 / (grad_norm_r + 1e-6))
    dtrans = -eta_t * gt
    drot   = -eta_r * gr
    nt = float(np.linalg.norm(dtrans))
    nr = float(np.linalg.norm(drot))
    if nt > args.root_step_trans: dtrans *= (args.root_step_trans / (nt + 1e-12))
    if nr > args.root_step_rot:   drot   *= (args.root_step_rot   / (nr + 1e-12))
    delta = np.concatenate([dtrans, drot], axis=0)

    base = eval_state(model, data, args.lego_geom, args.if_geom, args.th_geom, cos_alpha, dist_tol)
    if base is None: return False, None
    J0 = base['J']
    scales = [0.7, 0.5, 0.3, 0.15, 0.08]
    for scale in scales:
        restore_root_pose(model, data, args.base_body, pose0)
        success = apply_delta_pose_se3(model, data, args.base_body, delta * scale)
        if not success: continue
        mj_step_n_pin_root(model, data, args.base_body, 1, v=v)
        ensure_upright_and_above_floor(model, data, args.base_body, args.z_guard_min, yaw_only=True)
        cur = eval_state(model, data, args.lego_geom, args.if_geom, args.th_geom, cos_alpha, dist_tol)
        if cur is None: continue
        improved_J   = cur['J'] < J0 * 0.995
        improved_cone= (cur['c_if'] <= base['c_if'] * 0.95) and (cur['c_th'] <= base['c_th'] * 0.95)
        no_worse     = cur['J'] <= J0 * 1.01
        if improved_J or (improved_cone and no_worse):
            jid, qadr, dof = find_freejoint_of_body(model, args.base_body)
            if dof is not None: data.qvel[dof:dof+6] = 0.0
            return True, cur
    restore_root_pose(model, data, args.base_body, pose0)
    mj_step_n_pin_root(model, data, args.base_body, 1, v=v)
    return False, base

def smart_retreat(model, data, args, v, d_hat=None, retreat_level=1):
    print(f"[RETREAT] Level {retreat_level}")
    try:
        ga = site_xpos(model, data, args.siteA); gb = site_xpos(model, data, args.siteB)
        gap_now = float(np.linalg.norm(ga - gb))
    except:
        gap_now = 0.01
    if retreat_level == 1:
        gap_increase, z_lift, xy_retreat = args.backoff_gap, args.backoff_z, 0.0
    elif retreat_level == 2:
        gap_increase, z_lift, xy_retreat = args.backoff_gap*1.5, args.backoff_z*1.5, 1e-3
    else:
        gap_increase, z_lift, xy_retreat = args.backoff_gap*2.0, args.backoff_z*2.0, 2e-3
    target_gap = gap_now + gap_increase
    motor_track_gap(model, data, args.siteA, args.siteB, args.motor,
                    target_gap, iters=25, v=v, k=1.0,
                    base_body=args.base_body, z_guard_min=args.z_guard_min)
    pose = get_body_freejoint_pose(model, data, args.base_body)
    if pose is not None:
        pos, quat = pose
        step = np.array([0, 0, z_lift], float)
        if d_hat is not None:
            step += (-xy_retreat) * d_hat
        set_body_freejoint_pose(model, data, args.base_body, pos + step, quat)
        mj_step_n_pin_root(model, data, args.base_body, 10, v=v)
        ensure_upright_and_above_floor(model, data, args.base_body, args.z_guard_min, yaw_only=True)

# ====================== 拇指优先：拇指移动到悬空目标 ======================
def move_thumb_to_antipodal_point(model, data, base_body, siteA, siteB, p1, p2, hover_z=0.01):
    """
    第一步：把拇指(siteB)移动到离它更近的 antipodal 点的“上方 hover_z”处；
    返回 (success, thumb_target_world, finger_target_world, used_hover_z)
    """
    try:
        a_now = site_xpos(model, data, siteA)
        b_now = site_xpos(model, data, siteB)

        if np.linalg.norm(b_now - p1) <= np.linalg.norm(b_now - p2):
            thumb_ground  = p1
            finger_ground = p2
            print(f"[THUMB-MOVE] Thumb -> p1 (finger -> p2)")
        else:
            thumb_ground  = p2
            finger_ground = p1
            print(f"[THUMB-MOVE] Thumb -> p2 (finger -> p1)")

        thumb_target_w  = thumb_ground  + np.array([0,0,hover_z])
        finger_target_w = finger_ground + np.array([0,0,hover_z])

        root_pos, root_quat = get_body_freejoint_pose(model, data, base_body)
        thumb_offset = b_now - root_pos
        new_root_pos = thumb_target_w - thumb_offset
        set_body_freejoint_pose(model, data, base_body, new_root_pos, root_quat)
        mj.mj_forward(model, data)

        b_new = site_xpos(model, data, siteB)
        err   = np.linalg.norm(b_new - thumb_target_w)
        print(f"[THUMB-MOVE] thumb_err={err*1e3:.2f}mm | thumb_now={b_new} | thumb_tgt={thumb_target_w}")

        return True, thumb_target_w, finger_target_w, hover_z
    except Exception as e:
        print(f"[THUMB-MOVE] Error: {e}")
        return False, None, None, None

# ====================== 联合优化（motor+root，对齐两端悬空目标） ======================
def joint_motor_root_optimization(model, data, args,
                                  thumb_target_w, finger_target_w,
                                  cos_alpha, dist_tol, v=None, hover_z_opt=None):
    """
    同时调 motor + root，让 (siteB,siteA) → (thumb_target_w, finger_target_w)
    * 使用悬空目标，确保“预处理阶段不接触”
    * root 步进有限幅，且包含 yaw 微调
    """
    print(f"[JOINT-OPT] Start (hover_z={hover_z_opt if hover_z_opt is not None else 'N/A'})")
    target_distance = float(np.linalg.norm(finger_target_w - thumb_target_w))
    print(f"[JOINT-OPT] Target gap {target_distance*1e3:.2f}mm")

    best_pose  = save_root_pose(model, data, args.base_body)
    motor_id   = name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, args.motor)
    best_motor = data.ctrl[motor_id]
    best_cost  = float('inf')

    for it in range(args.max_iter):
        a = site_xpos(model, data, args.siteA)
        b = site_xpos(model, data, args.siteB)

        e_t = b - thumb_target_w
        e_f = a - finger_target_w

        gap_now = float(np.linalg.norm(a - b))
        gap_err = gap_now - target_distance

        motor_track_gap(model, data, args.siteA, args.siteB, args.motor,
                        target_distance, iters=2, k=0.6, v=v,
                        base_body=args.base_body, z_guard_min=args.z_guard_min)

        pos_cost  = np.linalg.norm(e_t)**2 + np.linalg.norm(e_f)**2
        gap_cost  = (gap_err**2)
        total     = 1e6*pos_cost + 1e6*gap_cost

        if total < best_cost:
            best_cost = total
            best_pose = save_root_pose(model, data, args.base_body)
            best_motor= data.ctrl[motor_id]

        if (np.linalg.norm(e_t) < 1.0e-3 and
            np.linalg.norm(e_f) < 1.0e-3 and
            abs(gap_err) < 5e-4):
            print(f"[JOINT-OPT] Converged at iter {it}")
            return True

        step_vec = 0.4*e_t + 0.6*e_f
        step_norm= float(np.linalg.norm(step_vec))
        if step_norm > args.root_dxy:
            step_vec *= args.root_dxy / (step_norm + 1e-12)

        cur_dir = normalize(b - a)
        tgt_dir = normalize(finger_target_w - thumb_target_w)
        yaw_err = math.atan2(np.cross(cur_dir, tgt_dir)[2], np.dot(cur_dir, tgt_dir))
        dquat   = quat_from_axis_angle(np.array([0,0,1.0]), float(np.clip(-0.5*yaw_err, -args.root_dang, args.root_dang)))

        pos, quat = get_body_freejoint_pose(model, data, args.base_body)
        pos_new   = pos - step_vec
        if pos_new[2] < args.z_guard_min:
            pos_new[2] = args.z_guard_min
        quat_new  = quat_mul(dquat, quat)
        set_body_freejoint_pose(model, data, args.base_body, pos_new, quat_new)
        ensure_upright_and_above_floor(model, data, args.base_body, args.z_guard_min, yaw_only=True)

        mj_step_n_pin_root(model, data, args.base_body, 3, v=v)

        if it % 10 == 0:
            print(f"[JOINT-OPT] it={it:03d} | gap={gap_now*1e3:.2f}->{target_distance*1e3:.2f}mm | "
                  f"thumb_err={np.linalg.norm(e_t)*1e3:.2f}mm | finger_err={np.linalg.norm(e_f)*1e3:.2f}mm | "
                  f"yaw_err={yaw_err*180/np.pi:.2f}deg")

        if it % 60 == 45 and total > best_cost * 2.0:
            restore_root_pose(model, data, args.base_body, best_pose)
            data.ctrl[motor_id] = best_motor
            mj_step_n_pin_root(model, data, args.base_body, 3, v=v)

    a = site_xpos(model, data, args.siteA)
    b = site_xpos(model, data, args.siteB)
    print(f"[JOINT-OPT] Final thumb_err={np.linalg.norm(b-thumb_target_w)*1e3:.2f}mm "
          f"| finger_err={np.linalg.norm(a-finger_target_w)*1e3:.2f}mm "
          f"| gap={np.linalg.norm(a-b)*1e3:.2f}mm (target={target_distance*1e3:.2f}mm)")
    return False

# ====================== 终端优化（摩擦锥+距离） ======================
def run_improved_optimize(model, data, args, p1, p2, D, v=None):
    cos_alpha = math.cos(math.atan(args.mu))
    success = False
    G_open = D + 2*args.outside_extra
    best_pose = save_root_pose(model, data, args.base_body)
    best_J = float('inf')
    bad_cone_cnt = 0
    stall_cnt = 0
    retreat_level = 1

    print(f"[OPT] cos_alpha={cos_alpha:.3f}, D={D*1e3:.2f}mm")

    for it in range(args.max_iter):
        progress = min(1.0, it / max(args.max_iter * 0.7, 1))
        G_t = G_open - (G_open - (D - args.squeeze)) * progress
        G_t = max(D - args.squeeze, G_t)
        motor_track_gap(model, data, args.siteA, args.siteB, args.motor,
                        G_t, iters=15, k=0.6, v=v, base_body=args.base_body,
                        k_far=3.0, k_near=1.2, z_guard_min=args.z_guard_min)

        cur_eval = eval_state(model, data, args.lego_geom, args.if_geom, args.th_geom, cos_alpha, args.dist_tol)
        if cur_eval is None:
            mj_step_n_pin_root(model, data, args.base_body, 3, v=v)
            stall_cnt += 1
            if stall_cnt > 10:
                print("[WARN] Too many evaluation failures")
                break
            continue
        stall_cnt = 0
        current_J = cur_eval['J']

        grad, J0, parts = numeric_grad_cost_root(
            model, data, args.base_body, args.lego_geom, args.if_geom, args.th_geom,
            cos_alpha, args.dist_tol,
            h_trans=args.gd_h_trans, h_rot=args.gd_h_rot, pin_steps=2
        )

        step_ok, cur_after = adaptive_gd_linesearch(model, data, args, grad, cos_alpha, args.dist_tol, v=v)

        if current_J < best_J:
            best_J = current_J
            best_pose = save_root_pose(model, data, args.base_body)
            retreat_level = 1

        ok_dist = cur_eval['ok_dist']
        ok_cone = cur_eval['ok_cone']

        if (it % 8 == 0) or ok_dist or ok_cone:
            try:
                ga = site_xpos(model, data, args.siteA); gb = site_xpos(model, data, args.siteB)
                gap_now = np.linalg.norm(ga-gb)*1e3
            except:
                gap_now = 0.0
            print(f"[OPT {it:03d}] gap={gap_now:.2f}mm | J={current_J:.3e} "
                  f"c_if={cur_eval['c_if']:.3e} c_th={cur_eval['c_th']:.3e} c_gap={cur_eval['c_gap']:.3e} | "
                  f"d_if={cur_eval['d_if']*1e3:.2f}mm d_th={cur_eval['d_th']*1e3:.2f}mm | "
                  f"step_ok={step_ok} | cone={ok_cone} dist={ok_dist}")
            if it % 20 == 0:
                print_root_pose(model, data, args.base_body, lego_geom=args.lego_geom, tag=f"[OPT it={it:03d}]")

        if ok_dist and ok_cone:
            success = True
            print(f"[SUCCESS] 满足摩擦锥+距离条件: it={it}")
            print(f"[SUCCESS] d_if={cur_eval['d_if']*1e3:.2f}mm, d_th={cur_eval['d_th']*1e3:.2f}mm")
            break

        if not ok_cone: 
            bad_cone_cnt += 1
        else:
            bad_cone_cnt = max(0, bad_cone_cnt - 1)

        if bad_cone_cnt >= args.bad_cone_max:
            print(f"[RETREAT] Bad cone count {bad_cone_cnt} -> Level {retreat_level} retreat")
            restore_root_pose(model, data, args.base_body, best_pose)
            mj_step_n_pin_root(model, data, args.base_body, 3, v=v)
            smart_retreat(model, data, args, v, d_hat=cur_eval.get('d_hat'), retreat_level=retreat_level)
            bad_cone_cnt = 0
            retreat_level = min(3, retreat_level + 1)
            mj_step_n_pin_root(model, data, args.base_body, 10, v=v)

        if it > 50 and current_J > best_J * 2.0:
            print("[STAGNATION] Cost too high vs best, revert")
            restore_root_pose(model, data, args.base_body, best_pose)
            mj_step_n_pin_root(model, data, args.base_body, 5, v=v)

    return success

# ====================== 结果与提升 ======================
def report_final_state(model, data, args):
    try:
        Rg, tg = geom_RT(model, data, args.lego_geom)
        Rb, tb = body_RT(model, data, args.base_body)
        p_L = Rg.T @ (tb - tg)
        print(f"[RESULT] root 在 LEGO 坐标系位置 (m): x={p_L[0]:.6f}, y={p_L[1]:.6f}, z={p_L[2]:.6f}")
        try:
            a = site_xpos(model, data, args.siteA)
            b = site_xpos(model, data, args.siteB)
            final_gap = np.linalg.norm(a - b)
            print(f"[RESULT] 最终夹爪间距: {final_gap*1e3:.2f}mm")
            a_L = Rg.T @ (a - tg)
            b_L = Rg.T @ (b - tg)
            print(f"[RESULT] siteA 在 LEGO 坐标系: [{a_L[0]:.4f}, {a_L[1]:.4f}, {a_L[2]:.4f}]")
            print(f"[RESULT] siteB 在 LEGO 坐标系: [{b_L[0]:.4f}, {b_L[1]:.4f}, {b_L[2]:.4f}]")
        except Exception as e:
            print(f"[WARN] Cannot get site positions: {e}")
        return p_L
    except Exception as e:
        print(f"[ERROR] Cannot report final state: {e}")
        return None

def perform_success_lift(model, data, args, v=None, lift_height=0.03, lift_steps=300):
    print(f"[LIFT] Start: {(lift_height)*1e3:.0f}mm over {lift_steps} steps")
    jid, qadr, _ = find_freejoint_of_body(model, args.base_body)
    if qadr is None:
        print("[WARN] Cannot find freejoint for lifting")
        return
    initial_z = data.qpos[qadr+2]
    target_z = initial_z + lift_height
    for step in range(int(lift_steps)):
        progress = (step + 1) / lift_steps
        current_z = initial_z + lift_height * progress
        data.qpos[qadr+2] = current_z
        mj.mj_forward(model, data)
        if step % 10 == 0:
            try:
                ga = site_xpos(model, data, args.siteA)
                gb = site_xpos(model, data, args.siteB)
                current_gap = np.linalg.norm(ga - gb)
                motor_track_gap(model, data, args.siteA, args.siteB, args.motor,
                                current_gap * 0.98, iters=3, k=0.3, v=v,
                                base_body=args.base_body, z_guard_min=args.z_guard_min)
            except:
                pass
        mj_step_n_pin_root(model, data, args.base_body, 1, v=v)
        if step % 50 == 0:
            print(f"[LIFT] Step {step}/{lift_steps}, height: {(current_z-initial_z)*1e3:.1f}mm")
    print(f"[LIFT] Completed lift")

# ====================== 主策略：拇指优先 + 温柔下降 + 外侧锁定 ======================
def run_thumb_first_approach(model, data, args, v=None):
    # 1) 设置拇指根关节
    try:
        actuator_ctrl(model, data, args.th_root_act, args.th_root_q)
        mj_step_n(model, data, 10, v)
        print(f"[INFO] th_root via controller -> {args.th_root_q:.3f}")
    except Exception:
        try:
            set_hinge_qpos(model, data, args.th_root_joint, args.th_root_q)
            print(f"[INFO] th_root set qpos (fallback) -> {args.th_root_q:.3f}")
        except Exception as e:
            print(f"[WARN] Failed to set thumb root: {e}")

    # 2) 等待LEGO稳定
    print('[INFO] 等待 LEGO 落稳...')
    try: 
        wait_body_settled(model, data, args.lego_body, v=v, max_steps=200)
    except Exception as e:
        print(f"[WARN] LEGO settle check failed: {e}")

    # 3) 选择antipodal点对
    p1, p2, n_obj, D = pick_pair_world(args.npy, model, data, args.lego_geom, minD_m=args.minD_m)
    print(f'[ANTIPODAL] D = {D*1e3:.2f} mm | p1={p1} | p2={p2}')
    print_root_pose(model, data, args.base_body, lego_geom=args.lego_geom, tag="[INITIAL]")

    # 4) 第一步：移动拇指到antipodal点（悬空）
    print("\n" + "="*50)
    print("STEP 1: MOVE THUMB TO ANTIPODAL (HOVER)")
    print("="*50)
    hz = max(0.006, args.approach_h*0.4)
    ok_thumb, thumb_target_w, finger_target_w, hz = move_thumb_to_antipodal_point(
        model, data, args.base_body, args.siteA, args.siteB, p1, p2, hover_z=hz
    )
    if not ok_thumb:
        raise RuntimeError("[THUMB-MOVE] Failed to move thumb to antipodal point")
    print_root_pose(model, data, args.base_body, lego_geom=args.lego_geom, tag="[AFTER-THUMB-MOVE]")
    mj_step_n_pin_root(model, data, args.base_body, 20, v=v)

    # 5) 第二步：联合优化（不接触，悬空对齐）
    print("\n" + "="*50)
    print("STEP 2: JOINT MOTOR + ROOT (HOVER ALIGN)")
    print("="*50)
    cos_alpha = math.cos(math.atan(args.mu))
    ok_joint = joint_motor_root_optimization(
        model, data, args, thumb_target_w, finger_target_w, cos_alpha, args.dist_tol, v=v, hover_z_opt=hz
    )
    print_root_pose(model, data, args.base_body, lego_geom=args.lego_geom, tag="[AFTER-JOINT-OPT]")

    # 6) 第三步：温柔下降 + 外侧锁定（SDF门控）
    print("\n" + "="*50)
    print("STEP 3: GENTLE DESCENT + OUTSIDE LOCK")
    print("="*50)
    smart_pregrasp_descent(
        model, data, args.base_body, args.siteA, args.siteB,
        p1, p2, outside_extra=args.outside_extra, gap_margin=args.gap_margin,
        dz=args.dz, steps=args.down_steps, max_xy_step=args.root_dxy,
        v=v, motor=args.motor, z_guard_min=args.z_guard_min,
        lego_geom=args.lego_geom, if_geom=args.if_geom, th_geom=args.th_geom
    )
    ok_outside = improved_sdf_pregrasp_lockroot(
        model, data, args.base_body, args.motor, args.siteA, args.siteB,
        args.lego_geom, args.if_geom, args.th_geom,
        p1, p2, outside_extra=args.outside_extra, gap_margin=args.gap_margin,
        align_iters=args.outside_iters, max_shift_step=args.root_dxy/2, v=v,
        z_guard_min=args.z_guard_min
    )
    print(f"[LOCKROOT] Success: {ok_outside}")
    print_root_pose(model, data, args.base_body, lego_geom=args.lego_geom, tag="[LOCKROOT-END]")

    return p1, p2, D

# ====================== 入口函数 ======================
def main():
    ap = argparse.ArgumentParser(description="Safe LEGO Grasping (Thumb-first, Hover Align, SDF-gated Descent)")
    # 基本文件
    ap.add_argument('--xml',  default='assets/tsinghua_lego.xml', help='MuJoCo XML file')
    ap.add_argument('--npy',  default='results/longshort_pairs_fc/pairs_brick2_10_medium_lavender.npy', help='Antipodal pairs file')
    ap.add_argument('--viewer', action='store_true', help='Enable MuJoCo viewer')
    # 模型命名（请与XML一致）
    ap.add_argument('--lego_body', type=str, default='plate3_3_tan')
    ap.add_argument('--lego_geom', type=str, default='plate3_3_tan')
    ap.add_argument('--if_geom',   type=str, default='if_distal_link_collision')
    ap.add_argument('--th_geom',   type=str, default='th_distal_link_collision')
    ap.add_argument('--siteA', type=str, default='if_distal_site_a')  # 食指
    ap.add_argument('--siteB', type=str, default='th_distal_site_a')  # 拇指
    # 控制器/关节
    ap.add_argument('--motor', type=str, default='gripper_motor')
    ap.add_argument('--th_root_joint', type=str, default='th_root_link')  
    ap.add_argument('--th_root_act',   type=str, default='th_root_link')
    ap.add_argument('--th_root_q',     type=float, default=1.4)
    # 数值梯度（终端优化用）
    ap.add_argument('--gd_h_trans',      type=float, default=1e-4)
    ap.add_argument('--gd_h_rot',        type=float, default=1e-3)
    ap.add_argument('--gd_eta_t',        type=float, default=5e-3)
    ap.add_argument('--gd_eta_r',        type=float, default=1e-2)
    ap.add_argument('--root_step_trans', type=float, default=8e-4)
    ap.add_argument('--root_step_rot',   type=float, default=1e-3)
    # base freejoint（hand root）
    ap.add_argument('--base_body', type=str, default='base_link')
    ap.add_argument('--approach_h', type=float, default=0.025, help='Hover height (m)')
    # 预抓取（慢且稳）
    ap.add_argument('--ang_tol',       type=float, default=3.0*np.pi/180.0)
    ap.add_argument('--outside_extra', type=float, default=0.008)
    ap.add_argument('--gap_margin',    type=float, default=1.2e-3)
    ap.add_argument('--outside_iters', type=int,   default=300)
    ap.add_argument('--dz',            type=float, default=2e-4)
    ap.add_argument('--down_steps',    type=int,   default=400)
    ap.add_argument('--root_dxy',      type=float, default=4e-4)
    ap.add_argument('--root_dang',     type=float, default=4e-4)
    # 终端优化
    ap.add_argument('--squeeze',     type=float, default=2e-4)
    ap.add_argument('--dist_tol',    type=float, default=1e-3)
    ap.add_argument('--mu',          type=float, default=0.5)
    ap.add_argument('--minD_m',      type=float, default=0.003)
    ap.add_argument('--max_iter',    type=int,   default=400)
    # 安全和回退（更保守）
    ap.add_argument('--z_guard_min',   type=float, default=0.015)
    ap.add_argument('--bad_cone_max',  type=int,   default=12)
    ap.add_argument('--backoff_z',     type=float, default=0.008)
    ap.add_argument('--backoff_gap',   type=float, default=0.003)

    args = ap.parse_args()
    if not os.path.isfile(args.xml): 
        raise FileNotFoundError(f"XML file not found: {args.xml}")
    if not os.path.isfile(args.npy): 
        raise FileNotFoundError(f"NPY file not found: {args.npy}")

    print(f"[INFO] Loading model: {args.xml}")
    print(f"[INFO] Loading antipodal pairs: {args.npy}")
    model = mj.MjModel.from_xml_path(args.xml)
    data  = mj.MjData(model)
    print(f"[INFO] Model loaded: nq={model.nq}, nv={model.nv}")

    # 预设置拇指根关节
    try:
        actuator_ctrl(model, data, args.th_root_act, args.th_root_q)
        mj_step_n(model, data, 50)
        print(f"[INFO] Pre-set th_root via controller -> {args.th_root_q:.3f}")
    except Exception:
        try:
            set_hinge_qpos(model, data, args.th_root_joint, args.th_root_q)
            print(f"[INFO] Pre-set th_root via qpos -> {args.th_root_q:.3f}")
        except Exception as e:
            print(f"[WARN] Failed to pre-set thumb root: {e}")

    if args.viewer:
        print("[INFO] Starting viewer...")
        with mjviewer.launch_passive(model, data) as v:
            try:
                print("\n" + "="*60)
                print("PHASE 1: THUMB-FIRST + HOVER ALIGN + SAFE DESCENT")
                print("="*60)
                p1, p2, D = run_thumb_first_approach(model, data, args, v)

                # 短暂停顿观察
                for _ in range(50): mj_step_n_pin_root(model, data, args.base_body, 1, v=v)

                print("\n" + "="*60)
                print("PHASE 2: TERMINAL OPTIMIZATION (CONE + DIST)")
                print("="*60)
                success = run_improved_optimize(model, data, args, p1, p2, D, v)

                if success:
                    print("\n" + "="*60)
                    print("SUCCESS: LIFT")
                    print("="*60)
                    perform_success_lift(model, data, args, v=v)
                    report_final_state(model, data, args)
                    print("[SUCCESS] 抓取任务完成！物体已成功提升。")
                else:
                    print("\n" + "="*60)
                    print("OPTIMIZATION FAILED")
                    print("="*60)
                    print('[FAIL] 未达成功判据；建议：检查 pairs 的 frame/scale、增大 --outside_extra/--down_steps、放宽 --dist_tol 或调整 --mu、核对几何命名')

                print("\n[INFO] 任务完成，viewer保持运行。Ctrl+C 退出。")
                try:
                    while v.is_running():
                        mj_step_n_pin_root(model, data, args.base_body, 1, v=v)
                except KeyboardInterrupt:
                    print("\n[INFO] User interrupted, exiting...")

            except Exception as e:
                print(f"\n[ERROR] Exception during execution: {e}")
                import traceback; traceback.print_exc()

    else:
        print("[INFO] Running headless...")
        try:
            print("\n" + "="*60)
            print("PHASE 1: THUMB-FIRST + HOVER ALIGN + SAFE DESCENT")
            print("="*60)
            p1, p2, D = run_thumb_first_approach(model, data, args, v=None)

            print("\n" + "="*60)
            print("PHASE 2: TERMINAL OPTIMIZATION (CONE + DIST)")
            print("="*60)
            success = run_improved_optimize(model, data, args, p1, p2, D, v=None)

            if success:
                print("\n" + "="*60)
                print("SUCCESS: LIFT")
                print("="*60)
                perform_success_lift(model, data, args, v=None)
                report_final_state(model, data, args)
                print("[SUCCESS] 抓取任务完成！")
            else:
                print("\n" + "="*60)
                print("OPTIMIZATION FAILED")
                print("="*60)
                print('[FAIL] 未达成功判据；建议：检查 pairs 的 frame/scale、增大 --outside_extra/--down_steps、放宽 --dist_tol 或调整 --mu、核对几何命名、使用 --viewer 观察')
        except Exception as e:
            print(f"\n[ERROR] Exception during execution: {e}")
            import traceback; traceback.print_exc()
            return 1
    return 0

if __name__ == '__main__':
    exit(main())
