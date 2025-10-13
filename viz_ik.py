#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse
import numpy as np
import mujoco as mj

# ------------------ utils: IO ------------------
def load_pairs_npy(path: str):
    payload = np.load(path, allow_pickle=True).item()
    pairs = np.asarray(payload["pairs"], np.float32)  # (K,2,3) LEGO local coords
    lego_id = payload.get("lego_id", os.path.splitext(os.path.basename(path))[0])
    if pairs.ndim != 3 or pairs.shape[1:] != (2,3):
        raise RuntimeError(f"pairs.npy 形状异常，期望 (K,2,3)，得到 {pairs.shape}")
    if pairs.shape[0] == 0:
        raise RuntimeError("pairs.npy 为空，无法挑选对点")
    return lego_id, pairs

def resolve_site_name(model, name: str):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
    if sid >= 0: return name
    if "_ball" in name:
        alt = name.replace("_ball", "_a")
        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, alt) >= 0:
            print(f"[info] site '{name}' 不存在，自动改用 '{alt}'")
            return alt
    raise RuntimeError(f"site '{name}' not found")

def get_site_pos(model, data, site_name: str):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0: raise RuntimeError(f"site '{site_name}' not found")
    return data.site_xpos[sid].copy()

def lego_pose(model, data, geom_or_body: str):
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_or_body)
    if gid < 0:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, geom_or_body)
        if bid < 0: raise RuntimeError(f"geom/body '{geom_or_body}' not found")
    else:
        bid = int(model.geom_bodyid[gid])
    p = data.xpos[bid].copy()
    R = data.xmat[bid].reshape(3,3).copy()
    return p, R

def world_from_lego_local(p_local, lego_p, lego_R):
    return lego_p + lego_R @ p_local

def mat_to_quat_xyzw(R):
    q = np.empty(4, dtype=np.float64)
    mj.mju_mat2Quat(q, R.reshape(9))
    return q

def rodrigues(axis, angle):
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12: return np.eye(3)
    a = a/n
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]], np.float64)
    c, s = np.cos(angle), np.sin(angle)
    return np.eye(3)+s*K+(1-c)*(K@K)

# ------------------ id/name helpers ------------------
def _jid(model, name):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0: raise RuntimeError(f"joint '{name}' not found")
    return int(jid)

def _aid_for_joint(model, joint_id):
    for ai in range(model.nu):
        if int(model.actuator_trnid[ai,0]) == int(joint_id):
            return int(ai)
    return -1

# ------------------ keep other actuators position ------------------
def _hold_other_position_actuators(model, data, exclude_aids):
    ex = set(exclude_aids)
    for ai in range(model.nu):
        if ai in ex: continue
        j = int(model.actuator_trnid[ai,0])
        if j >= 0 and model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            qadr = int(model.jnt_qposadr[j])
            data.ctrl[ai] = float(data.qpos[qadr])

# ------------------ ramp motors (不改全局碰撞开关) ------------------
def ramp_motors_to_ctrl(model, data, joint_names, target_ctrl,
                        ramp_steps=300, settle_steps=150,
                        zero_gravity=None, disable_contact=None,
                        viewer=None, draw_fn=None):

    # （可选）如果传了开关，就按位运算设置；否则不动全局设置
    if zero_gravity is True:
        model.opt.gravity[:] = 0.0
    elif zero_gravity is False:
        pass  # 使用 xml 里的重力

    if disable_contact is True:
        model.opt.disableflags = int(model.opt.disableflags) | int(mj.mjtDisableBit.mjDSBL_CONTACT)
    elif disable_contact is False:
        model.opt.disableflags = int(model.opt.disableflags) & ~int(mj.mjtDisableBit.mjDSBL_CONTACT)

    jids = [_jid(model, n) for n in joint_names]
    aids = [_aid_for_joint(model, j) for j in jids]
    if any(a < 0 for a in aids):
        miss = [joint_names[i] for i, a in enumerate(aids) if a < 0]
        raise RuntimeError(f"未找到执行器: {miss}")

    qaddrs = [int(model.jnt_qposadr[j]) for j in jids]
    cur = np.array([float(data.qpos[q]) for q in qaddrs], np.float64)
    tgt = np.asarray(target_ctrl, np.float64).ravel()[:len(joint_names)]

    # ramp
    for t in range(int(ramp_steps)):
        alpha = (t + 1) / float(ramp_steps)
        cmd = (1 - alpha) * cur + alpha * tgt
        for k, ai in enumerate(aids):
            data.ctrl[ai] = float(cmd[k])
        _hold_other_position_actuators(model, data, aids)
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mj.mj_step(model, data)

        if viewer is not None and draw_fn is not None:
            viewer.user_scn.ngeom = 0
            draw_fn()
            viewer.sync()
            time.sleep(1/120)

    # settle
    for _ in range(int(settle_steps)):
        _hold_other_position_actuators(model, data, aids)
        mj.mj_step(model, data)

        if viewer is not None and draw_fn is not None:
            viewer.user_scn.ngeom = 0
            draw_fn()
            viewer.sync()
            time.sleep(1/120)

# ------------------ freejoint helpers ------------------
def get_freejoint_qadr(model, name="root"):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0 or model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"freejoint '{name}' 不存在或不是 freejoint")
    return int(model.jnt_qposadr[jid])

def set_freejoint_pose_about_point(model, data, qadr, base_pos, base_quat, R_world, pivot_world):
    q_rot = mat_to_quat_xyzw(R_world)
    q_new = np.empty(4); mj.mju_mulQuat(q_new, q_rot, base_quat)
    p0 = pivot_world
    p_new = p0 + R_world @ (base_pos - p0)
    data.qpos[qadr:qadr+3]   = p_new
    data.qpos[qadr+3:qadr+7] = q_new
    mj.mj_forward(model, data)
    return p_new, q_new

# ------------------ draw helpers ------------------
def add_sphere(viewer, pos, radius, rgba):
    scn = viewer.user_scn
    g = scn.geoms[scn.ngeom]
    size = np.array([radius, radius, radius], dtype=np.float32)
    mat  = np.eye(3, dtype=np.float32).reshape(-1)
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    scn.ngeom += 1

# ------------------ contact mask (屏蔽/恢复手部碰撞) ------------------
class ContactMask:
    def __init__(self, model, data, body_substrs):
        self.model = model
        self.data  = data
        self.body_substrs = tuple(body_substrs)
        gids = []
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
            if any(s in bname for s in self.body_substrs):
                gids.append(gid)
        self.gids = np.asarray(gids, dtype=np.int32)
        self._orig_type = model.geom_contype[self.gids].copy() if len(self.gids)>0 else None
        self._orig_aff  = model.geom_conaffinity[self.gids].copy() if len(self.gids)>0 else None
        print(f"[mask] hand geoms={len(self.gids)}  (bodies substr={self.body_substrs})")

    def set_enabled(self, enabled: bool):
        if len(self.gids) == 0: return
        if enabled:
            self.model.geom_contype[self.gids]     = self._orig_type
            self.model.geom_conaffinity[self.gids] = self._orig_aff
        else:
            self.model.geom_contype[self.gids]     = 0
            self.model.geom_conaffinity[self.gids] = 0
        mj.mj_forward(self.model, self.data)

# ------------------ 查找所有“连线平行 XY”的候选，并随机/最优选一 ------------------
def find_pairs_parallel_xy_all(pairs_b: np.ndarray,
                               lego_R: np.ndarray,
                               max_tilt_deg: float = 90.0,
                               min_seg: float = 0.0,
                               debug: bool = False):
    """
    返回：
      candidates: 满足条件的索引数组（np.int64）
      tilts_deg:  每个候选的倾角（与 XY 的夹角，度）
      best_k:     倾角最小的索引（全体中最优）
    评分：v_w = R*(p2-p1)，s = |v_w.z|/||v_w||， tilt = asin(s)
    """
    v = pairs_b[:,1,:] - pairs_b[:,0,:]        # (K,3) in LEGO local
    seg = np.linalg.norm(v, axis=1)            # (K,)
    if min_seg > 0:
        mask_len = seg >= float(min_seg)
    else:
        mask_len = np.ones(len(seg), dtype=bool)

    v_w = (lego_R @ v.T).T                     # (K,3) in world
    n = np.linalg.norm(v_w, axis=1) + 1e-12
    s = np.abs(v_w[:,2]) / n
    tilts = np.degrees(np.arcsin(np.clip(s, 0.0, 1.0)))  # 与 XY 的倾角（度）

    best_k = int(np.argmin(tilts))  # 全体最优
    mask_tilt = tilts <= float(max_tilt_deg) + 1e-9
    mask = mask_len & mask_tilt
    candidates = np.nonzero(mask)[0].astype(np.int64)

    if debug:
        print(f"[pick] total={len(seg)}  min_seg={min_seg:.6g}  "
              f"cand={len(candidates)}  best_k={best_k}  best_tilt={tilts[best_k]:.3f}°")
    return candidates, tilts, best_k

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser("LEGO 下落（开全局接触、屏蔽手部）→ 找到所有连线平行 XY 的对点 → 随机或最优选一 → 原手部流程 → 恢复接触")
    ap.add_argument("--xml", default="assets/tsinghua_lego.xml")
    ap.add_argument("--lego", default="plate3_3_tan", help="LEGO geom 或 body 名")
    ap.add_argument("--site1", default="if_distal_site_ball")
    ap.add_argument("--site2", default="tf_distal_site_ball")
    ap.add_argument("--freejoint", default="root")
    ap.add_argument("--pairs", default="results/longshort_pairs_fc/pairs_brick2_10_medium_lavender.npy")
    ap.add_argument("--r1", type=float, default=0.004)
    ap.add_argument("--r2", type=float, default=0.0055)
    ap.add_argument("--joints", default="if_proximal_link,th_proximal_link,th_root_link")
    ap.add_argument("--db", default="scan.npy", help="形如 (N,4): [ctrl(3), dist]")
    ap.add_argument("--ramp-steps", type=int, default=300)
    ap.add_argument("--settle-steps", type=int, default=150)
    ap.add_argument("--rot-steps", type=int, default=120)
    ap.add_argument("--drop-steps", type=int, default=200, help="先让 LEGO 自然下落的仿真步数")
    ap.add_argument("--hand-bodies", default="if_,th_", help="认为是手的 body 名子串, 逗号分隔")
    ap.add_argument("--tol", type=float, default=1e-4, help="匹配 L 的容差")
    ap.add_argument("--tol-fallback", type=float, default=5e-4, help="兜底容差（严格找不到时）")
    ap.add_argument("--max-tilt-deg", type=float, default=90.0, help="与 XY 平面夹角允许的上限")
    ap.add_argument("--min-seg", type=float, default=0.0, help="过滤连线长度 < 此值的 pair（0 为不过滤）")
    ap.add_argument("--random-pick", action="store_true", help="在所有候选中随机挑一个（否则取最优）")
    ap.add_argument("--seed", type=int, default=42, help="--random-pick 时的随机种子")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default="align_motor_result.npy")
    ap.add_argument("--debug-pick", action="store_true")
    args = ap.parse_args()

    # ====== MuJoCo 上下文 ======
    model = mj.MjModel.from_xml_path(args.xml)
    data  = mj.MjData(model)
    mj.mj_forward(model, data)

    args.site1 = resolve_site_name(model, args.site1)
    args.site2 = resolve_site_name(model, args.site2)

    # ====== 屏蔽手部接触，开启全局接触 ======
    hand_substrs = [s.strip() for s in args.hand_bodies.split(",") if s.strip()]
    mask = ContactMask(model, data, hand_substrs)
    mask.set_enabled(False)
    model.opt.disableflags = int(model.opt.disableflags) & ~int(mj.mjtDisableBit.mjDSBL_CONTACT)

    # ====== LEGO 自然下落 ======
    for _ in range(max(0, int(args.drop_steps))):
        mj.mj_step(model, data)

    # ====== 读取 LEGO 姿态 & 所有候选 ======
    lego_id, pairs_b = load_pairs_npy(args.pairs)
    lego_p, lego_R = lego_pose(model, data, args.lego)

    cands, tilts, best_k = find_pairs_parallel_xy_all(
        pairs_b, lego_R, max_tilt_deg=args.max_tilt_deg, min_seg=args.min_seg, debug=args.debug_pick
    )

    if len(cands) == 0:
        print(f"[pick] 警告：没有满足 tilt<= {args.max_tilt_deg}° 的候选，改用全体最优 best_k={best_k} (tilt={tilts[best_k]:.3f}°)")
        k = int(best_k)
    else:
        if args.random_pick:
            rng = np.random.default_rng(args.seed)
            k = int(rng.choice(cands))
            print(f"[pick] candidates={len(cands)}  随机选中 k={k}  tilt={tilts[k]:.3f}°  (seed={args.seed})")
        else:
            # 取候选中的最优（倾角最小）
            k = int(cands[np.argmin(tilts[cands])])
            print(f"[pick] candidates={len(cands)}  选择最优 k={k}  tilt={tilts[k]:.3f}°")

    # ====== 目标构造 ======
    p1_b = pairs_b[k,0].astype(np.float64)
    p2_b = pairs_b[k,1].astype(np.float64)
    seg = float(np.linalg.norm(p2_b - p1_b))
    vhat = (p2_b - p1_b) / (seg + 1e-12)
    c1_b = p1_b - args.r1 * vhat
    c2_b = p2_b + args.r2 * vhat
    L = seg + args.r1 + args.r2
    print(f"[target] L = {L:.6f}  (pair_idx={k}, lego_id={lego_id}, tilt_to_XY={tilts[k]:.3f}°)")

    # ====== 选择 ctrl（匹配 L）======
    arr = np.load(args.db, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.ndim==2 and arr.shape[1]>=4:
        ctrls = arr[:,:3].astype(np.float32); dists = arr[:,3].astype(np.float32)
    else:
        tmpc,tmpd = [],[]
        for row in np.atleast_1d(arr):
            if isinstance(row, dict) and "ctrl" in row and "dist" in row:
                c = np.array(row["ctrl"], float).ravel()
                if c.size>=3: tmpc.append(c[:3]); tmpd.append(float(row["dist"]))
        ctrls = np.asarray(tmpc, np.float32); dists = np.asarray(tmpd, np.float32)

    mask_idx = np.where(np.abs(dists - L) < args.tol)[0]
    used_tol = args.tol
    if len(mask_idx) == 0 and args.tol_fallback > args.tol:
        mask_idx = np.where(np.abs(dists - L) < args.tol_fallback)[0]
        used_tol = args.tol_fallback
        if len(mask_idx) > 0:
            print(f"[db] 严格容差未命中，使用兜底 tol={used_tol:g}（命中 {len(mask_idx)}）")

    if len(mask_idx) == 0:
        j = int(np.argmin(np.abs(dists - L)))
        print(f"[db] 未找到 |dist-L| < {args.tol_fallback:g}，改用最接近项 idx={j}, |dist-L|={abs(dists[j]-L):.3e}")
        pick = j
    else:
        pick = int(mask_idx[0])

    ctrl = ctrls[pick]
    print(f"[db] pick={pick}  dist={dists[pick]:.6f}  |dist-L|={abs(dists[pick]-L):.3e}")
    print(f"[db] ctrl={ctrl}")

    # ====== 可视化准备 ======
    def draw_spheres():
        pL, RL = lego_pose(model, data, args.lego)
        p1w = world_from_lego_local(p1_b, pL, RL)
        p2w = world_from_lego_local(p2_b, pL, RL)
        c1w = world_from_lego_local(c1_b, pL, RL)
        c2w = world_from_lego_local(c2_b, pL, RL)
        s1w = get_site_pos(model, data, args.site1)
        s2w = get_site_pos(model, data, args.site2)
        add_sphere(viewer, p1w, 0.0025, (1,1,0,0.9))
        add_sphere(viewer, p2w, 0.0025, (1,1,0,0.9))
        add_sphere(viewer, c1w, args.r1, (0,1,0,0.7))
        add_sphere(viewer, c2w, args.r2, (0,1,0,0.7))
        add_sphere(viewer, s1w, max(1e-4,0.6*args.r1), (1,0,0,0.9))
        add_sphere(viewer, s2w, max(1e-4,0.6*args.r2), (1,0,0,0.9))

    viewer = None
    if args.show:
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
        except Exception as e:
            print("[warn] 打不开 viewer：", e)
            viewer = None

    # ====== 原“手部”动作：ramp → 平移到 c1 → 围绕 c1 旋转到 c2 ======
    joint_names = [s.strip() for s in args.joints.split(",")]
    qadr_free = get_freejoint_qadr(model, args.freejoint)

    c1_w = world_from_lego_local(c1_b, lego_p, lego_R)
    c2_w = world_from_lego_local(c2_b, lego_p, lego_R)

    ramp_motors_to_ctrl(
        model, data, joint_names, ctrl,
        ramp_steps=args.ramp_steps, settle_steps=args.settle_steps,
        zero_gravity=None, disable_contact=None,
        viewer=viewer, draw_fn=(draw_spheres if viewer else None)
    )

    # 平移到 c1
    s1 = get_site_pos(model, data, args.site1)
    base_pos = data.qpos[qadr_free:qadr_free+3].copy()
    base_quat = data.qpos[qadr_free+3:qadr_free+7].copy()
    delta = c1_w - s1
    trans_steps = 60 if viewer is not None else 1
    for i in range(trans_steps):
        a = (i+1)/trans_steps
        p_new = base_pos + a*delta
        data.qpos[qadr_free:qadr_free+3]   = p_new
        data.qpos[qadr_free+3:qadr_free+7] = base_quat
        mj.mj_forward(model, data)
        if viewer is not None:
            viewer.user_scn.ngeom = 0; draw_spheres(); viewer.sync(); time.sleep(1/120)
    base_pos = data.qpos[qadr_free:qadr_free+3].copy()

    # 围绕 c1 旋转到 c2
    s2a = get_site_pos(model, data, args.site2)
    v = s2a - c1_w; w = c2_w - c1_w
    vn = v/(np.linalg.norm(v)+1e-12); wn = w/(np.linalg.norm(w)+1e-12)
    axis = np.cross(vn, wn); l = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(vn, wn), -1.0, 1.0))
    ang = float(np.arctan2(l, dot))
    if l < 1e-9:
        tmp = np.array([1,0,0], np.float64)
        if abs(np.dot(tmp, vn)) > 0.9: tmp = np.array([0,1,0], np.float64)
        axis = np.cross(vn, tmp); axis /= (np.linalg.norm(axis)+1e-12)
    rot_steps = max(1, args.rot_steps)
    dR = rodrigues(axis, ang/rot_steps)
    for _ in range(rot_steps):
        base_pos, base_quat = set_freejoint_pose_about_point(
            model, data, qadr_free, base_pos, base_quat, dR, c1_w
        )
        if viewer is not None:
            viewer.user_scn.ngeom = 0; draw_spheres(); viewer.sync(); time.sleep(1/120)

    # ====== 恢复手部的碰撞 ======
    mask.set_enabled(True)

    # ====== 评估 ======
    s1f = get_site_pos(model, data, args.site1)
    s2f = get_site_pos(model, data, args.site2)
    e1 = float(np.linalg.norm(s1f - c1_w))
    e2 = float(np.linalg.norm(s2f - c2_w))
    print(f"[align] errors (m): site1={e1:.6e}, site2={e2:.6e}")

    out = {
        "lego_id": lego_id,
        "pair_index": int(k),
        "tilt_to_XY_deg": float(tilts[k]),
        "radii": {"r1": args.r1, "r2": args.r2},
        "green_distance": L,
        "picked_ctrl": ctrl.tolist(),
        "targets_world": {"c1": c1_w.tolist(), "c2": c2_w.tolist()},
        "sites_world_final": {"s1": s1f.tolist(), "s2": s2f.tolist()},
        "errors": {"e1": e1, "e2": e2},
        "base_pose": {"pos": base_pos.tolist(), "quat_xyzw": base_quat.tolist()},
        "pick_stats": {
            "num_candidates": int(len(cands)),
            "max_tilt_deg": float(args.max_tilt_deg),
            "min_seg": float(args.min_seg),
            "random_pick": bool(args.random_pick),
            "seed": int(args.seed) if args.random_pick else None
        }
    }
    np.save(args.save, out, allow_pickle=True)
    print(f"[done] 保存: {args.save}")

    if viewer is not None:
        print("[info] 按 Ctrl+C 手动结束程序，窗口会保持")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()
