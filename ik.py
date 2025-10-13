#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse, numpy as np
import mujoco as mj
from dm_control import mujoco as dm_mj

# ---------------- basic utils ----------------
def load_pairs_npy(path: str):
    payload = np.load(path, allow_pickle=True).item()
    pairs = np.asarray(payload["pairs"], np.float32)          # (K,2,3)
    lego_id = payload.get("lego_id", os.path.splitext(os.path.basename(path))[0])
    return lego_id, pairs

def resolve_site_name(physics, name: str):
    sid = physics.model.name2id(name, "site")
    if sid >= 0: return name
    if "_ball" in name:
        alt = name.replace("_ball", "_a")
        if physics.model.name2id(alt, "site") >= 0:
            print(f"[info] site '{name}' 不存在，自动改用 '{alt}'")
            return alt
    raise RuntimeError(f"site '{name}' not found")

def body_pose_from_geom(physics, geom_or_body: str):
    # 允许传 geom 或 body 名
    gid = physics.model.name2id(geom_or_body, "geom")
    if gid < 0:
        bid = physics.model.name2id(geom_or_body, "body")
        if bid < 0:
            raise RuntimeError(f"geom/body '{geom_or_body}' not found")
    else:
        bid = int(physics.model.geom_bodyid[gid])
    R = physics.data.xmat[bid].reshape(3,3).copy()
    p = physics.data.xpos[bid].copy()
    return p, R, bid

def world_from_lego_local(p_local, lego_p_w, lego_R_w):
    return lego_p_w + lego_R_w @ p_local

def get_site_pos(physics, site_name: str):
    return physics.named.data.site_xpos[site_name].copy()

def mat_to_quat_xyzw(R):
    q = np.empty(4, dtype=np.float64)
    mj.mju_mat2Quat(q, R.reshape(9))
    return q

def rodrigues(axis, angle):
    a = np.asarray(axis, np.float64); n = np.linalg.norm(a)
    if n < 1e-12: return np.eye(3)
    a = a / n
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]], np.float64)
    c, s = np.cos(angle), np.sin(angle)
    return np.eye(3)+s*K+(1-c)*(K@K)

# ---------------- motor helpers ----------------
def _jid(model, name):
    j = model.name2id(name, "joint")
    if j < 0: raise RuntimeError(f"joint '{name}' not found")
    return int(j)

def _aid_for_joint(model, joint_id):
    # 通过 TRNID 映射到 actuator，避免依赖命名（XML里 th_root_link 的执行器名就是关节名）
    for ai in range(model.nu):
        if int(model.actuator_trnid[ai,0]) == int(joint_id):
            return int(ai)
    return -1

def _hold_other_position_actuators(model, data, exclude):
    ex = set(exclude)
    for ai in range(model.nu):
        if ai in ex: continue
        j = int(model.actuator_trnid[ai,0])
        if j >= 0 and model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            qadr = int(model.jnt_qposadr[j])
            data.ctrl[ai] = float(data.qpos[qadr])

def ramp_motors_to_ctrl(physics, joint_names, target_ctrl, ramp_steps=300, settle_steps=200):
    """分阶段插值，把关节逐渐拉到目标角度"""
    model, data = physics.model, physics.data
    jids = [_jid(model, n) for n in joint_names]
    aids = [_aid_for_joint(model, j) for j in jids]
    if any(a<0 for a in aids):
        miss = [joint_names[i] for i,a in enumerate(aids) if a<0]
        raise RuntimeError(f"未找到执行器: {miss}")
    qaddrs = [int(model.jnt_qposadr[j]) for j in jids]
    cur = np.array([float(data.qpos[q]) for q in qaddrs], np.float64)
    tgt = np.asarray(target_ctrl, np.float64).ravel()[:len(joint_names)]

    for t in range(ramp_steps):
        a = (t+1)/float(ramp_steps)
        cmd = (1-a)*cur + a*tgt
        for k, ai in enumerate(aids): data.ctrl[ai] = float(cmd[k])
        _hold_other_position_actuators(model, data, aids)
        data.qvel[:] = 0.0; data.qacc[:] = 0.0
        mj.mj_step(model.ptr, data.ptr)
    for _ in range(settle_steps):
        _hold_other_position_actuators(model, data, aids)
        mj.mj_step(model.ptr, data.ptr)

def servo_to(physics, joint_names, ctrl, settle=120):
    """直接给定 motor ctrl，让系统收敛到目标"""
    model, data = physics.model, physics.data
    jids = [_jid(model, n) for n in joint_names]
    aids = [_aid_for_joint(model, j) for j in jids]
    for k, ai in enumerate(aids):
        if ai >= 0: data.ctrl[ai] = float(ctrl[k])
    for _ in range(settle):
        _hold_other_position_actuators(model, data, aids)
        mj.mj_step(model.ptr, data.ptr)

def motor_drive_to_ctrl(physics, joint_names, target_ctrl, max_steps=3000, tol=1e-3):
    """
    用 actuator 索引（非名字）驱动到目标角度；命名如何都不怕
    """
    model, data = physics.model, physics.data
    target_ctrl = np.asarray(target_ctrl, float).ravel()[:len(joint_names)]
    jids  = [_jid(model, n) for n in joint_names]
    aids  = [_aid_for_joint(model, j) for j in jids]
    qaddr = [int(model.jnt_qposadr[j]) for j in jids]
    if any(a<0 for a in aids):
        miss = [joint_names[i] for i,a in enumerate(aids) if a<0]
        raise RuntimeError(f"未找到执行器: {miss}")

    for step in range(max_steps):
        for k, ai in enumerate(aids):
            data.ctrl[ai] = float(target_ctrl[k])
        _hold_other_position_actuators(model, data, aids)
        mj.mj_step(model.ptr, data.ptr)

        qpos_now = np.array([data.qpos[q] for q in qaddr], np.float64)
        err = float(np.max(np.abs(qpos_now - target_ctrl)))
        if err < tol:
            print(f"[motor_drive] 收敛成功 step={step}, err={err:.2e}")
            return True
    print(f"[motor_drive] 未收敛")
    return False

def measure_dist(physics, site1, site2):
    s1 = get_site_pos(physics, site1)
    s2 = get_site_pos(physics, site2)
    return s1, s2, float(np.linalg.norm(s2 - s1))

def refine_ctrl_to_match_distance(physics, joint_names, ctrl0, target_L,
                                  site1, site2,
                                  tol=1e-5, iters=8, eps=1e-3, step_clip=0.2):
    """有限差分，把 |‖s2-s1‖ - target_L| 调到 tol 以内"""
    q = np.asarray(ctrl0, np.float64).copy()
    servo_to(physics, joint_names, q, settle=150)

    for _ in range(iters):
        _, _, L = measure_dist(physics, site1, site2)
        err = L - target_L
        if abs(err) < tol: break

        g = np.zeros(len(joint_names), np.float64)
        for i in range(len(joint_names)):
            q_pert = q.copy(); q_pert[i] += eps
            servo_to(physics, joint_names, q_pert, settle=80)
            _, _, Lp = measure_dist(physics, site1, site2)
            g[i] = (Lp - L) / eps

        gn = float(np.dot(g, g))
        if gn < 1e-12: break
        alpha = np.clip(-err/gn, -step_clip, step_clip)
        q = q + alpha * g
        servo_to(physics, joint_names, q, settle=120)

    _, _, Lf = measure_dist(physics, site1, site2)
    return q, Lf

# ---------------- viewer ----------------
def add_sphere_to_userscn(viewer, pos, radius, rgba):
    scn = viewer.user_scn
    g = scn.geoms[scn.ngeom]
    size = np.array([radius, radius, radius], dtype=np.float32)
    mat = np.eye(3, dtype=np.float32).reshape(-1)
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    scn.ngeom += 1

def _lego_pose_in_viewer(model, data, name):
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:  bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    else:       bid = int(model.geom_bodyid[gid])
    p = data.xpos[bid].copy(); R = data.xmat[bid].reshape(3,3).copy()
    return p, R

def launch_viewer(xml_path, lego_name, base_freejoint, joint_names, ctrl, base_pos, base_quat,
                  site1, site2, p1_b, p2_b, c1_b, c2_b, r1, r2):
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    # joints
    for jn,v in zip(joint_names, ctrl):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn)
        qadr = int(model.jnt_qposadr[jid]); data.qpos[qadr] = float(v)
    # base
    if base_freejoint:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, base_freejoint)
        if jid >= 0 and model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr:qadr+3] = base_pos
            data.qpos[qadr+3:qadr+7] = base_quat
    mj.mj_forward(model, data)

    sid1 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site1)
    sid2 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site2)

    try:
        import mujoco.viewer as mjv
    except Exception as e:
        print("[warn] viewer skip:", e); return

    with mjv.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            lego_p_w, lego_R_w = _lego_pose_in_viewer(model, data, lego_name)
            p1_w = world_from_lego_local(p1_b, lego_p_w, lego_R_w)
            p2_w = world_from_lego_local(p2_b, lego_p_w, lego_R_w)
            c1_w = world_from_lego_local(c1_b, lego_p_w, lego_R_w)
            c2_w = world_from_lego_local(c2_b, lego_p_w, lego_R_w)
            s1_w = data.site_xpos[sid1].copy(); s2_w = data.site_xpos[sid2].copy()

            # 黄 = 原始pair点；绿 = 外推球；红 = 当前site
            add_sphere_to_userscn(viewer, p1_w, 0.0025, (1,1,0,0.9))
            add_sphere_to_userscn(viewer, p2_w, 0.0025, (1,1,0,0.9))
            add_sphere_to_userscn(viewer, c1_w, r1, (0,1,0,0.7))
            add_sphere_to_userscn(viewer, c2_w, r2, (0,1,0,0.7))
            add_sphere_to_userscn(viewer, s1_w, max(1e-4,0.6*r1), (1,0,0,0.9))
            add_sphere_to_userscn(viewer, s2_w, max(1e-4,0.6*r2), (1,0,0,0.9))
            viewer.sync(); time.sleep(1/120)

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser("DB→motors→refine→s1->c1→rotate about c1 align s2")
    parser.add_argument("--xml", default="assets/tsinghua_lego.xml")
    parser.add_argument("--lego-geom", default="plate3_3_tan")
    parser.add_argument("--site1", default="if_distal_site_ball")
    parser.add_argument("--site2", default="tf_distal_site_ball")
    parser.add_argument("--hand-freejoint", default="root")
    parser.add_argument("--pairs", default="results/longshort_pairs_fc/pairs_brick2_10_medium_lavender.npy")
    parser.add_argument("--pairs-index", type=int, default=0)
    parser.add_argument("--r1", type=float, default=0.004)
    parser.add_argument("--r2", type=float, default=0.0055)
    parser.add_argument("--joints", default="if_proximal_link,th_proximal_link,th_root_link")
    parser.add_argument("--db", default="scan100_servo.npy")
    parser.add_argument("--drive-mode", choices=["ramp","drive","servo"], default="ramp")
    parser.add_argument("--ramp-steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--rot-steps", type=int, default=120)
    parser.add_argument("--offset-mode", choices=["antipodal","normal"], default="antipodal")
    parser.add_argument("--flip-antipodal", action="store_true", help="反向外推向量")
    parser.add_argument("--save", default="align_motor_result.npy")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # 1) 读 pair，并外推 c1/c2
    lego_id, pairs = load_pairs_npy(args.pairs)
    idx = int(np.clip(args.pairs_index, 0, pairs.shape[0]-1))
    p1_b = pairs[idx,0].astype(np.float64)
    p2_b = pairs[idx,1].astype(np.float64)

    if args.offset_mode == "antipodal":
        vhat = p2_b - p1_b; vhat /= (np.linalg.norm(vhat)+1e-12)
        if args.flip_antipodal: vhat = -vhat
        c1_b = p1_b - args.r1 * vhat
        c2_b = p2_b + args.r2 * vhat
    else:
        # normal 模式需要 trimesh
        try:
            import trimesh
        except Exception as e:
            raise RuntimeError("offset-mode=normal 需要安装 trimesh") from e
        mesh = trimesh.load("assets/object/brick2_10_medium_lavender.obj", force='mesh')
        if isinstance(mesh, trimesh.Scene): mesh = mesh.dump().sum()

        def nearest_normal(mesh, p):
            p = np.asarray(p, dtype=np.float64).reshape(1, 3)
            try:
                ret = mesh.nearest.on_surface(p)
            except AttributeError:
                ret = None
            face_idx = None
            if isinstance(ret, (tuple, list)):
                if len(ret) == 3: _, _, face_idx = ret
                elif len(ret) == 2: _, face_idx = ret
            if face_idx is None or len(face_idx) == 0 or int(face_idx[0]) < 0:
                _, _, tri_idx = trimesh.proximity.closest_point(mesh, p)
                face_idx = tri_idx
            n = mesh.face_normals[int(face_idx[0])].astype(np.float64)
            n /= (np.linalg.norm(n) + 1e-12)
            # 朝外：与包围盒中心相反
            center = mesh.bounds.mean(axis=0)
            if np.dot(p.reshape(-1)-center, n) < 0: n = -n
            return n

        n1_b = nearest_normal(mesh, p1_b)
        n2_b = nearest_normal(mesh, p2_b)
        c1_b = p1_b + args.r1 * n1_b
        c2_b = p2_b + args.r2 * n2_b

    # 2) 目标距离
    physics = dm_mj.Physics.from_xml_path(args.xml)
    args.site1 = resolve_site_name(physics, args.site1)
    args.site2 = resolve_site_name(physics, args.site2)

    lego_p_w, lego_R_w, _ = body_pose_from_geom(physics, args.lego_geom)
    c1_w = world_from_lego_local(c1_b, lego_p_w, lego_R_w)
    c2_w = world_from_lego_local(c2_b, lego_p_w, lego_R_w)
    target_L = float(np.linalg.norm(c2_w - c1_w))

    # 3) 库中最近距离 → 电机执行
    arr = np.load(args.db, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.ndim==2 and arr.shape[1]>=4:
        ctrls, dists = arr[:,:3].astype(np.float64), arr[:,3].astype(np.float64)
    else:
        tmpc, tmpd = [], []
        for row in np.atleast_1d(arr):
            if isinstance(row, dict) and "ctrl" in row and "dist" in row:
                c = np.array(row["ctrl"], float).ravel()
                if c.size>=3: tmpc.append(c[:3]); tmpd.append(float(row["dist"]))
        ctrls, dists = np.asarray(tmpc,np.float64), np.asarray(tmpd,np.float64)
    pick = int(np.argmin(np.abs(dists - target_L)))
    q = ctrls[pick].copy()
    joint_names = [s.strip() for s in args.joints.split(",")]

    if args.drive_mode == "ramp":
        ramp_motors_to_ctrl(physics, joint_names, q, args.ramp_steps, args.settle_steps)
    elif args.drive_mode == "drive":
        motor_drive_to_ctrl(physics, joint_names, q)
    else:
        servo_to(physics, joint_names, q, settle=300)

    _, _, L_after = measure_dist(physics, args.site1, args.site2)

    # 4) 精调距离
    q_refined, L_refined = refine_ctrl_to_match_distance(
        physics, joint_names, q, target_L, args.site1, args.site2
    )
    print(f"[dist] target={target_L:.6f}  after={L_after:.6f}  refined={L_refined:.6f}")

    # 5) 平移 s1→c1 ，绕 c1 旋转使 (s2-c1)//(c2-c1)
    jid = physics.model.name2id(args.hand_freejoint, "joint")
    if jid < 0 or physics.model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"freejoint '{args.hand_freejoint}' 不存在或不是 freejoint")
    qadr = int(physics.model.jnt_qposadr[jid])

    s1, s2, _ = measure_dist(physics, args.site1, args.site2)
    base_pos = physics.data.qpos[qadr:qadr+3].copy()
    base_quat = physics.data.qpos[qadr+3:qadr+7].copy()
    # 先把 s1 平移到 c1
    base_pos = base_pos + (c1_w - s1)
    physics.data.qpos[qadr:qadr+3] = base_pos
    physics.data.qpos[qadr+3:qadr+7] = base_quat
    mj.mj_forward(physics.model.ptr, physics.data.ptr)

    # 再绕 c1 旋转，使方向对齐
    s1a = get_site_pos(physics, args.site1); s2a = get_site_pos(physics, args.site2)
    v = s2a - c1_w; w = c2_w - c1_w
    vn = v/(np.linalg.norm(v)+1e-12); wn = w/(np.linalg.norm(w)+1e-12)
    axis = np.cross(vn, wn); l = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(vn, wn), -1.0, 1.0))
    ang = float(np.arctan2(l, dot))
    axis = (axis/l) if l>1e-9 else np.array([1.0,0,0])

    steps = max(1, args.rot_steps)
    for i in range(steps):
        a = ang*(i+1)/float(steps)
        Rw = rodrigues(axis, a)
        q_rot = mat_to_quat_xyzw(Rw)
        q_new = np.empty(4); mj.mju_mulQuat(q_new, q_rot, physics.data.qpos[qadr+3:qadr+7])
        p0 = c1_w; p  = physics.data.qpos[qadr:qadr+3].copy()
        p_new = p0 + Rw @ (p - p0)
        physics.data.qpos[qadr:qadr+3]   = p_new
        physics.data.qpos[qadr+3:qadr+7] = q_new
        mj.mj_forward(physics.model.ptr, physics.data.ptr)

    # （可选）沿 c1c2 方向做一维微调，消掉数值误差
    s1f = get_site_pos(physics, args.site1); s2f = get_site_pos(physics, args.site2)
    w = c2_w - c1_w; wn = w/(np.linalg.norm(w)+1e-12)
    delta = (c2_w - s2f)
    shift = np.dot(delta, wn) * wn
    physics.data.qpos[qadr:qadr+3] += shift
    mj.mj_forward(physics.model.ptr, physics.data.ptr)

    # 误差统计
    s1f = get_site_pos(physics, args.site1); s2f = get_site_pos(physics, args.site2)
    e1 = float(np.linalg.norm(s1f - c1_w))
    e2 = float(np.linalg.norm(s2f - c2_w))
    print(f"[align] err: site1={e1:.6e}, site2={e2:.6e}")

    # 6) 保存
    out = dict(
        pairs_index=idx, lego_id=lego_id,
        ctrl_pick=q, ctrl_refined=q_refined,
        target_dist=target_L, after=L_after, refined=L_refined,
        c1_w=c1_w, c2_w=c2_w, s1_w=s1f, s2_w=s2f,
        err1=e1, err2=e2,
        drive_mode=args.drive_mode, offset_mode=args.offset_mode, flip_antipodal=bool(args.flip_antipodal),
    )
    np.save(args.save, out, allow_pickle=True)
    print(f"[save] {args.save}")

    # 7) 可视化
    if args.show:
        launch_viewer(args.xml, args.lego_geom, args.hand_freejoint,
                      joint_names, q_refined,
                      physics.data.qpos[qadr:qadr+3].copy(),
                      physics.data.qpos[qadr+3:qadr+7].copy(),
                      args.site1, args.site2,
                      p1_b, p2_b, c1_b, c2_b, args.r1, args.r2)

if __name__ == "__main__":
    main()
