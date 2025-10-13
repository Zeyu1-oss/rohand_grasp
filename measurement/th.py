#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, argparse, numpy as np
import mujoco as mj

def _joint_id(model, name):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"joint '{name}' not found")
    return jid

def _act_id_for_joint(model, joint_id):
    """返回第一个映射到 joint 的 actuator id"""
    for ai in range(model.nu):
        if model.actuator_trnid[ai, 0] == joint_id:
            return ai
    return -1

def measure_angles(xml_path,
                   prox_joint="th_proximal_link",
                   dist_joint="th_distal_link",
                   start=0.0, end=-1.57, steps=100,
                   settle_steps=1000, extra_wait=200,
                   csv_path="th_relation.csv"):
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    jid_prox = _joint_id(model, prox_joint)
    jid_dist = _joint_id(model, dist_joint)
    adr_prox = model.jnt_qposadr[jid_prox]
    adr_dist = model.jnt_qposadr[jid_dist]

    aid_prox = _act_id_for_joint(model, jid_prox)
    if aid_prox < 0:
        raise RuntimeError(f"no actuator controlling joint '{prox_joint}'")

    # 其它 actuator 固定
    joint_qadr = model.jnt_qposadr.copy()
    other_act = []
    for ai in range(model.nu):
        if ai == aid_prox: 
            continue
        j = model.actuator_trnid[ai, 0]
        if j >= 0 and model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            other_act.append((ai, joint_qadr[j]))

    print("prox_angle (rad), distal_angle (rad)")
    rows = [("prox_angle", "distal_angle")]
    
    thetas = np.linspace(start, end, steps)

    for theta_cmd in thetas:
        data.ctrl[:] = 0.0
        data.ctrl[aid_prox] = theta_cmd
        for ai, qadr in other_act:
            data.ctrl[ai] = float(data.qpos[qadr])

        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        for _ in range(settle_steps):
            mj.mj_step(model, data)

        # 额外等待稳定
        last_prox, last_dist = None, None
        for _ in range(extra_wait):
            mj.mj_step(model, data)
            prox_now = float(data.qpos[adr_prox])
            dist_now = float(data.qpos[adr_dist])
            if last_prox is not None and abs(prox_now-last_prox)<1e-6 and abs(dist_now-last_dist)<1e-6:
                break
            last_prox, last_dist = prox_now, dist_now

        prox_angle = float(data.qpos[adr_prox])
        dist_angle = float(data.qpos[adr_dist])
        print(f"{prox_angle:.6f}, {dist_angle:.6f}")
        rows.append((prox_angle, dist_angle))

    if csv_path:
        d = os.path.dirname(csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
        print(f"\n已保存到 {csv_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml",   default="assets/left_hand.xml")
    ap.add_argument("--prox",  default="th_proximal_link")
    ap.add_argument("--dist",  default="th_distal_link")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end",   type=float, default=-1.57)
    ap.add_argument("--steps", type=int,   default=1000)
    ap.add_argument("--settle",type=int,   default=500, help="收敛步数")
    ap.add_argument("--extra", type=int,   default=200, help="额外稳定等待步数")
    ap.add_argument("--csv",   type=str,   default="th_relation.csv", help="CSV 文件路径")
    args = ap.parse_args()

    measure_angles(args.xml, args.prox, args.dist,
                   start=args.start, end=args.end, steps=args.steps,
                   settle_steps=args.settle, extra_wait=args.extra,
                   csv_path=args.csv)
