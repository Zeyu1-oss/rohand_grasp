#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AABB All-Faces Sampler + Force-Closure (Ferrari-Canny ε)
- 在 AABB 的 6 个对置面上均匀格点采样射线
- 每条射线取【首命中、末命中】两点作为对向点对（LEGO 局部坐标系）
- 过滤：首点法向需与该面外法向一致、末点与对面外法向一致（阈值由 --aabb_normal_align_deg 控制）
- 计算 Ferrari–Canny ε（摩擦锥离散 + Batched Frank-Wolfe）后按阈值筛
- 保存丰富信息：
  - pairs(K,2,3)、normals(K,2,3)
  - face_ids(K,2) ∈ {0:x-,1:x+,2:y-,3:y+,4:z-,5:z+}
  - pair_axis(K,) ∈ {0:x,1:y,2:z}
  - face_uv: {p1(K,2), p2(K,2)}  // 点在各自 AABB 面内的 2D 坐标（[0,1]）
  - epsilon(K,), outward(K,3)（首面对外法向）
  - meta: scale、aabb{min,max}、参数记录
"""

import os
import json
from pathlib import Path
from glob import glob
from typing import Optional, List, Tuple

import numpy as np
import trimesh
from tqdm import tqdm

import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def _should_skip_file(out_path: str) -> bool:
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

def _safe_load_mesh(mesh_path: str) -> trimesh.Trimesh:
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        return mesh
    except Exception as e1:
        try:
            mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump().sum()
            return mesh
        except Exception as e2:
            raise RuntimeError(f"加载失败: {mesh_path}\n 1st: {e1}\n 2nd: {e2}")

def _convexify_mesh(mesh: trimesh.Trimesh, mode: str = "hull") -> trimesh.Trimesh:
    mode = str(mode).lower()
    if mode == "none":
        return mesh
    if mode == "vhacd":
        try:
            from trimesh.interfaces.vhacd import convex_decomposition
            parts = convex_decomposition(mesh)
            if isinstance(parts, (list, tuple)) and len(parts) > 0:
                mesh = trimesh.util.concatenate(parts)
                return mesh.convex_hull
        except Exception:
            pass
    return mesh.convex_hull

def _get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def _to_t(x, device, dtype=torch.float32):
    if isinstance(x, np.ndarray) and (not x.flags.writeable):
        x = x.copy()
    return torch.tensor(x, device=device, dtype=dtype)

def _np(x: torch.Tensor):
    return x.detach().cpu().numpy()


# --------------------- Ferrari–Canny ε 评估 ---------------------
class ForceClosureEps:
    def __init__(self, mu=0.4, m_dirs=8, device=None, dtype=torch.float32, eps_max_iters=40, eps_tol=1e-6):
        self.mu = float(mu)
        self.m_dirs = int(m_dirs)
        self.device = _get_device(device)
        self.dtype = dtype
        self._phi = _to_t(2 * np.pi * np.arange(self.m_dirs) / self.m_dirs, self.device, self.dtype)
        self.eps_max_iters = int(eps_max_iters)
        self.eps_tol = float(eps_tol)

    def _unit(self, v: torch.Tensor, eps=1e-12) -> torch.Tensor:
        n = torch.linalg.norm(v, dim=-1, keepdim=True)
        return torch.where(n < eps, v, v / (n + 1e-12))

    def _tangent_basis_batch(self, axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        axis = self._unit(axis)
        up1 = torch.tensor([0., 0., 1.], device=self.device, dtype=self.dtype).expand_as(axis)
        up2 = torch.tensor([0., 1., 0.], device=self.device, dtype=self.dtype).expand_as(axis)
        mask = (torch.abs(torch.sum(axis * up1, dim=-1)) >= 0.95).unsqueeze(-1)
        up = torch.where(mask, up2, up1)
        t1 = self._unit(torch.linalg.cross(axis, up, dim=-1))
        t2 = self._unit(torch.linalg.cross(axis, t1, dim=-1))
        return t1, t2

    # p/n/com: (B,3) -> Wp: (B,6,m_dirs)
    def contact_wrench_rays_batch(self, p: torch.Tensor, n: torch.Tensor, com: torch.Tensor) -> torch.Tensor:
        t1, t2 = self._tangent_basis_batch(n)
        r = (p - com).unsqueeze(1)
        c = torch.cos(self._phi).view(1, self.m_dirs, 1)
        s = torch.sin(self._phi).view(1, self.m_dirs, 1)
        t = c * t1.unsqueeze(1) + s * t2.unsqueeze(1)
        f = -n.unsqueeze(1) + self.mu * t
        f = f / (torch.linalg.norm(f, dim=-1, keepdim=True) + 1e-12)
        tau = torch.linalg.cross(r.expand_as(f), f, dim=-1)
        Wp = torch.cat([f, tau], dim=-1)
        return Wp.permute(0, 2, 1).contiguous()

    # Batched Frank–Wolfe，返回 eps: (B,)
    def epsilon_fw_batched(self, W: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            B, _, M = W.shape
            col_norm2 = (W * W).sum(dim=1)
            j = torch.argmin(col_norm2, dim=1)                         # init: 最短列
            lam = torch.zeros(B, M, device=self.device, dtype=self.dtype)
            lam.scatter_(1, j.view(B, 1), 1.0)
            x = W.gather(2, j.view(B, 1, 1).expand(B, 6, 1)).squeeze(2)
            prev_obj = 0.5 * (x * x).sum(dim=1)
            for _ in range(self.eps_max_iters):
                g = torch.bmm(W.transpose(1, 2), x.unsqueeze(2)).squeeze(2)  # (B,M)
                j = torch.argmin(g, dim=1)
                wj = W.gather(2, j.view(B, 1, 1).expand(B, 6, 1)).squeeze(2)
                d = wj - x
                denom = (d * d).sum(dim=1)
                num = (x * (x - wj)).sum(dim=1)
                gamma = torch.where(denom > 1e-18, torch.clamp(num / denom, 0.0, 1.0), torch.zeros_like(denom))
                x = x + gamma.unsqueeze(1) * d
                lam = lam * (1.0 - gamma).unsqueeze(1)
                lam.scatter_add_(1, j.view(B, 1), gamma.view(B, 1))
                obj = 0.5 * (x * x).sum(dim=1)
                rel = torch.abs(obj - prev_obj) / torch.clamp(prev_obj, min=1.0)
                if torch.max(rel).item() <= self.eps_tol:
                    break
                prev_obj = obj
            return torch.linalg.norm(x, dim=1)


# --------------------- 采样（六个对置面） ---------------------
class AllFacesSampler:
    def __init__(self,
                 aabb_margin_ratio: float = 0.0,
                 face_jitter: float = 0.0,
                 rand_rotate: bool = False,
                 convexify: str = "hull",
                 seed: int = 42):
        self.aabb_margin_ratio = float(aabb_margin_ratio)
        self.face_jitter = float(face_jitter)
        self.rand_rotate = bool(rand_rotate)
        self.convexify = str(convexify)
        self.rng = np.random.default_rng(seed)

    def gen_rays(self, mesh: trimesh.Trimesh, total_rays: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        返回：
          origins(R,3), dirs(R,3), face_ids(R,), outward(R,3), bb_min(3), bb_max(3)
        其中 face_ids ∈ {0:x-,1:x+,2:y-,3:y+,4:z-,5:z+}
        """
        if self.rand_rotate:
            R = trimesh.transformations.random_rotation_matrix()[:3, :3]
            c = mesh.centroid
            V = np.asarray(mesh.vertices) - c
            V_rot = (V @ R.T) + c
            bb_min = V_rot.min(axis=0).astype(float)
            bb_max = V_rot.max(axis=0).astype(float)

            def invR_vec(v): return v @ R
            def invR_pt(p):  return (p - c) @ R + c
        else:
            bb_min = mesh.bounds[0].astype(float)
            bb_max = mesh.bounds[1].astype(float)
            def invR_vec(v): return v
            def invR_pt(p):  return p

        ext = bb_max - bb_min
        margin = self.aabb_margin_ratio * (np.linalg.norm(ext) + 1e-12)
        bb_min -= margin; bb_max += margin
        ext = bb_max - bb_min

        # 六个面：0:x-,1:x+,2:y-,3:y+,4:z-,5:z+；inward 指向 AABB 内部
        faces = [
            (np.array([bb_min[0], 0, 0]),  np.array([ 1, 0, 0]), np.array([0,1,0]), np.array([0,0,1]), (ext[1], ext[2])),  # 0 x- inward +x
            (np.array([bb_max[0], 0, 0]),  np.array([-1, 0, 0]), np.array([0,1,0]), np.array([0,0,1]), (ext[1], ext[2])),  # 1 x+ inward -x
            (np.array([0, bb_min[1], 0]),  np.array([ 0, 1, 0]), np.array([1,0,0]), np.array([0,0,1]), (ext[0], ext[2])),  # 2 y- inward +y
            (np.array([0, bb_max[1], 0]),  np.array([ 0,-1, 0]), np.array([1,0,0]), np.array([0,0,1]), (ext[0], ext[2])),  # 3 y+ inward -y
            (np.array([0, 0, bb_min[2]]),  np.array([ 0, 0, 1]), np.array([1,0,0]), np.array([0,1,0]), (ext[0], ext[1])),  # 4 z- inward +z
            (np.array([0, 0, bb_max[2]]),  np.array([ 0, 0,-1]), np.array([1,0,0]), np.array([0,1,0]), (ext[0], ext[1])),  # 5 z+ inward -z
        ]

        def make_anchor(i, anc):
            if i in (0, 1):   return np.array([anc[0], bb_min[1], bb_min[2]])
            if i in (2, 3):   return np.array([bb_min[0], anc[1], bb_min[2]])
            return np.array([bb_min[0], bb_min[1], anc[2]])

        selected_faces = [0,1,2,3,4,5]
        nfaces = len(selected_faces)
        per_face = max(1, int(total_rays // nfaces))
        eff_total = per_face * nfaces
        if eff_total < total_rays:
            print(f"[AABB-ALL] 调整总射线数为 {eff_total}（原 {total_rays}）")

        origins, dirs, face_ids, outward = [], [], [], []
        for i_face in selected_faces:
            anc, inward, u_axis, v_axis, (W, H) = faces[i_face]
            inward = inward / (np.linalg.norm(inward) + 1e-12)
            outward_face = -inward  # AABB 外法向
            n = int(np.ceil(np.sqrt(per_face)))
            m = int(np.ceil(per_face / max(n, 1)))
            du = W / max(n, 1)
            dv = H / max(m, 1)
            anchor = make_anchor(i_face, anc)

            idxs = [(iu, iv) for iu in range(n) for iv in range(m)]
            self.rng.shuffle(idxs)

            cnt = 0
            for (iu, iv) in idxs:
                if cnt >= per_face: break
                u01 = (iu + 0.5) / max(n, 1)
                v01 = (iv + 0.5) / max(m, 1)
                o_rot = anchor + u01 * W * u_axis + v01 * H * v_axis
                if self.face_jitter > 0:
                    j = self.face_jitter
                    o_rot = o_rot + (self.rng.random(3) * 2 - 1) * j * max(du, dv)
                o = invR_pt(o_rot)
                d = invR_vec(inward)
                d = d / (np.linalg.norm(d) + 1e-12)
                n_out = invR_vec(outward_face)
                origins.append(o)
                dirs.append(d)
                face_ids.append(i_face)
                outward.append(n_out)
                cnt += 1

        return (np.asarray(origins, float),
                np.asarray(dirs, float),
                np.asarray(face_ids, int),
                np.asarray(outward, float),
                bb_min.astype(float), bb_max.astype(float))

    @staticmethod
    def first_last_hits(origins: np.ndarray, dirs: np.ndarray, mesh: trimesh.Trimesh):
        ray_engine = (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
                      if getattr(trimesh.ray, "has_embree", False)
                      else trimesh.ray.ray_triangle.RayMeshIntersector(mesh))
        locs, ray_ids, tri_idx = ray_engine.intersects_location(origins, dirs, multiple_hits=True)
        if len(locs) == 0:
            return (np.empty((0,3)), np.empty((0,3)), np.empty((0,3)), np.empty((0,3)), np.empty((0,), int), np.empty((0,), int))
        order = np.argsort(ray_ids, kind='mergesort')
        locs = locs[order]; ray_ids = ray_ids[order]; tri_idx = tri_idx[order]
        dists = np.linalg.norm(locs - origins[ray_ids], axis=1)
        first_idx=[]; last_idx=[]; first_ray=[]; first_tri=[]
        i=0; N=len(locs)
        while i<N:
            j=i; rid=ray_ids[i]
            while j<N and ray_ids[j]==rid: j+=1
            seg = slice(i,j)
            idx = np.argsort(dists[seg])
            fi = i + idx[0]      # 最近命中
            li = i + idx[-1]     # 最远命中
            first_idx.append(fi); last_idx.append(li); first_ray.append(rid); first_tri.append(tri_idx[fi])
            i=j
        first_idx=np.asarray(first_idx,int); last_idx=np.asarray(last_idx,int)
        first_ray=np.asarray(first_ray,int); first_tri=np.asarray(first_tri,int)
        p1=locs[first_idx]; n1=mesh.face_normals[tri_idx[first_idx]]
        p2=locs[last_idx];  n2=mesh.face_normals[tri_idx[last_idx]]
        return p1,n1,p2,n2,first_ray,first_tri


# --------------------- 批量 Runner ---------------------
class BatchAllFacesWithEps:
    def __init__(self,
                 config_list_json: str = "configs/list.json",
                 assets_root: str = "assets/object/lego_set",
                 out_dir: str = "results/longshort_pairs_fc",
                 scale: float = 0.001,
                 num_rays: int = 20000,
                 eps_thresh: float = 0.0,
                 mu: float = 0.4,
                 m_dirs: int = 8,
                 convexify: str = "hull",
                 aabb_margin_ratio: float = 0.0,
                 face_jitter: float = 0.0,
                 rand_rotate: bool = False,
                 aabb_normal_align_deg: float = 45.0,
                 device: Optional[str] = None,
                 max_workers: Optional[int] = 1):

        self.aabb_normal_align_deg = float(aabb_normal_align_deg)
        self.cos_align = float(np.cos(np.deg2rad(self.aabb_normal_align_deg)))

        script_dir = Path(__file__).resolve().parent
        project_root = script_dir

        def _resolve(p: str | Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else (project_root / p).resolve()

        self.config_list_json = _resolve(config_list_json)
        self.assets_root      = _resolve(assets_root)
        self.out_dir          = _resolve(out_dir)

        self.scale = float(scale)
        self.num_rays = int(num_rays)
        self.eps_thresh = float(eps_thresh)

        self.sampler = AllFacesSampler(
            aabb_margin_ratio=aabb_margin_ratio,
            face_jitter=face_jitter,
            rand_rotate=rand_rotate,
            convexify=convexify,
            seed=42
        )
        self.eps_eval = ForceClosureEps(mu=mu, m_dirs=m_dirs, device=device, dtype=torch.float32)

        self.max_workers = int(max_workers or 1)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_list_json.exists():
            raise FileNotFoundError(f"未找到 configs 列表文件: {self.config_list_json}")
        with open(self.config_list_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            self.names: List[str] = raw
        elif isinstance(raw, list) and all(isinstance(x, dict) and "name" in x for x in raw):
            self.names = [x["name"] for x in raw]
        else:
            raise ValueError("configs/list.json 需为字符串数组或包含 name 字段的对象数组。")

        self.allowed_exts = (".stl", ".obj", ".ply")

    def _resolve_meshes_from_name(self, name: str) -> List[str]:
        ar = str(self.assets_root)
        if "*" in name or "?" in name:
            hits = []
            for ext in self.allowed_exts:
                pattern = name if name.lower().endswith(ext) else f"{name}{ext}"
                hits.extend(sorted(glob(os.path.join(ar, pattern))))
            return [os.path.abspath(x) for x in hits]

        base, ext = os.path.splitext(name)
        if ext.lower() in self.allowed_exts:
            path = os.path.join(ar, name)
            return [os.path.abspath(path)] if os.path.isfile(path) else []

        hits = []
        for e in self.allowed_exts:
            cand = os.path.join(ar, f"{name}{e}")
            if os.path.isfile(cand):
                hits.append(os.path.abspath(cand))
        return hits

    @staticmethod
    def _uv_on_face(face_id: int, pts: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray) -> np.ndarray:
        """
        将三维点映射到 face_id 对应的 AABB 面内坐标 (u,v)∈[0,1]^2
        0:x-,1:x+,2:y-,3:y+,4:z-,5:z+
        """
        ext = (bb_max - bb_min)
        ext = np.where(ext == 0, 1.0, ext)
        uv = np.zeros((len(pts), 2), dtype=np.float32)
        if face_id in (0,1):  # 固定 x=min/max，面内轴为 (y,z)
            uv[:,0] = (pts[:,1] - bb_min[1]) / ext[1]
            uv[:,1] = (pts[:,2] - bb_min[2]) / ext[2]
        elif face_id in (2,3):  # 固定 y=min/max，面内轴为 (x,z)
            uv[:,0] = (pts[:,0] - bb_min[0]) / ext[0]
            uv[:,1] = (pts[:,2] - bb_min[2]) / ext[2]
        else:  # (4,5) 固定 z=min/max，面内轴为 (x,y)
            uv[:,0] = (pts[:,0] - bb_min[0]) / ext[0]
            uv[:,1] = (pts[:,1] - bb_min[1]) / ext[1]
        return np.clip(uv, 0.0, 1.0)

    def _run_one(self, mesh_path: str) -> Tuple[str, str]:
        try:
            stem = os.path.splitext(os.path.basename(mesh_path))[0]
            out_path = os.path.join(self.out_dir, f"pairs_{stem}.npy")
            if _should_skip_file(out_path):
                return (mesh_path, "skipped")

            mesh = _safe_load_mesh(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump().sum()
            if self.scale and self.scale != 1.0:
                mesh.apply_scale(self.scale)
            mesh = _convexify_mesh(mesh, mode=self.sampler.convexify)

            origins, dirs, face_ids_all, outward_all, bb_min, bb_max = self.sampler.gen_rays(mesh, self.num_rays)
            # 从 AABB 面往里发射，原点轻微反向偏移，避免数值退化
            eps_offset = 1e-6 * (np.linalg.norm(bb_max - bb_min) + 1e-9)
            origins = origins - eps_offset * dirs

            p1, n1, p2, n2, first_ray_idx, _ = self.sampler.first_last_hits(origins, dirs, mesh)
            if p1.shape[0] == 0:
                payload = {"lego_id": stem, "pairs": np.zeros((0,2,3), dtype=np.float32),
                           "normals": np.zeros((0,2,3), dtype=np.float32),
                           "epsilon": np.zeros((0,), dtype=np.float32),
                           "face_ids": np.zeros((0,2), dtype=np.int8),
                           "pair_axis": np.zeros((0,), dtype=np.int8),
                           "face_uv": {"p1": np.zeros((0,2), dtype=np.float32),
                                       "p2": np.zeros((0,2), dtype=np.float32)},
                           "meta": {"scale": self.scale, "aabb": {"min": bb_min.tolist(), "max": bb_max.tolist()}}}
                np.save(out_path, payload, allow_pickle=True)
                return (mesh_path, "ok(empty)")

            # ---------- AABB 外法向一致性过滤 ----------
            outv = outward_all[first_ray_idx]  # 与每一对 (p1,p2) 对应的 AABB 首面外法向
            ok1 = np.sum(n1 * outv, axis=1) >= self.cos_align          # p1 朝外
            ok2 = np.sum(n2 * (-outv), axis=1) >= self.cos_align       # p2 朝对面外
            mask_orient = ok1 & ok2
            if not np.any(mask_orient):
                payload = {"lego_id": stem, "pairs": np.zeros((0,2,3), dtype=np.float32),
                           "normals": np.zeros((0,2,3), dtype=np.float32),
                           "epsilon": np.zeros((0,), dtype=np.float32),
                           "face_ids": np.zeros((0,2), dtype=np.int8),
                           "pair_axis": np.zeros((0,), dtype=np.int8),
                           "face_uv": {"p1": np.zeros((0,2), dtype=np.float32),
                                       "p2": np.zeros((0,2), dtype=np.float32)},
                           "meta": {"scale": self.scale, "aabb": {"min": bb_min.tolist(), "max": bb_max.tolist()}}}
                np.save(out_path, payload, allow_pickle=True)
                return (mesh_path, "ok(orient=0)")

            p1 = p1[mask_orient]; p2 = p2[mask_orient]
            n1 = n1[mask_orient]; n2 = n2[mask_orient]
            outv = outv[mask_orient]
            f1 = face_ids_all[first_ray_idx][mask_orient].astype(np.int8)
            f2 = (f1 ^ 1).astype(np.int8)  # 对面 face 索引
            ax = (f1 // 2).astype(np.int8) # 成对所在轴：0/1/2 -> x/y/z

            B  = p1.shape[0]
            com_np = mesh.center_mass if mesh.is_watertight else mesh.centroid
            device = self.eps_eval.device; dtype = self.eps_eval.dtype
            p1_t = _to_t(p1, device, dtype)
            p2_t = _to_t(p2, device, dtype)
            n1_t = _to_t(n1, device, dtype)
            n2_t = _to_t(n2, device, dtype)
            com_t = _to_t(np.broadcast_to(com_np, (B,3)), device, dtype)

            W1 = self.eps_eval.contact_wrench_rays_batch(p1_t, n1_t, com_t)  # (B,6,m)
            W2 = self.eps_eval.contact_wrench_rays_batch(p2_t, n2_t, com_t)  # (B,6,m)
            W  = torch.cat([W1, W2], dim=2)                                  # (B,6,2m)

            eps_vec = self.eps_eval.epsilon_fw_batched(W)
            eps_np  = _np(eps_vec)
            keep    = eps_np >= self.eps_thresh

            if not np.any(keep):
                payload = {"lego_id": stem, "pairs": np.zeros((0,2,3), dtype=np.float32),
                           "normals": np.zeros((0,2,3), dtype=np.float32),
                           "epsilon": np.zeros((0,), dtype=np.float32),
                           "face_ids": np.zeros((0,2), dtype=np.int8),
                           "pair_axis": np.zeros((0,), dtype=np.int8),
                           "face_uv": {"p1": np.zeros((0,2), dtype=np.float32),
                                       "p2": np.zeros((0,2), dtype=np.float32)},
                           "meta": {"scale": self.scale, "aabb": {"min": bb_min.tolist(), "max": bb_max.tolist()},
                                    "sampler":"aabb_all_faces_rays_first_last",
                                    "eps":{"mu": float(self.eps_eval.mu), "m_dirs": int(self.eps_eval.m_dirs),
                                           "thresh": float(self.eps_thresh)}}}
                np.save(out_path, payload, allow_pickle=True)
                return (mesh_path, "ok(eps=0)")

            # 过滤并打包
            p1 = p1[keep]; p2 = p2[keep]
            n1 = n1[keep]; n2 = n2[keep]
            outv = outv[keep]
            f1  = f1[keep]; f2 = f2[keep]
            ax  = ax[keep]
            eps_np = eps_np[keep]

            pairs   = np.stack([p1, p2], axis=1).astype(np.float32)      # (K,2,3)
            normals = np.stack([n1, n2], axis=1).astype(np.float32)      # (K,2,3)
            face_ids= np.stack([f1, f2], axis=1).astype(np.int8)         # (K,2)
            pair_axis = ax.astype(np.int8)                                # (K,)

            # 计算各自面内 (u,v)
            uv1 = np.vstack([self._uv_on_face(int(fi), p1[i:i+1], bb_min, bb_max)[0] for i,fi in enumerate(f1)]).astype(np.float32)
            uv2 = np.vstack([self._uv_on_face(int(fi), p2[i:i+1], bb_min, bb_max)[0] for i,fi in enumerate(f2)]).astype(np.float32)

            payload = {
                "lego_id": stem,
                "pairs": pairs,
                "normals": normals,
                "epsilon": eps_np.astype(np.float32),
                "face_ids": face_ids,
                "pair_axis": pair_axis,            # 0:x, 1:y, 2:z
                "outward": outv.astype(np.float32),# 首面对外法向
                "face_uv": {"p1": uv1, "p2": uv2}, # 两点在各自面的 2D 坐标
                "meta": {
                    "scale": float(self.scale),
                    "aabb": {"min": bb_min.tolist(), "max": bb_max.tolist()},
                    "sampler": "aabb_all_faces_rays_first_last",
                    "aabb_normal_align_deg": float(self.aabb_normal_align_deg),
                    "eps": {"mu": float(self.eps_eval.mu), "m_dirs": int(self.eps_eval.m_dirs),
                            "thresh": float(self.eps_thresh)}
                }
            }
            np.save(out_path, payload, allow_pickle=True)
            return (mesh_path, f"ok(n={pairs.shape[0]})")

        except Exception as e:
            import traceback
            return (mesh_path, f"error: {e}\n{traceback.format_exc()}")

    def run(self):
        todo = []
        for name in self.names:
            mesh_paths = self._resolve_meshes_from_name(name)
            if not mesh_paths:
                print(f"[跳过] {name}: 未匹配到任何 mesh")
                continue
            todo.extend(mesh_paths)
        if not todo:
            print("[INFO] 没有可运行的任务。")
            return

        if self.max_workers <= 1:
            for p in tqdm(todo, desc="单进程"):
                _, status = self._run_one(p)
                if status.startswith("error"):
                    print(f"[错误] {p} -> {status}")
            return

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(self._run_one, p) for p in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"并行（{self.max_workers}）"):
                mesh, status = fut.result()
                if status.startswith("error"):
                    print(f"[错误] {mesh} -> {status}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="All-Faces AABB sampling + FC ε (save rich antipodal info)")

    parser.add_argument("--config_list_json", type=str, default="configs/list.json")
    parser.add_argument("--assets_root", type=str, default="assets/object")
    parser.add_argument("--out_dir", type=str, default="results/longshort_pairs_fc")

    parser.add_argument("--scale", type=float, default=0.0004)
    parser.add_argument("--num_rays", type=int, default=20000)
    parser.add_argument("--max_workers", type=int, default=1)

    parser.add_argument("--mu", type=float, default=0.4)
    parser.add_argument("--m_dirs", type=int, default=8)
    parser.add_argument("--eps_thresh", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--convexify", type=str, choices=["none","hull","vhacd"], default="hull")

    # AABB 采样相关
    parser.add_argument("--aabb_margin_ratio", type=float, default=0.0)
    parser.add_argument("--face_jitter", type=float, default=0.0)
    parser.add_argument("--rand_rotate", action="store_true", default=False)
    parser.add_argument("--aabb_normal_align_deg", type=float, default=45.0)

    args = parser.parse_args()

    dev = args.device
    if dev.startswith("cuda") and (not torch.cuda.is_available()):
        dev = "cpu"

    runner = BatchAllFacesWithEps(
        config_list_json=args.config_list_json,
        assets_root=args.assets_root,
        out_dir=args.out_dir,
        scale=args.scale,
        num_rays=args.num_rays,
        eps_thresh=args.eps_thresh,
        mu=args.mu,
        m_dirs=args.m_dirs,
        convexify=args.convexify,
        aabb_margin_ratio=args.aabb_margin_ratio,
        face_jitter=args.face_jitter,
        rand_rotate=args.rand_rotate,
        aabb_normal_align_deg=args.aabb_normal_align_deg,
        device=dev,
        max_workers=args.max_workers,
    )
    runner.run()
