#!/usr/bin/env python3
import os
import json

src_dir = "/home/xzy/rohand_grasp/assets/object"
out_json = "configs/list.json"

# 找到所有 obj 文件
obj_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".obj")]

# 保留完整的文件名（包括 .obj 后缀）
names = [f for f in obj_files]

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(names, f, indent=4, ensure_ascii=False)

print(f"✅ 已保存 {len(names)} 个名字到 {out_json}")