import os
import json

folder_path = "assets/object"  # 修改为你的路径
output_json = "src/list1.json"  # 输出的 JSON 文件名

obj_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".obj")]

# 排序（可选）
obj_files.sort()

# 保存为 JSON
output_path = os.path.join(folder_path, output_json)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(obj_files, f, indent=2, ensure_ascii=False)

print(f"✅ 已保存 {len(obj_files)} 个文件名到: {output_path}")
print(f"📄 内容预览:")
print(json.dumps(obj_files[:5], indent=2, ensure_ascii=False))
if len(obj_files) > 5:
    print(f"... 还有 {len(obj_files) - 5} 个文件")