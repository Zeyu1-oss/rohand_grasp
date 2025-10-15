import bpy
import os

folder_path = "/home/xzy/Downloads/清华积木相框/1"

obj_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".obj")]
total = len(obj_files)

for idx, filename in enumerate(obj_files, 1):
    file_path = os.path.join(folder_path, filename)
    print(f"[{idx}/{total}] {filename}")
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # 导入 OBJ
    bpy.ops.wm.obj_import(filepath=file_path)
    
    # 选中所有网格
    bpy.ops.object.select_all(action='DESELECT')
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if meshes:
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        
        # 1. 设置原点到几何中心
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        # 2. 移动物体使质心在世界原点 (0,0,0)
        for obj in meshes:
            obj.location = (0.0, 0.0, 0.0)
    
    bpy.ops.wm.obj_export(
        filepath=file_path,
        export_selected_objects=False
    )
    
    print(f"✅ 完成")

print(f"🎉 全部处理完成，共 {total} 个文件")