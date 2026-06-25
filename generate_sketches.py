import os
import glob
import numpy as np
import trimesh
import cv2
from tqdm import tqdm
import platform
import random
import shutil

# ================= 配置区域 =================
DATA_ROOT = r"F:\DataSetF\FaMoS"  # 数据集根目录
OUTPUT_DIR = r"./output_sketches_all"  # 输出根目录
IMAGE_SIZE = 512  # 输出图片分辨率
RANDOM_SEED = 42  # 随机种子，保证每次划分结果一致
# ===========================================

# --- Windows 兼容性设置 ---
if platform.system() == 'Windows':
    if 'PYOPENGL_PLATFORM' in os.environ:
        del os.environ['PYOPENGL_PLATFORM']
else:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 尝试导入 pyrender
try:
    import pyrender

    PYRENDER_AVAILABLE = True
except ImportError:
    print("警告: Pyrender 未安装或无法加载，将使用纯数学投影模式。")
    PYRENDER_AVAILABLE = False
except OSError:
    print("警告: 系统缺少 OpenGL 驱动，将使用纯数学投影模式。")
    PYRENDER_AVAILABLE = False


def render_sketch_math_fallback(mesh):
    """[Plan B] 纯数学投影模式"""
    vertices = mesh.vertices
    if vertices.shape[0] == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    min_xy = vertices[:, :2].min(axis=0)
    max_xy = vertices[:, :2].max(axis=0)
    scale = max_xy - min_xy
    scale[scale == 0] = 1
    normalized = (vertices[:, :2] - min_xy) / scale

    h, w = IMAGE_SIZE, IMAGE_SIZE
    img = np.zeros((h, w), dtype=np.uint8)

    px = (normalized[:, 0] * (w - 1)).astype(np.int32)
    py = ((1 - normalized[:, 1]) * (h - 1)).astype(np.int32)

    for x, y in zip(px, py):
        cv2.circle(img, (x, y), 1, 255, -1)

    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img, kernel, iterations=1)
    edges = cv2.Canny(img_dilated, 50, 150)

    return 255 - edges


def setup_renderer(mesh):
    """配置 Pyrender 场景"""
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.4, 0.4, 0.4])

    mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_pr)

    center = mesh.bounds.mean(axis=0)
    camera_pose = np.eye(4)
    # 调整相机位置
    camera_pose[:3, 3] = center + np.array([0, 0, 0.6])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=IMAGE_SIZE, viewport_height=IMAGE_SIZE)
    return scene, r


def extract_gctd_sketch(mesh_path):
    """基于深度和光照的混合线条提取 (保持原逻辑不变)"""
    mesh = trimesh.load(mesh_path, process=False)

    if not PYRENDER_AVAILABLE:
        return render_sketch_math_fallback(mesh)

    try:
        scene, r = setup_renderer(mesh)
        color, depth = r.render(scene)
        r.delete()

        if np.max(depth) == 0:
            return render_sketch_math_fallback(mesh)

        # 1. 深度图边缘 (轮廓)
        depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-6)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        edges_silhouette = cv2.Canny(depth_uint8, 50, 150)

        # 2. 法线/光照边缘 (内部细节)
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        # 保持你认为效果好的参数 (30, 75)
        edges_internal = cv2.Canny(gray, 30, 75)

        # 3. 融合
        edges_combined = cv2.bitwise_or(edges_silhouette, edges_internal)

        # 4. 风格化
        kernel = np.ones((2, 2), np.uint8)
        sketch_thick = cv2.dilate(edges_combined, kernel, iterations=1)

        # 反色
        sketch_final = 255 - sketch_thick

        # 模拟渗墨
        sketch_gctd = cv2.bilateralFilter(sketch_final.astype(np.uint8), 9, 75, 75)

        return sketch_gctd

    except Exception as e:
        return render_sketch_math_fallback(mesh)


def process_subject_list(subjects, split_name):
    """
    处理特定列表中的受试者，将结果保存到对应子文件夹
    """
    save_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    print(f"\n--- 正在处理 {split_name} 集 (共 {len(subjects)} 人) ---")
    print(f"    保存路径: {save_dir}")

    # 遍历该集合中的每个受试者
    for subj in tqdm(subjects, desc=f"Generating {split_name}"):
        subj_path = os.path.join(DATA_ROOT, subj)

        # 获取该受试者下的所有序列
        sequences = sorted([d for d in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, d))])

        # [关键需求] 遍历该受试者的 *所有* 序列
        for seq_name in sequences:
            seq_path = os.path.join(subj_path, seq_name)

            # 查找网格
            mesh_files = sorted(glob.glob(os.path.join(seq_path, "*.ply")))
            if not mesh_files: continue

            # 取第一帧作为身份参考
            ref_mesh_path = mesh_files[0]

            # 生成草图
            sketch_img = extract_gctd_sketch(ref_mesh_path)

            # 保存 (文件名包含序列名，避免冲突)
            save_name = f"{subj}_{seq_name}_sketch.png"
            save_path = os.path.join(save_dir, save_name)

            try:
                cv2.imwrite(save_path, sketch_img)
                count += 1
            except Exception as e:
                print(f"保存失败: {save_path}, Error: {e}")

    return count


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"错误：数据集路径不存在 {DATA_ROOT}")
        return

    # 1. 准备目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 获取所有受试者并打乱
    subjects = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("FaMoS_subject")])

    # 设置随机种子，保证实验可复现
    random.seed(RANDOM_SEED)
    random.shuffle(subjects)

    total_subjects = len(subjects)
    print(f"总受试者人数: {total_subjects}")

    # 3. 按 8:1:1 划分受试者
    n_train = int(total_subjects * 0.8)
    n_val = int(total_subjects * 0.1)

    train_subs = subjects[:n_train]
    val_subs = subjects[n_train: n_train + n_val]
    test_subs = subjects[n_train + n_val:]

    print(f"数据集划分情况:")
    print(f"  - 训练集 (Train): {len(train_subs)} 人")
    print(f"  - 验证集 (Val)  : {len(val_subs)} 人")
    print(f"  - 测试集 (Test) : {len(test_subs)} 人")

    # 4. 分别执行生成任务
    n_train_imgs = process_subject_list(train_subs, "train")
    n_val_imgs = process_subject_list(val_subs, "val")
    n_test_imgs = process_subject_list(test_subs, "test")

    print("\n================ 处理完成 ================")
    print(f"生成统计:")
    print(f"  Train 草图数: {n_train_imgs}")
    print(f"  Val   草图数: {n_val_imgs}")
    print(f"  Test  草图数: {n_test_imgs}")
    print(f"  总计        : {n_train_imgs + n_val_imgs + n_test_imgs}")
    print(f"文件已保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()