import os
import glob
import numpy as np
from tqdm import tqdm

# 配置你的路径
DATA_ROOT = r"F:\DataSetF\FaMoS"


def find_anomalies():
    print(f"正在扫描异常序列: {DATA_ROOT} ...")

    if not os.path.exists(DATA_ROOT):
        print("路径不存在！")
        return

    subjects = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("FaMoS_subject")])

    # 存储结构: (帧数, 相对路径)
    seq_records = []

    for subj in tqdm(subjects):
        subj_path = os.path.join(DATA_ROOT, subj)
        sequences = [d for d in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, d))]

        for seq in sequences:
            seq_path = os.path.join(subj_path, seq)
            ply_files = glob.glob(os.path.join(seq_path, "*.ply"))
            n_frames = len(ply_files)

            # 记录所有序列的信息
            # 使用相对路径方便查看: "FaMoS_subject_001\Seq_Smile"
            rel_path = os.path.join(subj, seq)
            seq_records.append((n_frames, rel_path))

    # --- 分析与输出 ---
    # 按帧数从小到大排序
    seq_records.sort(key=lambda x: x[0])

    print("\n" + "=" * 50)
    print("【最短的 30 个序列】")
    print("=" * 50)
    for i in range(min(30, len(seq_records))):
        n, path = seq_records[i]
        print(f"帧数: {n}\t 路径: {path}")

    print("\n" + "=" * 50)
    print("【建议操作】")
    print("1. 请手动检查上述帧数为 1 的文件夹。")
    print("2. 这种数据无法用于 4D 训练，请直接删除该文件夹，或者重新下载该部分数据。")
    print("3. 建议设置 config.py 中的 SEQ_LENGTH = 128 (因为大部分数据都在200帧以上)。")
    print("=" * 50)


if __name__ == "__main__":
    find_anomalies()