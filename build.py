import os
import re
from build_gnn_sample import build_training_graph_for_one_instance

"""
功能：
  自动批量读取目录下所有形如：
     1764039581741 items.txt
     1764039581741 limits.txt
  的数据对，并对每一对文件自动生成一个 GNN 训练 npz。

输入：
  data_dir : 你的 items 和 limits 所在文件夹
输出：
  data_dir/processed/xxxxxxx_reduced_train.npz
"""

# ============================
# 正则匹配：十三位数字
# ============================
ITEMS_PATTERN = re.compile(r"^(\d{13}) items")
LIMITS_PATTERN = re.compile(r"^(\d{13}) limits")


def find_all_data_pairs(data_dir: str):
    """
    在 data_dir 下找到所有成对的 (items, limits)
    返回：
        pairs = [
            (id_str, items_path, limits_path),
            ...
        ]
    """
    files = os.listdir(data_dir)

    items_map = {}
    limits_map = {}

    for f in files:
        m = ITEMS_PATTERN.match(f)
        if m:
            id_str = m.group(1)
            items_map[id_str] = os.path.join(data_dir, f)
            continue

        m = LIMITS_PATTERN.match(f)
        if m:
            id_str = m.group(1)
            limits_map[id_str] = os.path.join(data_dir, f)
            continue

    # 取交集才是完整一组
    ids = sorted(list(set(items_map.keys()) & set(limits_map.keys())))

    pairs = []
    for id_str in ids:
        pairs.append((id_str, items_map[id_str], limits_map[id_str]))

    return pairs


def build_all_training_samples(data_dir: str):
    """
    对目录下所有数据组批量生成训练 npz
    """
    print("============================================")
    print("[Batch] 扫描目录 =", data_dir)

    # 找到所有成对的 (items, limits)
    pairs = find_all_data_pairs(data_dir)
    print(f"[Batch] 共找到 {len(pairs)} 组数据")

    # 输出目录
    out_dir = os.path.join(data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 逐组处理
    for id_str, items_path, limits_path in pairs:
        print("============================================")
        print(f"[Batch] 处理数据组 {id_str}")
        out_npz = os.path.join(out_dir, f"{id_str}_reduced_train.npz")

        build_training_graph_for_one_instance(
            items_path=items_path,
            limits_path=limits_path,
            out_npz_path=out_npz,
            max_outer_iter=50,
            random_seed=42,
        )

    print("============================================")
    print("[Batch] 全部训练数据构造完毕！")


if __name__ == "__main__":
    # === 修改这里 ===
    data_dir = r"C:\Users\89328\Desktop\demo\match_map_3600"

    build_all_training_samples(data_dir)
