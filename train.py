# train.py (最终修正版：自动路径)
import os
import glob
import random
import numpy as np
from typing import List

import torch
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError as e:
    raise ImportError("请安装 torch_geometric") from e

from model import MISScoreGNN, compute_bce_loss

# =========================
# 路径配置 (自动适配)
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 这些默认值仅在直接运行 train.py 时生效，pipeline 调用时会传入具体路径
DEFAULT_NPZ_DIR = os.path.join(CURRENT_DIR, "match_map_3600", "processed")

# =========================
# 1. 读取所有 npz
# =========================
def load_all_graphs_from_npz_dir(npz_dir: str) -> List[Data]:
    print(f"[Train] 正在扫描数据集目录: {npz_dir}") 
    pattern = os.path.join(npz_dir, "*_reduced_train.npz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        sub_dir = os.path.join(npz_dir, "processed")
        if os.path.exists(sub_dir):
            pattern = os.path.join(sub_dir, "*_reduced_train.npz")
            files = sorted(glob.glob(pattern))

    print(f"[Train] 找到图文件数: {len(files)}")
    if not files:
        # 如果还是找不到，为了防止 Crash，返回空列表
        print(f"⚠️ [Warning] 在 {npz_dir} 未找到 *_reduced_train.npz")
        return []

    graphs = []
    for path in files:
        try:
            data = np.load(path)
            x = torch.from_numpy(data["x"]).float()
            edge_index = torch.from_numpy(data["edge_index"])
            # 兼容处理：如果没有 y，生成全 0
            if "y" in data:
                y = torch.from_numpy(data["y"]).long()
            else:
                y = torch.zeros(x.size(0), dtype=torch.long)

            g = Data(x=x, edge_index=edge_index, y=y)
            g.file_id = os.path.basename(path)
            graphs.append(g)
        except Exception as e:
            print(f"⚠️ 跳过损坏文件 {os.path.basename(path)}: {e}")
    
    return graphs

# =========================
# 2. 划分 train / val
# =========================
def split_train_val(graphs: List[Data], val_ratio: float = 0.2, seed: int = 42):
    if not graphs: return [], []
    
    random.Random(seed).shuffle(graphs)
    n = len(graphs)
    
    if n == 1:
        print(f"[Split] 只有 1 张图，进入【自举训练模式】(Train=Val)")
        return graphs, graphs

    n_val = max(1, int(n * val_ratio))
    val_graphs = graphs[:n_val]
    train_graphs = graphs[n_val:]
    
    if len(train_graphs) == 0 and len(graphs) > 1:
        train_graphs = graphs
        val_graphs = graphs[:1]

    print(f"[Split] Train={len(train_graphs)}, Val={len(val_graphs)}")
    return train_graphs, val_graphs

# =========================
# 3. 训练 & 验证 Core
# =========================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = compute_bce_loss(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
    return total_loss / max(1, total_nodes)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        loss = compute_bce_loss(logits, batch.y)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).long()
        correct = (pred == batch.y).sum().item()
        
        total_loss += loss.item() * batch.num_nodes
        total_correct += correct
        total_nodes += batch.num_nodes
    
    return total_loss / max(1, total_nodes), total_correct / max(1, total_nodes)

# =========================
# 4. 封装好的训练管线 (供 pipeline 调用)
# =========================
def run_training_pipeline(
    npz_dir: str, 
    save_path: str = "best_mis_gnn.pt",
    max_epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 4,
    force_cpu: bool = False
):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print(f"[Train] 设备: {device}")

    # 1. 加载数据
    graphs = load_all_graphs_from_npz_dir(npz_dir)
    if not graphs:
        print("[Train] 没有有效数据，跳过训练。")
        return

    train_graphs, val_graphs = split_train_val(graphs, seed=seed)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    # 2. 获取维度
    if len(train_graphs) > 0:
        in_dim = train_graphs[0].x.size(1)
    elif len(val_graphs) > 0:
        in_dim = val_graphs[0].x.size(1)
    else:
        return

    # 3. 初始化/加载模型
    model = MISScoreGNN(in_dim=in_dim, hidden_dim=64, num_layers=3, dropout=0.2).to(device)
    
    # 确保保存目录存在
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_path):
        print(f"[Init] 加载旧模型微调: {save_path}")
        try:
            ckpt = torch.load(save_path, map_location=device)
            # 维度检查
            saved_in_dim = ckpt.get("in_dim", in_dim) # 如果没有记录，假设一致
            if saved_in_dim == in_dim:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                print(f"[Init] 维度不匹配 (Saved={saved_in_dim}, Curr={in_dim})，从头训练")
        except:
            print("[Init] 加载失败，从头训练")
    else:
        print("[Init] 从头训练")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")

    # 4. Loop
    print(f"[Train] 开始训练 {max_epochs} epochs...")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                "model_state_dict": model.state_dict(),
                "in_dim": in_dim,
                "val_loss": val_loss
            }
            torch.save(state, save_path)
    
    print(f"[Train] 结束。Best Val Loss={best_val_loss:.6f}, Saved to {save_path}")

# =========================
# 5. 命令行入口
# =========================
if __name__ == "__main__":
    run_training_pipeline(npz_dir=DEFAULT_NPZ_DIR)