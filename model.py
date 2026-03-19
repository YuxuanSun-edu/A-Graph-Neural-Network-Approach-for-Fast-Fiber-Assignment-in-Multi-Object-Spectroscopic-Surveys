# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, BatchNorm
except ImportError as e:
    raise ImportError(
        "需要安装 pytorch_geometric，建议使用已匹配版本的安装指令：\n"
    ) from e


# ================================================
# 1. GNN 模型：用于 MIS 节点打分
# ================================================
class MISScoreGNN(nn.Module):
    """
    用于 Maximum-Weight Independent Set 的节点打分模型。
    输入图 (x, edge_index)，输出每个节点一个 logit（未过 sigmoid）。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        assert num_layers >= 2, "建议 num_layers >= 2"

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # -------------------------
        # GNN 主干（GraphSAGE）
        # -------------------------
        convs = []
        bns = []

        # 第一层：in_dim → hidden_dim
        convs.append(SAGEConv(in_dim, hidden_dim))
        bns.append(BatchNorm(hidden_dim))

        # 后续层：hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            convs.append(SAGEConv(hidden_dim, hidden_dim))
            bns.append(BatchNorm(hidden_dim))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        # -------------------------
        # 输出层：hidden_dim → 1（logit）
        # -------------------------
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index):
        """
        x: [N, in_dim]
        edge_index: [2, E]

        returns:
            logits: [N]  未经过 sigmoid
        """
        h = x

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_in = h
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # 从第二层开始添加残差
            if i > 0:
                h = h + h_in

        out = self.mlp_out(h)     # [N, 1]
        out = out.squeeze(-1)     # [N]
        return out


# ================================================
# 2. BCE Loss 计算（train.py 需要）
# ================================================
def compute_bce_loss(logits, labels, weight=None):
    """
    计算二分类 BCE loss（适用于 MIS 打分任务）。
    - logits: [N] 或 [N,1]，未过 sigmoid
    - labels: [N] 或 [N,1]，0/1
    - weight: 每个节点的权重（optional）
    """
    logits = logits.view(-1)
    labels = labels.view(-1).float()

    if weight is not None:
        weight = weight.view(-1).float()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            weight=weight,
            reduction="mean",
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="mean"
        )

    return loss


# ================================================
# 3. Sigmoid 概率（用于推理 / 排序）
# ================================================
def predict_prob(logits: torch.Tensor):
    """将 logit 转成 0~1 概率"""
    return torch.sigmoid(logits)


# ================================================
# 模块导出
# ================================================
__all__ = [
    "MISScoreGNN",
    "predict_prob",
    "compute_bce_loss",
]
