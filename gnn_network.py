import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.pool import SAGPooling, ASAPooling, PANPooling, MemPooling
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

class GCN_Net_ASAP(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super().__init__()
        self.conv1 = GCNConv(in_channel, hidden)
        self.pool1 = ASAPooling(hidden, ratio=0.5)
        self.conv2 = GCNConv(hidden, out_channel)
        self.pool2 = ASAPooling(out_channel, ratio=0.5)
        self.lin = nn.Linear(out_channel, out_channel)
        # self.classifier = nn.Linear(out_channel, out_channel)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, edge_index, _, batch,  _ = self.pool1(x, edge_index, None, batch)

        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x, edge_index, _, batch,  _ = self.pool2(x, edge_index, None, batch)
        #
        # x = self.lin(x)

        x = global_mean_pool(x, batch)
        # x = self.classifier(x)
        return x

class GIN_Net_ASAP(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(GIN_Net_ASAP, self).__init__()
        self.conv1 = GINConv(nn=nn.Sequential(
            nn.Linear(in_channel, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        ), eps=1e-9)
        self.conv2 = GINConv(nn=nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channel)
        ), eps=1e-13)
        self.pool1 = ASAPooling(hidden, ratio=0.5)
        self.pool2 = ASAPooling(out_channel, ratio=0.5)
        self.lin = nn.Linear(out_channel, out_channel)
        # self.classifier = nn.Linear(out_channel, out_channel)  # 根据需要调整输出维度

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, edge_index, _, batch,  _ = self.pool1(x, edge_index, None, batch)

        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x, edge_index, _, batch,  _ = self.pool2(x, edge_index, None, batch)
        # x = self.lin(x)
        x = global_mean_pool(x, batch)
        # x = self.classifier(x)
        return x

class GraphSAGE_Net_ASAP(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(GraphSAGE_Net_ASAP, self).__init__()
        self.sage1 = SAGEConv(in_channel, hidden)
        self.pool1 = ASAPooling(hidden, ratio=0.5)
        self.sage2 = SAGEConv(hidden, out_channel)
        self.pool2 = ASAPooling(out_channel, ratio=0.5)
        self.lin = nn.Linear(out_channel, out_channel)
        # self.classifier = nn.Linear(out_channel, out_channel)  # 根据需要调整输出维度

    def forward(self, x, edge_index, batch):

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, edge_index, _, batch,  _ = self.pool1(x, edge_index, None, batch)

        x = self.sage2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        # x = self.lin(x)
        x = global_mean_pool(x, batch)
        # x = self.classifier(x)

        return x

class GAT_Net_ASAP(nn.Module):
    def __init__(self, in_channel, hidden, out_channel, heads=2):
        super(GAT_Net_ASAP, self).__init__()
        self.gat1 = GATConv(in_channel, hidden, heads=heads)
        self.pool1 = ASAPooling(hidden * heads, ratio=0.5)
        self.gat2 = GATConv(hidden * heads, out_channel)
        self.pool2 = ASAPooling(out_channel, ratio=0.5)
        self.lin = nn.Linear(out_channel, out_channel)
        # self.classifier = nn.Linear(out_channel, out_channel)  # 根据需要调整输出维度

    def forward(self, x, edge_index, batch):

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)

        x = self.gat2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        # x = self.lin(x)
        x = global_mean_pool(x, batch)
        # x = self.classifier(x)
        return x

class Linear_Net_ASAP(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(Linear_Net_ASAP, self).__init__()
        # 定义两个线性层
        self.lin1 = nn.Linear(in_channel, hidden)
        self.lin2 = nn.Linear(hidden, out_channel)
        self.lin = nn.Linear(out_channel, out_channel)
        # 定义两个ASAPooling层
        self.pool1 = ASAPooling(hidden, ratio=0.5)
        self.pool2 = ASAPooling(out_channel, ratio=0.5)
        # 最后的全局平均池化层和分类器层
        # self.classifier = nn.Linear(out_channel, out_channel)  # 输出大小根据任务调整

    def forward(self, x, edge_index, batch):

        # 第一层线性变换
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # 第一层ASAPooling
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)

        # 第二层线性变换
        x = self.lin2(x)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # # 第二层ASAPooling
        # x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        # x = self.lin(x)
        # 全局平均池化获取图级特征
        x = global_mean_pool(x, batch)
        # 通过分类器
        # x = self.classifier(x)

        return x


