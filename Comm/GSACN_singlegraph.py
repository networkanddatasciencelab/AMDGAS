import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import numpy as np
import random
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.pool import SAGPooling, ASAPooling, PANPooling, MemPooling
from torch_geometric.utils import degree, add_self_loops, to_networkx
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GraphConv
from torch_scatter import scatter_add


def mlc_loss(arch_param):
    y_pred_neg = arch_param
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    aux_loss = torch.mean(neg_loss)
    return aux_loss


def cos_loss(embeddings):
    # 正规化 embedding 向量使其更容易计算内积
    norm_embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)

    # 计算所有节点所有向量的内积矩阵
    dot_products = torch.matmul(norm_embeddings, norm_embeddings.transpose(1, 2))

    # 计算正则化项，即非对角线元素（不同向量之间的内积）的平方和
    # 注意：需要排除自己与自己的内积（对角线元素）
    reg_loss = (dot_products ** 2).sum() - (dot_products.diagonal(dim1=1, dim2=2) ** 2).sum()

    reg_loss_n = reg_loss / (embeddings.size(0) * (embeddings.size(1) ** 2 - embeddings.size(1)))

    return reg_loss_n


class SelfAttention_gate(nn.Module):
    def __init__(self, embed_dim, ac_shape, heads=1, useac=True):
        super(SelfAttention_gate, self).__init__()

        self.useac = useac
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.keys = nn.Linear(self.embed_dim, self.embed_dim)  # , bias=False)
        self.queries = nn.Linear(self.embed_dim, self.embed_dim)  # , bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)
        if useac:
            self.gate = nn.Sequential(nn.Linear(ac_shape, 6, bias=True))

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.heads, self.head_dim)
        x = x.view(new_size)  # [N.6,1,16]其中1是head
        return x.permute(0, 2, 1, 3)  # [N.1,6,16]其中1是head

    def forward(self, embeddings, ac):  # ac是图的同配系数

        N, T, C = embeddings.shape  # N*6*nhid(16)

        keys = self.trans_to_multiple_heads(self.keys(embeddings))
        queries = self.trans_to_multiple_heads(self.queries(embeddings))
        values = self.trans_to_multiple_heads(embeddings)

        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # [N,1,6,6]
        attention = attention_scores / math.sqrt(self.head_dim)  #
        if self.useac:
            ac_gate = self.gate(ac)
            ac_gate = ac_gate.view(N, 1, 1, ac_gate.size(1))  # 调整形状以匹配注意力分数
            attention = attention * ac_gate

        # if self.useac:
        #     ac_gate = self.gate(ac)
        #     ac_gate = ac_gate.view(N, 1, 1, ac_gate.size(1))
        #     keys = keys * ac_gate
        #
        # attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # [N,1,6,6]
        # attention = attention_scores / math.sqrt(self.head_dim)

        # attention =F.sigmoid(attention)
        # attention1 = attention
        attention = torch.softmax(attention, dim=-1)

        context = torch.matmul(attention, values)  # [N,1,6,16]
        context = context.permute(0, 2, 1, 3).contiguous()  # [N,6,1,16]
        new_size = context.size()[: -2] + (self.embed_dim,)  # [N,6,16]
        context = context.view(*new_size)  # [N,6,16]

        return context, attention


class ODS_Layer(nn.Module):
    def __init__(self, nhid1, nhid2, ac_shape, useac=True):
        super(ODS_Layer, self).__init__()
        self.gcn1 = GCNConv(nhid1, nhid2)
        self.sage1 = SAGEConv(nhid1, nhid2)
        self.gat1 = GATConv(nhid1, int(0.5 * nhid2), heads=2)
        self.gin1 = GINConv(nn=nn.Sequential(nn.Linear(nhid1, nhid2), nn.ReLU(), nn.Linear(nhid2, nhid2)), eps=1e-9)
        self.lin1 = nn.Linear(nhid1, nhid2)
        self.graph1 = GraphConv(nhid1, nhid2)
        self.att_layer1 = SelfAttention_gate(nhid2, ac_shape, useac=useac)
        # self.linout = nn.Linear(6, 1)

    def forward(self, x, edge_index, ac):
        gcnout = self.gcn1(x, edge_index)
        sageout = self.sage1(x, edge_index)
        gatout = self.gat1(x, edge_index)
        linout = self.lin1(x)
        ginout = self.gin1(x, edge_index)
        graphout = self.graph1(x, edge_index)
        embeddings = torch.stack([gcnout, sageout, gatout, linout, ginout, graphout], dim=1)

        # 正交性dpp
        # dpp_losses1 = []
        # for i in range(embeddings.shape[0]):  # 遍历每个嵌入
        #     emb = embeddings[i]  # 获取第i个嵌入
        #     norm_emb = emb / emb.norm(dim=1, keepdim=True)
        #     gram_matrix = norm_emb @ norm_emb.T  # 计算嵌入和它的转置的乘积
        #     det = torch.logdet(gram_matrix)  # 计算行列式并取对数
        #     dpp_losses1.append(det)
        # dpp_losses1 = torch.stack(dpp_losses1)

        # self attention
        output1, attention_weights1 = self.att_layer1(embeddings, ac)  # [N,6,16]
        x = output1.sum(dim=1)
        # x = output1.transpose(1, 2)  # 交换维度，变为 [N, 16, 6]
        # x = x.max(dim=-1)[0]
        # x = self.linout(x)  # 现在应用线性层，输出将是 [N, 16, 1]
        # x = x.squeeze(-1)  # 去掉最后一个维度，输出将是 [N, 16]
        # aux_loss = mlc_loss(attention_weights1)
        aux_loss = cos_loss(output1)
        return x, attention_weights1, aux_loss


class GSACN_Net(nn.Module):  # graph search and choose network
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nout, useac, ac_shape, usase=False, num_node=10000):
        super(GSACN_Net, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid1)
        # self.lin1 = nn.Linear(nfeat, nhid1)
        self.ac = ac_shape
        self.ods_layer1 = ODS_Layer(nhid1, nhid2, ac_shape=ac_shape, useac=useac)
        self.ods_layer2 = ODS_Layer(nhid2, nhid3, ac_shape=ac_shape, useac=useac)
        # self.ods_layer3 = ODS_Layer(nhid3, nhid3, ac_shape=ac_shape, useac=useac)
        # self.fc1 = nn.Linear(nhid2, nhid3)
        self.out = nn.Linear(nhid3, nout)
        # self.outgcn = GCNConv(nhid3, nout)
        # self.fc2 = nn.Linear(nfeat, nout)
        self.bn1 = nn.BatchNorm1d(nhid1)
        self.bn2 = nn.BatchNorm1d(nhid2)
        self.bn3 = nn.BatchNorm1d(nhid3)
        self.dropout = 0.2
        if usase:
            self.se = nn.Parameter(torch.rand(num_node, nhid1), requires_grad=True)

    def dropedge(self, edge_index, drop_prob=0.1):
        """
        随机丢弃图中的边。
        参数:
            edge_index: [2, E] 的张量，表示图的边，其中 E 是边的数量。
            drop_prob: 丢弃边的概率，默认为 0.1。
        返回:
            处理后的 edge_index。
        """
        # 确保在训练模式下才进行边的丢弃
        if self.training and drop_prob > 0:
            num_edges = edge_index.size(1)
            # 生成一个随机掩码来决定哪些边被保留
            keep_mask = torch.rand(num_edges) > drop_prob
            # 应用掩码
            edge_index = edge_index[:, keep_mask]
        return edge_index

    def forward(self, data):
        x1 = data.x
        x1 = x1.float()
        ac = data.ac
        edge_index = data.edge_index
        # edge_index = self.dropedge(edge_index, drop_prob=0.1)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # x2 = self.lin1(x1)
        x2 = self.conv1(x1, edge_index)
        x2 = self.bn1(x2)
        # x2 = F.relu(x2)
        # x2 = F.dropout(x2, training=self.training)
        x3, attention_weights1, aux_loss1 = self.ods_layer1(x2, edge_index, ac)
        x3 = self.bn2(x3)
        # x3 = F.relu(x3)
        x_1 = x3
        # x3 = F.dropout(x3, training=self.training)
        x3, attention_weights2, aux_loss2 = self.ods_layer2(x3, edge_index, ac)
        x3 = self.bn3(x3)
        # x4 = self.fc1(x3)
        # x4 = self.bn3(x4)
        x4 = F.relu(x3)
        x4 = F.dropout(x4, p=self.dropout, training=self.training)
        x4 = x4 + x_1 + x2
        # x, dpp_loss2 = self.ods_layer2(x, edge_index, ac)
        x5 = self.out(x4)
        return x5, attention_weights2, aux_loss1 + aux_loss2


class GCN_Net(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.conv3 = GCNConv(nhid, nhid)
        self.out = nn.Linear(nhid, nout)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class GAT_Net(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(nfeat, nhid // 2, heads=2, dropout=dropout)
        self.conv2 = GATConv(nhid, nhid // 2, heads=2, dropout=dropout)
        self.conv3 = GATConv(nhid, nhid // 2, heads=2, dropout=dropout)
        self.out = nn.Linear(nhid, nout)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class GIN_Net(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GIN_Net, self).__init__()
        self.conv1 = GINConv(nn=nn.Sequential(nn.Linear(nfeat, nhid), nn.ReLU(), nn.Linear(nhid, nhid)), eps=1e-9)
        self.conv2 = GINConv(nn=nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid)), eps=1e-9)
        self.conv3 = GINConv(nn=nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid)), eps=1e-9)
        self.out = nn.Linear(nhid, nout)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class SAGE_Net(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(SAGE_Net, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nhid)
        self.conv3 = SAGEConv(nhid, nhid)
        self.out = nn.Linear(nhid, nout)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class GraphConv_Net(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GraphConv_Net, self).__init__()
        self.conv1 = GraphConv(nfeat, nhid)
        self.conv2 = GraphConv(nhid, nhid)
        self.conv3 = GraphConv(nhid, nhid)
        self.out = nn.Linear(nhid, nout)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class MLP_Net(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MLP_Net, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)
        self.out = nn.Linear(nhid, nout)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x
