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
from gnn_network import GAT_Net_ASAP, GCN_Net_ASAP, GIN_Net_ASAP, GraphSAGE_Net_ASAP, Linear_Net_ASAP


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 固定torch
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(4211)


def calculate_ac_for_batch(data,no0=0,norm=True):
    edge_index, batch = data.edge_index, data.batch
    num_graphs = data.num_graphs

    ac_list = []

    for i in range(num_graphs):
        mask = (batch[edge_index[0]] == i) & (batch[edge_index[1]] == i)
        graph_edge_index = edge_index[:, mask]

        if graph_edge_index.size(1) == 0 or len(torch.unique(batch == i)) <= 1:  # 没有边或只有一个节点
            ac_list.append([no0,no0,no0,no0,no0])
            continue

        # 节点映射
        unique_nodes, new_node_indices = torch.unique(graph_edge_index, return_inverse=True)
        graph_edge_index_remap = new_node_indices.view_as(graph_edge_index)

        # 计算当前子图的节点的度
        deg = degree(graph_edge_index_remap[0], num_nodes=len(unique_nodes))

        # 创建一个数组来保存每个边的度对
        deg_pairs = torch.stack((deg[graph_edge_index_remap[0]], deg[graph_edge_index_remap[1]]), dim=0)

        # 度的乘积的均值、度的均值和度的平方的均值
        mean_deg_product = (deg_pairs[0] * deg_pairs[1]).mean()
        mean_deg = deg.mean()
        mean_deg_sq = (deg ** 2).mean()

        # 同配系数
        if mean_deg_sq - mean_deg ** 2 == 0:  # 防止除以零
            ac = torch.tensor(0, dtype=torch.float, device=deg.device)
        else:
            ac = (mean_deg_product - mean_deg ** 2) / (mean_deg_sq - mean_deg ** 2)

        # 聚类系数
        G = to_networkx(data.__class__(edge_index=graph_edge_index_remap), to_undirected=True)
        # 全图的平均聚类系数
        if G.number_of_edges()>0:
            cc = nx.average_clustering(G)
        else:
            cc = 0

        if norm:
            deg_min, deg_max = deg.min(), deg.max()
            mean_deg = (mean_deg - deg_min) / (deg_max - deg_min) if deg_max != deg_min else 0

            # 对于度的乘积的均值
            deg_product_min, deg_product_max = (deg_pairs[0] * deg_pairs[1]).min(), (deg_pairs[0] * deg_pairs[1]).max()
            mean_deg_product = (mean_deg_product - deg_product_min) / (
                        deg_product_max - deg_product_min) if deg_product_max != deg_product_min else 0

            # 对于度的平方的均值
            deg_sq_min, deg_sq_max = (deg ** 2).min(), (deg ** 2).max()
            mean_deg_sq = (mean_deg_sq - deg_sq_min) / (
                        deg_sq_max - deg_sq_min) if deg_sq_max != deg_sq_min else 0

        ac_list.append([mean_deg_product+no0, mean_deg+no0, mean_deg_sq+no0, ac+no0,cc+no0])

    return torch.tensor(ac_list, dtype=torch.float, device=edge_index.device)


class GCN_Net(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super().__init__()
        self.conv1 = GCNConv(in_channel, hidden)
        self.conv2 = GCNConv(hidden, out_channel)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # return F.log_softmax(x, dim=1)
        return x


class GIN_Net(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(GIN_Net, self).__init__()
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

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GraphSAGE_Net(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(in_channel, hidden)
        self.sage2 = SAGEConv(hidden, out_channel)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x


class GAT_Net(nn.Module):
    def __init__(self, in_channel, hidden, out_channel, heads=2):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(in_channel, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, out_channel)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x


class SkipGNN(nn.Module):

    def __init__(self, in_features, out_features, bias=True, skip=True):
        super(SkipGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_skip = Parameter(torch.FloatTensor(in_features, out_features))  # skip connection
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.skip = skip

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_skip.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, norm_adj):
        support1 = torch.mm(x, self.weight)
        if self.skip:
            support2 = torch.mm(x, self.weight_skip)  # skip connections
            output = torch.spmm(norm_adj, support1) + support2
        else:
            output = torch.spmm(norm_adj, support1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads=1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (
                self.head_dim * heads == embed_dim
        ), "Embedding dimension should be divisible by number of heads"

        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, embeddings, ac):
        N, T, C = embeddings.shape  # 32*5*20

        # Split the embedding into self.heads different pieces
        # values = self.values(embeddings).view(N, T, self.heads, self.head_dim)
        values = embeddings.view(N, T, self.heads, self.head_dim)  # 32*5*1*20
        keys = self.keys(embeddings).view(N, T, self.heads, self.head_dim)  # 32*5*1*20
        queries = self.queries(embeddings).view(N, T, self.heads, self.head_dim)  # 32*5*1*20

        # Compute the attention scores
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # 32*1*5*5
        attention_scores = attention_scores / (self.embed_dim ** (1 / 2))  # 32*1*5*5
        # #对每一组注意力系数截断
        # topk_values, topk_indices = torch.topk(attention_scores, 3, dim=-1)
        # attention_mask = torch.zeros_like(attention_scores).scatter_(-1, topk_indices, 1)
        # masked_attention_scores = attention_scores * attention_mask
        # #对截断后的注意力系数进行softmax
        # attention = torch.softmax(masked_attention_scores, dim=-1)
        topk_scores, topk_indices = torch.topk(attention_scores, 3, dim=-1)
        inf_mask = torch.full_like(attention_scores, float('-inf'))
        attention = inf_mask.scatter(-1, topk_indices, topk_scores)  # 截断后32*1*5*5

        # Apply softmax to the modified attention scores
        attention = torch.softmax(attention, dim=-1)  # softmax后32*1*5*5

        # Apply attention to the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, T, self.heads * self.head_dim)

        # Pass through the final linear layer
        out = self.fc_out(out)
        return out, attention


class SelfAttention_gate(nn.Module):
    def __init__(self, embed_dim, heads=1):
        super(SelfAttention_gate, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        # assert (
        #         self.head_dim * heads == embed_dim
        # ), "Embedding dimension should be divisible by number of heads"

        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.embed_dim)  # , bias=False)
        self.queries = nn.Linear(self.embed_dim, self.embed_dim)  # , bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)
        self.gate = nn.Sequential(nn.Linear(5, 5))  # ,nn.LeakyReLU())

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.heads, self.head_dim)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, embeddings, ac):  # ac是图的同配系数

        N, T, C = embeddings.shape  # 32*5*20

        # Split the embedding into self.heads different pieces
        # values = self.values(embeddings).view(N, T, self.heads, self.head_dim)
        # values = embeddings.view(N, T, self.heads, self.head_dim) #32*5*1*20
        # keys = self.keys(embeddings).view(N, T, self.heads, self.head_dim)#32*5*1*20
        # queries = self.queries(embeddings).view(N, T, self.heads, self.head_dim)#32*5*1*20

        keys = self.trans_to_multiple_heads(self.keys(embeddings))
        queries = self.trans_to_multiple_heads(self.queries(embeddings))
        values = self.trans_to_multiple_heads(embeddings)

        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Compute the attention scores
        # attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) #32*1*5*5
        # attention_scores = attention_scores / (self.embed_dim ** (1 / 2)) #32*1*5*5

        # 应用基于同配系数的门控机制（要调试看一下这里的数据乘出来对不对）——检验正确
        ac_gate = self.gate(ac)
        # ac_gate = self.gate(ac.unsqueeze(1))
        ac_gate = ac_gate.view(N, 1, 1, 5)  # 调整形状以匹配注意力分数
        attention_scores = attention_scores * ac_gate

        # #对每一组注意力系数截断
        not_abs = True
        if not_abs:
            topk_scores, topk_indices = torch.topk(attention_scores, 3, dim=-1)
            inf_mask = torch.full_like(attention_scores, float('-inf'))
            attention = inf_mask.scatter(-1, topk_indices, topk_scores)  # 截断后32*1*5*5
        else:
            # 利用绝对值的大小截断
            abs_attention_scores = torch.abs(attention_scores)
            topk_scores, topk_indices = torch.topk(abs_attention_scores, 3, dim=-1)
            topk_original_scores = attention_scores.gather(-1, topk_indices)
            inf_mask = torch.full_like(attention_scores, float('-inf'))
            attention = inf_mask.scatter(-1, topk_indices, topk_original_scores)

        # Apply softmax to the modified attention scores
        attention = torch.softmax(attention, dim=-1)  # softmax后32*1*5*5

        # Apply attention to the values这里计算可能有问题
        # out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, T, self.heads * self.head_dim)
        context = torch.matmul(attention, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.embed_dim,)
        context = context.view(*new_size)
        # Pass through the final linear layer
        context = self.fc_out(context)
        return context, attention


class SelfAttention_gate2(nn.Module):
    def __init__(self, embed_dim, heads=1,jieduan=4,useac=True):
        super(SelfAttention_gate2, self).__init__()
        self.jieduan = jieduan
        self.useac = useac
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.keys = nn.Linear(self.embed_dim, self.embed_dim)  # , bias=False)
        self.queries = nn.Linear(self.embed_dim, self.embed_dim)  # , bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)
        if useac:
            self.gate = nn.Sequential(nn.Linear(5,6,bias=True))  # ,nn.LeakyReLU())

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.heads, self.head_dim)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, embeddings, ac, batch):  # ac是图的同配系数

        N, T, C = embeddings.shape  # 912(32个图的节点数)*6*20

        # Split the embedding into self.heads different pieces
        # values = self.values(embeddings).view(N, T, self.heads, self.head_dim)
        # values = embeddings.view(N, T, self.heads, self.head_dim) #32*5*1*20
        # keys = self.keys(embeddings).view(N, T, self.heads, self.head_dim)#32*5*1*20
        # queries = self.queries(embeddings).view(N, T, self.heads, self.head_dim)#32*5*1*20

        keys = self.trans_to_multiple_heads(self.keys(embeddings))
        queries = self.trans_to_multiple_heads(self.queries(embeddings))
        values = self.trans_to_multiple_heads(embeddings)

        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        # print(self.useac)
        if self.useac:
            ac_gate = self.gate(ac)
            ac_gate = ac_gate[batch].view(N, 1, 1, 6)  # 调整形状以匹配注意力分数
            attention_scores = attention_scores * ac_gate
        # 应用基于同配系数的门控机制（要调试看一下这里的数据乘出来对不对）——检验正确
        # ac_gate = self.gate(ac)
        #法一
        # num_nodes_per_graph = torch.bincount(batch).to(ac.device)
        #
        # # 为每个节点分配对应图的ac
        # prob_tensor_expanded = torch.zeros((attention_scores.size(0), 6)).to(ac.device)
        # for graph_id in range(len(num_nodes_per_graph)):
        #     # 获取当前图的节点索引范围
        #     node_indices = (batch == graph_id).nonzero(as_tuple=True)[0].to(ac.device)
        #     # 将当前图的概率分配给所有节点
        #     prob_tensor_expanded[node_indices] = ac_gate[graph_id]
        # ac_gate = prob_tensor_expanded.view(N, 1, 1, 6)  # 调整形状以匹配注意力分数
        #法二
        # ac_gate = ac_gate[batch].view(N, 1, 1, 6)  # 调整形状以匹配注意力分数
        # ac_gate = torch.softmax(ac_gate, dim=-1)

        # attention_scores = attention_scores * ac_gate

        # #对每一组注意力系数截断
        not_abs = False
        if not_abs:
            topk_scores, topk_indices = torch.topk(attention_scores, self.jieduan, dim=-1)
            inf_mask = torch.full_like(attention_scores, float('-inf'))
            attention = inf_mask.scatter(-1, topk_indices, topk_scores)  # 截断后32*1*5*5
        else:
            # 利用绝对值的大小截断
            abs_attention_scores = torch.abs(attention_scores)
            topk_scores, topk_indices = torch.topk(abs_attention_scores, self.jieduan, dim=-1)
            topk_original_scores = attention_scores.gather(-1, topk_indices)
            inf_mask = torch.full_like(attention_scores, float('-inf'))
            attention = inf_mask.scatter(-1, topk_indices, topk_original_scores)

        # Apply softmax to the modified attention scores
        attention = torch.softmax(attention, dim=-1)  # softmax后32*1*5*5

        # Apply attention to the values这里计算可能有问题
        # out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, T, self.heads * self.head_dim)
        context = torch.matmul(attention, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.embed_dim,)
        context = context.view(*new_size)
        # Pass through the final linear layer
        # context = self.fc_out(context)
        return context, attention


class ODS_Layer(nn.Module):
    def __init__(self, nhid1, nhid2, nhid3, h_concat_ac=False):
        super(ODS_Layer, self).__init__()
        self.C = nhid3
        # self.gcn = GCN_Net(nhid1, nhid2, nhid3)
        # self.sage = GraphSAGE_Net(nhid1, nhid2, nhid3)
        # self.gat = GAT_Net(nhid1, nhid2, nhid3)
        # self.gin = GIN_Net(nhid1, nhid2, nhid3)
        # self.lin = nn.Sequential(nn.Linear(nhid1, nhid2), torch.nn.BatchNorm1d(nhid2), nn.ReLU(),
        #                          nn.Linear(nhid2, nhid3))
        self.gcn = GCN_Net_ASAP(nhid1, nhid2, nhid3)
        self.sage = GraphSAGE_Net_ASAP(nhid1, nhid2, nhid3)
        self.gat = GAT_Net_ASAP(nhid1, nhid2, nhid3)
        self.gin = GIN_Net_ASAP(nhid1, nhid2, nhid3)
        self.lin = Linear_Net_ASAP(nhid1, nhid2, nhid3)
        if h_concat_ac:
            self.att_layer = SelfAttention(nhid3 + 1)
        else:
            self.att_layer = SelfAttention_gate(nhid3)

        # self.pool1 = SAGPooling(nhid3, ratio=0.5)
        # self.pool2 = SAGPooling(nhid3, ratio=0.5)
        # self.pool3 = SAGPooling(nhid3, ratio=0.5)
        # self.pool4 = SAGPooling(nhid3, ratio=0.5)
        # self.pool5 = SAGPooling(nhid3, ratio=0.5)

    def forward(self, x, edge_index, batch, ac, h_concat_ac=False):
        # 可以改成其他pooling试试
        # gcnout = global_mean_pool(self.gcn(x, edge_index), batch)
        # sageout = global_mean_pool(self.sage(x, edge_index), batch)
        # gatout = global_mean_pool(self.gat(x, edge_index), batch)
        # linout = global_mean_pool(self.lin(x), batch)
        # ginout = global_mean_pool(self.gin(x, edge_index), batch)

        # gcn_x, gcn_edge_index,  _,gcn_batch, _, _ = self.pool1(self.gcn(x, edge_index), edge_index, None, batch)
        # gcnout = global_mean_pool(gcn_x, gcn_batch)
        # sage_x, sage_edge_index,  _, sage_batch, _, _ = self.pool2(self.sage(x, edge_index), edge_index, None, batch)
        # sageout = global_mean_pool(sage_x, sage_batch)
        # gat_x, gat_edge_index, _, gat_batch, _ , _= self.pool3(self.gat(x, edge_index), edge_index, None, batch)
        # gatout = global_mean_pool(gat_x, gat_batch)
        # gin_x, gin_edge_index, _, gin_batch, _, _ = self.pool4(self.gin(x, edge_index), edge_index, None, batch)
        # ginout = global_mean_pool(gin_x, gin_batch)
        # lin_x, lin_edge_index, _, lin_batch, _, _ = self.pool5(self.lin(x), edge_index, None, batch)
        # linout = global_mean_pool(lin_x, lin_batch)

        gcnout = self.gcn(x, edge_index, batch)
        sageout = self.sage(x, edge_index, batch)
        gatout = self.gat(x, edge_index, batch)
        linout = self.lin(x, edge_index, batch)
        ginout = self.gin(x, edge_index, batch)

        embeddings = torch.stack([gcnout, sageout, gatout, linout, ginout], dim=1)

        # 正交性dpp
        dpp_losses = []
        for i in range(embeddings.shape[0]):  # 遍历每个嵌入
            emb = embeddings[i]  # 获取第i个嵌入
            norm_emb = emb / emb.norm(dim=1, keepdim=True)
            gram_matrix = norm_emb @ norm_emb.T  # 计算嵌入和它的转置的乘积
            det = torch.logdet(gram_matrix)  # 计算行列式并取对数
            dpp_losses.append(det)
        dpp_loss = torch.stack(dpp_losses)

        # h最后一维度拼接同配系数
        if h_concat_ac:
            gcnout = torch.cat((gcnout, ac.unsqueeze(1)), dim=1)
            sageout = torch.cat((sageout, ac.unsqueeze(1)), dim=1)
            gatout = torch.cat((gatout, ac.unsqueeze(1)), dim=1)
            linout = torch.cat((linout, ac.unsqueeze(1)), dim=1)
            ginout = torch.cat((ginout, ac.unsqueeze(1)), dim=1)
            embeddings = torch.stack([gcnout, sageout, gatout, linout, ginout], dim=1)

        # self attention
        output_5att, attention_weights = self.att_layer(embeddings, ac)
        out = output_5att.sum(dim=1)
        out = F.relu(out)
        out = F.dropout(out, training=self.training)
        # 可以改成拼接3个向量
        # out = output_5att.view(batch.max().item() + 1,-1)

        # top_attentions, indices = torch.topk(attention_weights.squeeze(0), 3, dim=2)
        # top_out = torch.einsum("nhq,nhd->nhd", [top_attentions, embeddings.squeeze(0)])
        # top_out = top_out.sum(dim=1)
        # sum = p[0] * gcnout + p[1] * sageout + p[2] * gatout + p[3] * linout  # +p[4]*ginout
        # sum = (p[:, 0] * (gcnout.T)).T + (p[:, 1] * (sageout.T)).T + (p[:, 2] * (gatout.T)).T + (p[:, 3] * (linout.T)).T
        # sum = sum + x
        # sum = torch.hstack(((p[:, 0] * (gcnout.T)).T , (p[:, 1] * (sageout.T)).T , (p[:, 2] * (gatout.T)).T , (p[:, 3] * (linout.T)).T))
        # # x = F.relu(sum)
        # x = F.dropout(sum, training=self.training)
        # x = self.lin2(sum)
        return out, dpp_loss


class ODS_Layer2(nn.Module):
    def __init__(self, nhid1, nhid2, nhid3, h_concat_ac=False,jieduan=4,useac=True):
        super(ODS_Layer2, self).__init__()
        self.C = nhid3
        self.gcn1 = GCNConv(nhid1, nhid2)
        self.sage1 = SAGEConv(nhid1, nhid2)
        self.gat1 = GATConv(nhid1, int(0.5 * nhid2), heads=2)
        self.gin1 = GINConv(nn=nn.Sequential(nn.Linear(nhid1, nhid2), nn.ReLU(), nn.Linear(nhid2, nhid2)), eps=1e-9)
        self.lin1 = nn.Linear(nhid1, nhid2)
        self.graph1 = GraphConv(nhid1, nhid2)
        self.pool1 = ASAPooling(nhid2, ratio=0.5)
        self.att_layer1 = SelfAttention_gate2(nhid2,jieduan=jieduan,useac=useac)

        self.gcn2 = GCNConv(nhid2, nhid3)
        self.sage2 = SAGEConv(nhid2, nhid3)
        self.gat2 = GATConv(nhid2, int(0.5 * nhid3), heads=2)
        self.gin2 = GINConv(nn=nn.Sequential(nn.Linear(nhid2, nhid3), nn.ReLU(), nn.Linear(nhid3, nhid3)), eps=1e-13)
        self.lin2 = nn.Linear(nhid2, nhid3)
        self.pool2 = ASAPooling(nhid3, ratio=0.5)
        self.graph2 = GraphConv(nhid2, nhid3)
        self.att_layer2 = SelfAttention_gate2(nhid3,jieduan=jieduan,useac=useac)

        self.lin_final = nn.Linear(nhid3, nhid3)

        self.virtualnode_embedding = nn.Embedding(1, nhid1)  # 初始化虚拟节点嵌入
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)  # 虚拟节点权重初始化为0
        self.virtualnode_linear = nn.ModuleList()
        for i in range(6):
            self.virtualnode_linear.append(nn.Sequential(nn.Linear(nhid1, nhid1)))#,nn.BatchNorm1d(nhid1), nn.ReLU()))

    def update_virtualnode_embedding(self, x, batch, virtualnode_embedding, i):
        # 使用全局平均池化更新虚拟节点嵌入
        global_mean = global_mean_pool(x, batch)

        # 将全局平均池化的结果通过一个线性层,更新虚拟节点嵌入
        virtualnode_embedding = self.virtualnode_linear[i](global_mean+ virtualnode_embedding)

        return virtualnode_embedding

    def dpp_loss(self,embeddings):
        # 归一化嵌入
        norm_embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)
        # 计算所有嵌入的Gram矩阵
        gram_matrices = torch.matmul(norm_embeddings, norm_embeddings.transpose(1, 2))
        # 计算对数行列式
        dpp_losses = torch.logdet(gram_matrices)
        return dpp_losses


    def dpp_loss_nystrom(self,embeddings,ratio=0.1):
        # 选择样本数量m
        m = math.ceil(embeddings.size(0)*ratio)
        #emb模值为1
        embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)
        # 随机选择m个样本的索引
        indices = torch.randperm(embeddings.size(0))[:m]

        # 构建C和W
        representatives  = embeddings[indices]  # 选中的样本
        emb_flat = embeddings.view(embeddings.size(0),-1)#600*120
        representatives_flat = representatives.view(m,-1)#60*120
        # 计算C矩阵（代表样本之间的相似度）
        C = torch.matmul(representatives, representatives.transpose(1, 2))#6*6
        W = torch.softmax(torch.matmul(emb_flat, representatives_flat.T),dim=1)#600*60
        dpp_losses =torch.logdet(torch.einsum('ij,jkl->ikl', W, C) )
        # C_inv = torch.inverse(C)
        # 近似核矩阵K
        # dpp_losses = torch.bmm(W.unsqueeze(1))
        # K_approx = torch.matmul(torch.matmul(W, C), W.T)
        # dpp_losses = torch.logdet(K_approx+ torch.eye(600) * 1e-6)
        # dpp_losses = dpp_losses/embeddings.size(0)

        # norm_C = C / C.norm(dim=2, keepdim=True)
        # W = torch.matmul(norm_C, norm_C.transpose(1, 2))  # W是C的Gram矩阵
        #
        # # 计算W的逆
        # W_inv = torch.inverse(W)
        #
        # # 计算近似核矩阵
        # norm_embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)
        # C_full = torch.matmul(norm_embeddings, norm_C.transpose(1, 2))
        # K_tilde = torch.matmul(torch.matmul(C_full, W_inv), C_full.transpose(1, 2))
        #
        # # 使用近似核矩阵计算DPP损失
        # dpp_losses = torch.logdet(K_tilde)# + 1e-6 * torch.eye(K_tilde.size(-1)))  # 防止数值问题

        return dpp_losses


    def forward(self, x, edge_index, batch, ac, h_concat_ac=False):
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(dtype=torch.long).to(device=edge_index.device))
        origin = x
        # x = x + virtualnode_embedding[batch]
        virtualnode_embedding = self.update_virtualnode_embedding(origin, batch, virtualnode_embedding, 0)
        x=x+virtualnode_embedding[batch]
        gcnout = self.gcn1(x, edge_index)
        # virtualnode_embedding = self.update_virtualnode_embedding(origin, batch, virtualnode_embedding,0)
        # sageout = self.sage1(gcnout+virtualnode_embedding[batch], edge_index)
        sageout = self.sage1(x, edge_index)
        # virtualnode_embedding = self.update_virtualnode_embedding(gcnout, batch, virtualnode_embedding,1)
        #
        # gatout = self.gat1(sageout+virtualnode_embedding[batch], edge_index)
        gatout = self.gat1(x, edge_index)
        # virtualnode_embedding = self.update_virtualnode_embedding(sageout, batch, virtualnode_embedding,2)
        #
        # linout = self.lin1(gatout+virtualnode_embedding[batch])
        linout = self.lin1(x)
        # virtualnode_embedding = self.update_virtualnode_embedding(gatout, batch, virtualnode_embedding,3)
        #
        # ginout = self.gin1(linout+virtualnode_embedding[batch], edge_index)
        ginout = self.gin1(x, edge_index)
        # virtualnode_embedding = self.update_virtualnode_embedding(linout, batch, virtualnode_embedding,4)
        #
        # graphout = self.graph1(ginout+virtualnode_embedding[batch],edge_index)
        graphout = self.graph1(x, edge_index)
        # virtualnode_embedding = self.update_virtualnode_embedding(ginout, batch, virtualnode_embedding,5)

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
        #————————————————————————————————————————————————————————————————————————————————————————
        # # 归一化嵌入
        # norm_embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)
        # # 计算所有嵌入的Gram矩阵
        # gram_matrices = torch.matmul(norm_embeddings, norm_embeddings.transpose(1, 2))
        # # 计算对数行列式
        # dpp_losses1 = torch.logdet(gram_matrices)
        #——————————————————————————————————————————————————————————————————————————————————————————
        # dpp_losses1 = self.dpp_loss_nystrom(embeddings)

        # self attention
        output1, attention_weights1 = self.att_layer1(embeddings, ac, batch)
        output1 = output1.sum(dim=1)
        # output1 = F.relu(output1)
        # output1 = F.dropout(output1, training=self.training)

        virtualnode_embedding = self.update_virtualnode_embedding(origin, batch, virtualnode_embedding, 1)

        x, edge_index, _, batch, _ = self.pool1(output1, edge_index, None, batch)

        x = x+virtualnode_embedding[batch]

        gcnout2 = self.gcn2(x, edge_index)
        sageout2 = self.sage2(x, edge_index)
        gatout2 = self.gat2(x, edge_index)
        linout2 = self.lin2(x)
        ginout2 = self.gin2(x, edge_index)
        graphout2 = self.graph2(x, edge_index)

        embeddings = torch.stack([gcnout2, sageout2, gatout2, linout2, ginout2, graphout2], dim=1)

        # 正交性dpp
        # dpp_losses2 = []
        # for i in range(embeddings.shape[0]):  # 遍历每个嵌入
        #     emb = embeddings[i]  # 获取第i个嵌入
        #     norm_emb = emb / emb.norm(dim=1, keepdim=True)
        #     gram_matrix = norm_emb @ norm_emb.T  # 计算嵌入和它的转置的乘积
        #     det = torch.logdet(gram_matrix)  # 计算行列式并取对数
        #     dpp_losses2.append(det)
        # dpp_losses2 = torch.stack(dpp_losses2)
        # ——————————————————————————————————————————————————————————————————————————————————————————
        # # 归一化嵌入
        # norm_embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)
        # # 计算所有嵌入的Gram矩阵
        # gram_matrices = torch.matmul(norm_embeddings, norm_embeddings.transpose(1, 2))
        # # 计算对数行列式
        # dpp_losses2 = torch.logdet(gram_matrices)
        #——————————————————————————————————————————————————————————————————————————————————————————
        # dpp_losses2 = self.dpp_loss_nystrom(embeddings)

        # self attention
        output1, attention_weights2 = self.att_layer2(embeddings, ac, batch)
        output1 = output1.sum(dim=1)
        # output1 = F.relu(output1)
        # output1 = F.dropout(output1, training=self.training)
        x, edge_index, _, batch, _ = self.pool2(output1, edge_index, None, batch)
        x = self.lin_final(x)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        out = global_mean_pool(x, batch)
        return out, 0,0,  attention_weights1,attention_weights2#torch.sum(dpp_losses1)/len(dpp_losses1),torch.sum(dpp_losses2)/len(dpp_losses2) # torch.cat((dpp_losses1,dpp_losses2),dim=0)


class GSACN_Net(nn.Module):  # graph search and choose network
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nout,jieduan,useac, h_concat_ac=False):
        super(GSACN_Net, self).__init__()

        self.conv1 = GCNConv(nfeat, nhid1)
        self.lin1 = nn.Linear(nfeat, nhid1)
        self.h_concat_ac = h_concat_ac
        self.ods_layer1 = ODS_Layer2(nhid1, nhid2, nhid3,jieduan=jieduan,useac=useac, h_concat_ac=self.h_concat_ac)
        if h_concat_ac:
            nhid3 = nhid3 + 1
        self.conv2 = GCNConv(nhid3, nhid3)
        self.lin = nn.Linear(nhid3, nout)

    def forward(self, data):
        x = data.x
        x=x.float()
        edge_index = data.edge_index
        batch = data.batch
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin1(x)
        x = F.relu(x)
        ac = calculate_ac_for_batch(data)
        # x, dpp_loss1,dpp_loss2,attention_weights1,attention_weights2 = self.ods_layer1(x, edge_index, batch, ac, h_concat_ac=self.h_concat_ac)
        x, dpp_loss1, dpp_loss2,attention_weights1,attention_weights2 = self.ods_layer1(x, edge_index, batch, ac,h_concat_ac=self.h_concat_ac)
        x = self.lin(x)
        return x, dpp_loss1,dpp_loss2,attention_weights1,attention_weights2
