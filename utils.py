import math
import torch
import random
import numpy as np
import networkx as nx
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv
from deepsnap.graph import Graph
from torch_geometric.utils import to_dense_adj, dense_to_sparse, from_networkx, add_self_loops, to_networkx, degree
import copy
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法


# Calcualte the balanced_constraint_index of a subgraph (room) assignment.
def balanced_constraint_index(num_room, room_size, room):
    """
    Calcualte the balanced_constraint_index of a subgraph (room) assignment.

    Parameters
    ----------
    num_room : int
           The number of rooms (subgraphs).

    room_size : int
            The size (capacity) of rooms (subgraphs).

    room : scalar value, dict-like
           A dictionary of the subgraph (room) assignment. key is the room, value is the nodes.

    Returns
    -------
    eps : float.
          The maximum ratio of real load to room size

    bci : float.
          The balanced constraint index.
    """
    eps, bci = 0, 0
    for k, v in room.items():
        n_nodes = len(v)
        if len(v) / room_size > eps:
            eps = len(v) / room_size
        bci += np.absolute((len(v) - room_size) / room_size)
    return eps, bci / len(room)


def unbalanced_constraint_index(num_node, r_k, room):
    """
    Calcualte the balanced_constraint_index of a subgraph (room) assignment.

    Parameters
    ----------
    num_room : int
           The number of rooms (subgraphs).

    room_size : int
            The size (capacity) of rooms (subgraphs).

    room : scalar value, dict-like
           A dictionary of the subgraph (room) assignment. key is the room, value is the nodes.

    Returns
    -------
    eps : float.
          The maximum ratio of real load to room size

    bci : float.
          The balanced constraint index.
    """
    eps, bci = 0, 0
    for k, v in room.items():
        n_nodes = len(v)
        # if len(v) / torch.tensor(r_k) > eps:
        #     eps = len(v) / torch.tensor(r_k)
        bci += np.absolute((len(v) - num_node * r_k[k]) / (num_node * r_k[k]))
    return bci / len(room)


# Get hard node assignments from the soft node assignment s
def soft2hard(s, device='cuda:0'):
    """
    Get hard node assignments from the soft node assignment s
    """
    hard = torch.zeros(s.shape[0], s.shape[1])
    ones = torch.ones(s.shape[0], s.shape[1])
    label = torch.max(s.cpu(), 1, True)
    c = hard.scatter_(1, torch.LongTensor(label.indices), ones).to(device)
    return c


# Get the room assignment dictionary from the hard node assignment c
def hard2room(c):
    """
    Get the room assignment dictionary from the hard node assignment c
    """
    K = c.shape[1]
    room = {}
    for i in range(K):
        room[i] = torch.nonzero(c[:, i]).view(-1, ).tolist()
    return room


def plot_cora_hard_assignment(data, c, max_nodes=500, use_tsne=True):
    """
    绘制前 max_nodes 个节点，按硬分配矩阵 c 的社区划分进行着色的散点图。
    如果 use_tsne=True，则对前 max_nodes 个节点的特征 x 做 TSNE 降维到 2D；
    若已有 2D 坐标，也可直接传入 (比如 data.x 就是 2D)。
    """

    # 仅使用前 max_nodes 个节点
    x_data = data.x[:max_nodes]

    # 若需要降维，这里以 TSNE 为例
    if use_tsne:
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(x_data.cpu().numpy())
    else:
        # 如果 data.x 本身就是 2D，可以直接用
        X_embedded = x_data.cpu().numpy()

    # 获取硬分配字典: {社区编号: [节点编号列表]}
    room_dict = hard2room(c)

    # 这里我们给每个社区指定一种颜色；也可以使用 plt.cm 下的各种 colormap
    color_list = [
        "red", "blue", "green", "orange", "purple",
        "pink", "brown", "olive", "cyan", "black",
        # 如果社区大于10，可继续添加更多颜色
    ]

    plt.figure(figsize=(8, 6))

    # 遍历每个社区，并在散点图中绘制
    for k in range(c.shape[1]):
        node_list = [n for n in room_dict[k] if n < max_nodes]
        if len(node_list) == 0:
            continue

        # 取出这些节点的二维坐标
        x_coords = X_embedded[node_list, 0]
        y_coords = X_embedded[node_list, 1]

        # 绘制该社区下节点的散点
        plt.scatter(
            x_coords, y_coords,
            c=color_list[k % len(color_list)],
            label=f"Community {k}",
            alpha=0.8, s=20
        )

    plt.title(f"Cora Hard Node Assignment (first {max_nodes} nodes)")
    plt.legend()
    plt.show()


def number_inner_edges(s, adj, device):
    c = soft2hard(s, device=device)
    # K = c.shape[1]
    # in_edges = 0
    # for i in range(K):
    #     in_edges += c[:, i].t() @ torch.mv(adj, c[:, i])
    in_edges = (c.t() @ torch.sparse.mm(adj, c)).trace()
    return (1 / 2) * in_edges


def plot_pyg_graph_with_communities(data, c, name,use_tsne=True):
    """
    Args:
        data: PyG中的Data对象，包含edge_index和x等
        c: 形如 [N, K] 的硬分配矩阵，每行只有一个元素为1，其余为0
           c[i, j]=1 表示节点 i 属于社区 j
        use_tsne: 是否对节点特征进行TSNE降维
    """

    # 1. 从硬分配矩阵c中提取每个节点所属的社区ID（argmax即可）
    communities = c.argmax(dim=1).tolist()  # [N]，communities[i]是第i个节点的社区ID

    # 2. 转为NetworkX图（便于可视化）
    #    to_undirected=True 可根据需要选择是否无向化；Cora本身是有向图也可以
    data_simple = Data(
        x=data.x,
        edge_index=data.edge_index,  # 普通的 [2, E] 张量
        y=data.y
        # 其他需要的属性也可以加
    )
    G = to_networkx(data_simple, to_undirected=True)

    # 3. 获取或计算节点在2D平面上的坐标
    #    这里以TSNE为例，如果 data.x 本身已经是2D，就不需要再降维
    if use_tsne:
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(data.x.cpu().numpy())
    else:
        X_embedded = data.x[:, :2].cpu().numpy()  # 如果x本身是2D，比如用PCA/其他方式处理好的

    # 将坐标存到 pos 这个字典，NetworkX画图需要 {节点ID: (x, y)}
    pos = {i: (X_embedded[i, 0], X_embedded[i, 1]) for i in range(data.num_nodes)}

    # 4. 生成节点的颜色
    #    你可以换成更丰富的颜色表，这里示意给出一些
    color_list = [
        "#483D8B",  # 深紫
        "#00CED1",  # 浅蓝
        "#FF4500",  # 橘红
        "#FFD700",  # 金黄
        "#AABB99",  # 浅灰绿
        "#8B008B",  # 紫色
        "#ADFF2F",  # 亮绿色
        "#FA8072",  # 鲑鱼红
        "#1E90FF",  # 道奇蓝
        "#FF1493",  # 深粉红
    ]
    node_colors = [color_list[comm % len(color_list)] for comm in communities]

    # 5. 区分同一社区的边 和 跨社区的边
    same_community_edges = []
    cross_community_edges = []
    for u, v in G.edges():
        # 如果u和v的社区ID相同，就归为同社区边，否则跨社区边
        if communities[u] == communities[v]:
            same_community_edges.append((u, v))
        else:
            cross_community_edges.append((u, v))

    # 6. 开始绘图
    plt.figure(figsize=(8, 8))

    # 6.1 先画节点
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=120,
        linewidths=0.5,
        edgecolors="white",  # 节点边框颜色，可选
    )

    # 6.2 同社区的边，alpha=0.5
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=same_community_edges,
        edge_color="grey",
        alpha=0.5,
        width=1.0,
    )

    # 6.3 跨社区的边，alpha=0.1
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cross_community_edges,
        edge_color="grey",
        alpha=0.1,
        width=1.0,
    )

    # 6.4 隐藏坐标轴并显示
    plt.axis("off")
    # plt.title("PyG Graph with Communities (same-community edges darker)")
    fig_name = name+'.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')  # 保存图像

    plt.show()

# Get sparse identity matrix.
def get_sparse_eye(data):
    """
    Get sparse identity matrix.
    """
    num_nodes = data.num_nodes
    edge_weight = torch.ones((num_nodes,))
    node_index = [i for i in range(num_nodes)]
    indices = torch.from_numpy(np.vstack((node_index, node_index)).astype(np.int64))
    values = torch.from_numpy(edge_weight.numpy())
    shape = torch.Size((num_nodes, num_nodes))
    return torch.sparse.FloatTensor(indices, values, shape)


# Get sparse adjacency matrix of the graph.
def get_sparse_adj(data):
    """
    Get sparse adjacency matrix of the graph.
    """
    num_nodes = data.num_nodes
    edge_weight = torch.ones((data.edge_index.size(1),)).to(data.edge_index.device)
    row, col = data.edge_index[0], data.edge_index[1]
    indices = torch.from_numpy(np.vstack((row.cpu().numpy(), col.cpu().numpy())).astype(np.int64))
    values = torch.from_numpy(edge_weight.cpu().numpy())
    shape = torch.Size((num_nodes, num_nodes))
    return torch.sparse.FloatTensor(indices, values, shape)


# Get sparse normalized adjcent matrix of the graph.
def get_sparse_norm_adj(data):
    """
    Get sparse normalized adjcent matrix of the graph.
    """
    num_nodes = data.num_nodes
    edge_weight = torch.ones((data.edge_index.size(1),))
    row, col = data.edge_index[0], data.edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=data.num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    indices = torch.from_numpy(np.vstack((data.edge_index[0], data.edge_index[1])).astype(np.int64))
    values = torch.from_numpy(edge_weight.numpy())
    shape = torch.Size((num_nodes, num_nodes))
    return torch.sparse.FloatTensor(indices, values, shape)


# Calculate the loss of graph partitioning (cut loss and reg loss).
def gp_loss(s, adj):
    """
    Calculate the loss of graph partitioning (cut loss and reg loss).

    Parameters
    ----------
    s : dense tensor (N × K)
        The soft node assignment matrix s.

    adj : sparse adjecent matrix (N × N)
        The sparse adjecent matrix of the graph.

    deg : vector (N × 1)
          the degree vector of all nodes.

    Returns
    -------
    cut_loss : scalar value
               The cut loss.

    reg_loss : scalar value
               The regularization loss.
    """
    in_edges = s.t() @ torch.sparse.mm(adj, s)  # number of inner edges
    cut_loss = (1. / torch.sparse.sum(adj)) * in_edges.trace()  # cut loss term
    reg_loss = (math.sqrt(s.shape[1]) / s.shape[0]) * torch.norm(torch.sum(s, dim=0)) - 1  # regularization loss term

    # Orthogonality regularization.
    # ss = torch.matmul(s.t(), s)
    # i_s = torch.eye(s.shape[1]).type_as(ss)
    # ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
    # reg_loss = torch.mean(ortho_loss)

    return cut_loss, reg_loss


def ubgp_loss(s, adj, r_k):
    in_edges = s.t() @ torch.sparse.mm(adj, s)  # number of inner edges
    cut_loss = (1. / torch.sparse.sum(adj)) * in_edges.trace()  # cut loss term
    ub_reg_loss = torch.norm(torch.sum(s, dim=0) / s.sum() - r_k)  # regularization loss term
    return cut_loss, ub_reg_loss


def ubgp_loss2(s, adj, r_k):
    in_edges = s.t() @ torch.sparse.mm(adj, s)  # number of inner edges
    cut_loss = (1. / torch.sparse.sum(adj)) * in_edges.trace()  # cut loss term
    # reg_loss = (math.sqrt(s.shape[1]) / s.shape[0]) * torch.norm(torch.sum(s, dim=0)) - 1  # balanced regularization loss term
    ub_reg_loss = torch.mean((torch.sum(s, dim=0) / s.sum() - r_k) ** 2)

    # ub_reg_loss = torch.norm(torch.sum(s, dim=0) - r_k)
    return cut_loss, ub_reg_loss


def mincut_loss(s, adj):
    in_edges = s.t() @ torch.sparse.mm(adj, s)  # number of inner edges
    cut_loss = (1. / torch.sparse.sum(adj)) * in_edges.trace()  # cut loss term
    # 计算 S^T * S
    ST_S = torch.matmul(s.T, s)
    # 计算 Frobenius 范数
    norm_ST_S = torch.norm(ST_S, p='fro')
    K = s.shape[1]
    # 计算单位矩阵 I_k
    I_k = torch.eye(K, device=s.device)
    # 计算 L_o
    L_o = torch.norm(ST_S / norm_ST_S - I_k / torch.sqrt(torch.tensor(K, device=s.device, dtype=torch.float)), p='fro')
    return cut_loss, L_o


def gp_loss2(s, adj, deg):
    in_edges = s.t() @ torch.sparse.mm(adj, s)  # number of inner edges
    cut_loss = in_edges.trace()  # cut loss term
    reg_loss = (np.sqrt(s.shape[1]) / s.shape[0]) * torch.norm(torch.sum(s, dim=0)) - 1  # regularization loss term
    return cut_loss, reg_loss


def gp_loss3(s, adj, deg):
    """
    Why does this way become slow?
    """
    N, K = s.shape[0], s.shape[1]
    in_edges = 0
    for i in range(K):
        in_edges += s[:, i].t() @ torch.mv(adj, s[:, i])

    cut_loss = (1. / torch.sparse.sum(adj)) * in_edges  # cut loss term
    reg_loss = (math.sqrt(K) / N) * torch.norm(torch.sum(s, dim=0)) - 1  # regularization loss term
    return cut_loss, reg_loss


def ubgp_loss_dense(s, edge_index, r_k, batch):
    # 初始化累积损失
    total_cut_loss = 0.0
    total_reg_loss = 0.0

    # batch中图的数量
    num_graphs = batch.max().item() + 1

    # 计算每个图的损失
    for graph_id in range(num_graphs):
        # 获取当前图的节点索引
        mask = batch == graph_id
        s_graph = s[mask]

        # 选择当前图的边
        edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
        edge_index_graph = edge_index[:, edge_mask]

        # # 为每个节点构建局部索引映射
        # unique_nodes = torch.unique(edge_index_graph)
        # local_index_map = {node.item(): idx for idx, node in enumerate(unique_nodes)}
        #
        # # 将全局节点索引转换为局部索引
        # local_edge_index = torch.stack([
        #     torch.tensor([local_index_map[node.item()] for node in edge_index_graph[0]], dtype=torch.long),
        #     torch.tensor([local_index_map[node.item()] for node in edge_index_graph[1]], dtype=torch.long)
        # ], dim=0)
        local_edge_index = edge_index_graph - edge_index_graph.min()

        # 计算邻接矩阵（稠密形式）
        num_nodes = s_graph.shape[0]
        adj_graph = torch.zeros((num_nodes, num_nodes), device=s.device, dtype=torch.float)
        adj_graph[local_edge_index[0], local_edge_index[1]] = 1

        # 计算内部边
        in_edges = s_graph.t() @ (adj_graph @ s_graph)  # number of inner edges using dense matrix multiplication

        # 计算cut_loss
        total_edges = adj_graph.sum()
        if total_edges > 0:
            cut_loss = (1. / total_edges) * in_edges.trace()  # cut loss term
        else:
            cut_loss = 0  # 如果没有边，则cut_loss为0

        # 计算regularization loss
        ub_reg_loss = torch.norm(torch.sum(s_graph, dim=0) / s_graph.sum() - r_k)  # regularization loss term

        # 累加每个图的损失
        total_cut_loss += cut_loss
        total_reg_loss += ub_reg_loss

    return total_cut_loss, total_reg_loss


def ubgp_loss_sprase(s, edge_index, r_k, batch):
    # 初始化累积损失
    total_cut_loss = 0.0
    total_reg_loss = 0.0

    # batch中图的数量
    num_graphs = batch.max().item() + 1

    # 计算每个图的损失
    for graph_id in range(num_graphs):
        # 获取当前图的节点索引
        mask = batch == graph_id
        s_graph = s[mask]

        # 选择当前图的边
        edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
        edge_index_graph = edge_index[:, edge_mask]
        edge_index_graph = edge_index_graph - edge_index_graph.min()

        # 计算邻接矩阵
        num_nodes = s_graph.shape[0]
        adj_graph = torch.sparse_coo_tensor(edge_index_graph,
                                            torch.ones(edge_index_graph.shape[1]).to(edge_index.device),
                                            (num_nodes, num_nodes))

        # 计算内边
        in_edges = s_graph.t() @ torch.sparse.mm(adj_graph, s_graph)  # number of inner edges

        # 计算cut_loss
        if torch.sparse.sum(adj_graph) > 0:
            cut_loss = (1. / torch.sparse.sum(adj_graph)) * in_edges.trace()  # cut loss term
        else:
            cut_loss = 0  # 如果没有边，则cut_loss为0

        # 计算regularization loss
        ub_reg_loss = torch.norm(torch.sum(s_graph, dim=0) / s_graph.sum() - r_k)  # regularization loss term

        # 累加每个图的损失
        total_cut_loss += cut_loss
        total_reg_loss += ub_reg_loss

    return total_cut_loss, total_reg_loss


def calculate_ac_for_single(data, num_nodes, no0=0, norm=True):
    edge_index = data.edge_index

    # if edge_index.size(1) == 0 or data.x.size(0)==1:  # 没有边或只有一个节点
    #     ac_list.append([no0, no0, no0, no0, no0])
    # 节点映射
    # unique_nodes, new_node_indices = torch.unique(edge_index, return_inverse=True)
    # graph_edge_index_remap = new_node_indices.view_as(edge_index)

    # 计算当前子图的节点的度
    deg = degree(edge_index[0], num_nodes=num_nodes)

    # 创建一个数组来保存每个边的度对
    deg_pairs = torch.stack((deg[edge_index[0]], deg[edge_index[1]]), dim=0)

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
    G = to_networkx(data=data, to_undirected=True)
    # 全图的平均聚类系数和所有的聚类系数
    cluster = nx.clustering(G)
    cc = sum(cluster.values()) / len(cluster)
    pagerank = nx.pagerank(G)
    k_core = nx.core_number(G)

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
    ac_tensor = torch.tensor([mean_deg_product + no0, mean_deg + no0, mean_deg_sq + no0, ac + no0, cc + no0],
                             dtype=torch.float, device=edge_index.device)
    deg_tensor = torch.tensor(deg, dtype=torch.float, device=edge_index.device).unsqueeze(1)
    cluster_tensor = torch.tensor(list(cluster.values()), dtype=torch.float, device=edge_index.device).unsqueeze(1)
    pagerank_tensor = torch.tensor(list(pagerank.values()), dtype=torch.float, device=edge_index.device).unsqueeze(1)
    k_core_tensor = torch.tensor(list(k_core.values()), dtype=torch.float, device=edge_index.device).unsqueeze(1)
    ac_tensor = ac_tensor.repeat(num_nodes, 1)
    gate = torch.cat((deg_tensor, cluster_tensor, pagerank_tensor, k_core_tensor, ac_tensor), dim=1)
    # gate = torch.cat((deg_tensor, cluster_tensor, pagerank_tensor, ac_tensor), dim=1)
    return gate


def generate_features_from_nmf(data, n_components, num_node):
    """
    使用NMF为图生成节点特征。
    参数:
        data (torch_geometric.data.Data): PyTorch Geometric 图数据对象。
        n_components (int): NMF分解中的组件数量。
    返回:
        更新了节点特征的data对象。
    """
    # 将 edge_index 转换成邻接矩阵
    adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=num_node)[0]

    # 应用 NMF
    model_nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    features = model_nmf.fit_transform(adj_matrix.numpy())

    # 将生成的特征保存回图中
    data.x = torch.tensor(features, dtype=torch.float)

    return data


def graph_gen(graph_name, num_node, NMF_feature_num=20):
    if (graph_name == "WS"):
        G_test = nx.watts_strogatz_graph(num_node, 2, 0.1, seed=256)
    elif (graph_name == "ER"):
        G_test = nx.erdos_renyi_graph(num_node, 0.001, seed=256)
    elif (graph_name == "RR"):
        G_test = nx.random_regular_graph(10, num_node, seed=256)
    elif (graph_name == "BA"):
        G_test = nx.barabasi_albert_graph(num_node, 5, seed=256)
    elif (graph_name == "NW"):
        G_test = nx.newman_watts_strogatz_graph(num_node, 2, 0.1, seed=256)
    else:
        print("请从BA, WS, RR, ER,NW中选择一种生成图")
    # 添加自环,解决 生成nmf特征矩阵时因为存在节点孤立而使得孤立节点没有特征出现的问题
    # G_test.add_edges_from((i, i) for i in range(num_node))
    num_edge = nx.number_of_edges(G_test)
    data = generate_features_from_nmf(from_networkx(G_test), NMF_feature_num, num_node)
    data.num_nodes = data.x.size(0)
    data.num_edges = data.edge_index.size(1)
    data.ac = calculate_ac_for_single(data, num_node)
    data.sparse_adj = get_sparse_adj(data)
    data.norm_adj = get_sparse_norm_adj(data)
    print(graph_name, "图生成完成，图节点数", data.num_nodes, "图的边数", data.num_edges, "生成的测试图特征矩阵维度:",
          data.x.shape)
    return data


def real_graph_gen(graph_name, NMF_feature_num=20):
    graph_datasets = ["Cora", "Citeseer", "Pubmed"]
    assert graph_name in graph_datasets, f"Please select from {graph_datasets}"
    dataset = Planetoid(root=f'../data/{graph_name}', name=graph_name)
    data = dataset[0]
    data = generate_features_from_nmf(data, NMF_feature_num, data.x.size(0))
    data.num_nodes = data.x.size(0)
    data.num_edges = data.edge_index.size(1)
    data.ac = calculate_ac_for_single(data, data.num_nodes)
    data.sparse_adj = get_sparse_adj(data)
    data.norm_adj = get_sparse_norm_adj(data)
    print(graph_name, "图生成完成，图节点数", data.num_nodes, "图的边数", data.num_edges, "生成的测试图特征矩阵维度:",
          data.x.shape)
    return data
