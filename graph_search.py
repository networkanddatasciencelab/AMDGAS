from math import ceil
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGraphConv, GCNConv, dense_mincut_pool, knn_interpolate
from torch_geometric.utils import to_dense_adj, to_dense_batch
import numpy as np
import os.path as osp
import time
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import MoleculeNet
from spmotif_dataset import SPMotif, GroupAssigner
from GSACN import GSACN_Net, setup_seed
from sklearn.metrics import roc_auc_score
import argparse
import rdkit

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--seed', type=int, default=4211, help='Random seed')
parser.add_argument('--jieduan', type=int, default=4, help='Size of each dimension')
parser.add_argument('--useac', type=bool, default=True, help='if you use assortativity attention')
parser.add_argument('--device', type=str, default='0', help='cuda 0 or cuda 1?')
parser.add_argument('--dataset_str', type=str, default="sp", help='datasets name')
parser.add_argument('--sp_b', type=str, default='0.7', help='Size of each dimension')
parser.add_argument('--epochs', type=int, default=80, help='train epochs')
parser.add_argument('--draw_p', type=bool, default=False, help='if you want to save the p')
parser.add_argument('--ratio', type=float, default=0.1, help='few-shot learning')
args = parser.parse_args()
epochs = args.epochs
seed = args.seed
useac = args.useac
jieduan = args.jieduan
device = args.device
setup_seed(seed)
draw_p = args.draw_p
ratio = args.ratio
print(useac)
print(jieduan)
# generator = torch.Generator().manual_seed(seed)
batch_size = 32
alpha = 1
beta = 1
dataset_str = args.dataset_str
step = int(round(1.0 / ratio))
if (dataset_str == "sp"):
    root = '/home/imss/syn/SearchNet/data/graces/SPMotif-' + args.sp_b
    print("dataset:" + root)
    transform = GroupAssigner(n_groups=2)
    train_dataset = SPMotif(root, mode='train', transform=transform)
    train_dataset = train_dataset[::step]
    val_dataset = SPMotif(root, mode='val', transform=transform)
    test_dataset = SPMotif(root, mode='test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # , generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
elif (dataset_str == "hiv"):
    root = '/home/imss/syn/SearchNet/data/HIV'
    dataset = MoleculeNet(root=root, name="HIV")  # .shuffle()
    n = (len(dataset) + 9) // 10
    test_dataset = dataset[:n]
    val_dataset = dataset[n:2 * n]
    train_dataset = dataset[2 * n:]
    train_dataset = train_dataset[::step]
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # , generator=generator)
elif (dataset_str == "bace"):
    root = '/home/imss/syn/SearchNet/data/BACE'
    dataset = MoleculeNet(root=root, name="BACE")  # .shuffle()
    n = (len(dataset) + 9) // 10
    test_dataset = dataset[:n]
    val_dataset = dataset[n:2 * n]
    train_dataset = dataset[2 * n:]
    train_dataset = train_dataset[::step]
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # , generator=generator)

elif (dataset_str == "sider"):
    root = '/home/imss/syn/SearchNet/data/SIDER'
    dataset = MoleculeNet(root=root, name="SIDER")  # .shuffle()
    n = (len(dataset) + 9) // 10
    test_dataset = dataset[:n]
    val_dataset = dataset[n:2 * n]
    train_dataset = dataset[2 * n:]
    train_dataset = train_dataset[::step]
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # , generator=generator)
else:
    print("数据集未知")
    exit()
print("小样本比例"+str(ratio),'训练集图数量'+str(len(train_dataset)))
device = torch.device('cuda:' + device if torch.cuda.is_available() else 'cpu')
if dataset_str == "sp":
    model = GSACN_Net(
        nfeat=train_dataset.num_features,
        nhid1=20,
        nhid2=20,
        nhid3=16,
        nout=train_dataset.num_classes,
        jieduan=jieduan,
        useac=useac,
        h_concat_ac=False).to(device)
elif dataset_str == "bace" or dataset_str == "sider" or dataset_str == "hiv":
    model = GSACN_Net(  # nfeat=train_dataset.num_features,
        nfeat=dataset.num_features,
        nhid1=20,
        nhid2=20,
        nhid3=16,
        nout=dataset.num_classes,
        jieduan=jieduan,
        useac=useac,
        h_concat_ac=False).to(device)
# 两段ods都加上gcn+virembed
# model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  # 0.01


def dropout_and_add_edges(edge_index, num_nodes, drop_ratio=0.05, add_ratio=0.05):
    # 随机丢弃边
    num_edges = edge_index.size(1)
    num_edges_to_drop = int(num_edges * drop_ratio)
    indices_to_drop = np.random.choice(num_edges, num_edges_to_drop, replace=False)
    mask = torch.ones(num_edges, dtype=bool)
    mask[indices_to_drop] = False
    edge_index_dropped = edge_index[:, mask]

    # 随机添加边
    num_edges_to_add = int(num_edges * add_ratio)
    added_edges = []
    while len(added_edges) < num_edges_to_add:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and (u, v) not in added_edges and (v, u) not in added_edges:
            added_edges.append((u, v))
    added_edges = torch.tensor(added_edges).t().long().to(edge_index_dropped.device)

    # 合并原有边和新增的边
    edge_index_final = torch.cat([edge_index_dropped, added_edges], dim=1)
    return edge_index_final


def process_batch_edges(batch_data, batch, drop_ratio=0.05, add_ratio=0.05):
    batch_size = batch.max().item() + 1  # 确定批次中图的数量
    edge_index = batch_data.edge_index
    num_nodes = batch_data.num_nodes

    # 存储处理后的边
    new_edge_indices = []

    for i in range(batch_size):
        # 获取当前图的节点索引
        mask = batch == i
        num_nodes_in_graph = mask.sum().item()

        # 获取当前图的边
        node_indices = mask.nonzero(as_tuple=False).view(-1)
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]

        # 调整边索引，使其从0开始
        _, edge_index_sub = torch.unique(edge_index_sub, return_inverse=True)
        edge_index_sub = edge_index_sub.view(edge_index_sub.shape)

        # 对当前图的边进行处理
        processed_edges = dropout_and_add_edges(edge_index_sub, num_nodes_in_graph, drop_ratio, add_ratio)

        # 将处理后的边重新映射到全局节点索引
        processed_edges_global = node_indices[processed_edges]

        new_edge_indices.append(processed_edges_global)

    # 合并所有图的处理后的边
    new_edge_index = torch.cat(new_edge_indices, dim=1)
    return new_edge_index


def train():
    model.train()
    loss_all = 0
    loss_class_all = 0
    loss_dpp_all = 0
    correct = 0
    bce_loss = nn.BCEWithLogitsLoss()
    for data in train_loader:
        data = data.to(device)
        # data.edge_index = dropout_and_add_edges(data.edge_index, data.num_nodes, 0.05, 0)
        optimizer.zero_grad()
        out, loss_dpp1, loss_dpp2, _, _ = model(data)
        if dataset_str == "sp":
            loss_class = alpha * F.nll_loss(F.log_softmax(out, dim=1), data.y)
        elif dataset_str == "sider":
            loss_class = alpha * bce_loss(out, data.y)
        else:
            loss_class = alpha * F.nll_loss(F.log_softmax(out, dim=1), data.y.view(-1).long())
        loss_dpp = loss_dpp1 + loss_dpp2
        loss = loss_class  # - 0.5*loss_dpp
        loss.backward()
        loss_all += loss.item()
        loss_dpp_all += loss_dpp  # .item()
        loss_class_all += loss_class.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset), loss_class_all / len(train_loader.dataset), loss_dpp_all / len(
        train_loader.dataset)


@torch.no_grad()
def validate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    all_nodes_attention1_with_label = []  # 用于收集所有节点的注意力和标签
    for data in loader:
        data = data.to(device)
        output, loss_dpp1, loss_dpp2, attention1, attention2 = model(data)
        loss_dpp = loss_dpp1 + loss_dpp2
        loss = F.nll_loss(F.log_softmax(output, dim=1), data.y)  # -0.5*loss_dpp
        total_loss += loss.item()
        pred = output.max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
        if draw_p:
            # 将attention1重塑为600*36
            attention_reshaped = attention1.view(-1, 36)
            # 获取每个节点所属图的标签
            node_labels = data.y[data.batch]  # 使用data.batch找到每个节点所属的图
            # 将标签扩展为[600, 1]以匹配attention_reshaped的形状
            labels_expanded = node_labels.view(-1, 1)
            # 将注意力系数和标签拼接
            attention_with_label = torch.cat((attention_reshaped, labels_expanded.float()), dim=1)
            # 收集所有节点的注意力系数和标签
            all_nodes_attention1_with_label.append(attention_with_label.cpu())

    return total_loss / len(loader.dataset), correct / len(loader.dataset), all_nodes_attention1_with_label


@torch.no_grad()
def validate_molecule(loader):
    model.eval()
    total_loss = 0
    correct = 0
    pred_scores = []  # 存储预测分数
    true_labels = []  # 存储真实标签
    bce_loss = nn.BCEWithLogitsLoss()
    for data in loader:
        data = data.to(device)
        output, loss_dpp1, loss_dpp2, _, _ = model(data)
        loss_dpp = loss_dpp1 + loss_dpp2
        if dataset_str == "sider":
            loss = bce_loss(output, data.y)
            total_loss += loss.item()
            # 保存预测概率和真实标签
            pred_prob = torch.sigmoid(output).cpu().numpy()  # 多标签分类使用sigmoid
            true_labels.append(data.y.cpu().numpy())
            pred_scores.append(pred_prob)

        else:
            loss = F.nll_loss(F.log_softmax(output, dim=1), data.y.view(-1).long())  # -0.5*loss_dpp
            total_loss += loss.item()
            pred = output.max(dim=1)[1]
            correct += int(pred.eq(data.y.view(-1)).sum())
            # 保存预测概率和真实标签
            # 假设output是logits; 对于二分类问题，使用sigmoid函数获取正类的预测概率
            pred_prob = torch.sigmoid(output[:, 1])  # 适应你的模型输出
            pred_scores.extend(pred_prob.tolist())
            true_labels.extend(data.y.tolist())
    if dataset_str == "sider":
        roc_auc_scores = []
        true_labels = np.concatenate(true_labels, axis=0)
        pred_scores = np.concatenate(pred_scores, axis=0)
        for i in range(true_labels.shape[1]):  # 遍历每个标签
            roc_auc = roc_auc_score(true_labels[:, i], pred_scores[:, i])
            roc_auc_scores.append(roc_auc)
        roc_auc = np.mean(roc_auc_scores)
    else:
        roc_auc = roc_auc_score(true_labels, pred_scores)  # 计算ROC-AUC分数
    return total_loss / len(loader.dataset), correct / len(loader.dataset), roc_auc


best_test_loss = float('inf')
best_test_acc = 0.0
best_val_acc = 0.0
best_test_roc = 0.0
best_val_roc = 0.0
best_epoch = 0
best_val_attention1=[]
best_test_attention1=[]
start_train_time = time.time()
for epoch in range(1, epochs):  # 200 epochs
    train_loss, loss_class, loss_dpp = train()
        #——————————————————————————————————————————————————————————————
    # if dataset_str == "sp":
    #     val_loss, val_acc, attention_valid = validate(val_loader)
    #     test_loss, test_acc, attention_test = validate(test_loader)
    #     print(
    #         f'Epoch: {epoch}, Train Loss: {train_loss:.4f},class Loss: {loss_class:.4f},dpp Loss: {loss_dpp:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f},Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    # else:
    #     val_loss, val_acc, val_roc = validate_molecule(val_loader)
    #     test_loss, test_acc, test_roc = validate_molecule(test_loader)
    #     print(
    #         f'Epoch: {epoch}, Train Loss: {train_loss:.4f},class Loss: {loss_class:.4f},dpp Loss: {loss_dpp:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f},Val Roc:{val_roc:.4f},Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f},Test Roc:{test_roc:.4f}')
    # if dataset_str == "sp":
    #     if test_acc > best_test_acc:
    #         best_test_loss = test_loss
    #         best_test_acc = test_acc
    #         best_epoch = epoch
    #         best_test_attention1 = attention_test
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         best_val_attention1 = attention_valid
    #
    # else:
    #     if test_roc > best_test_roc:
    #         best_test_loss = test_loss
    #         best_test_acc = test_acc
    #         best_test_roc = test_roc
    #         best_epoch = epoch
    #     if val_roc > best_val_roc:
    #         best_val_roc = val_roc
      #————————————————————————————————————————————————
    # if test_roc > best_test_roc:
    #     best_test_loss = test_loss
    #     best_test_acc = test_acc
    #     best_test_roc = test_roc
    #     best_epoch = epoch

    # best_test_attention1 = attention_test1
    # best_test_attention2 = attention_test2

    # if val_acc > best_val_acc:
    # best_val_attention1 = attention_valid1
    # best_val_attention2 = attention_valid2
end_train_time = time.time()
print(f"train() run time: {end_train_time - start_train_time:.4f} s")
# if dataset_str == "sp":
#     print(
#         f'The Best Epoch: {best_epoch:03d},Test Loss: {best_test_loss:.4f}, Test Acc: {best_test_acc:.4f}')  # ,Test ROC:{best_test_roc:.4f}')
# else:
#     print(
#         f'The Best Test Epoch: {best_epoch:03d},Test Loss: {best_test_loss:.4f}, Test Acc: {best_test_acc:.4f}, Test ROC: {best_test_roc:.4f},Val ROC: {best_val_roc:.4f}')

# 保存注意力向量——————————————————————————————————————————————————————————————Start
if draw_p:
    all_data_df = pd.DataFrame()

    for tensor in best_val_attention1:
        # 将每个张量转换为NumPy数组，然后创建一个DataFrame
        df = pd.DataFrame(tensor.numpy())
        # 将当前DataFrame添加到总的DataFrame中
        all_data_df = pd.concat([all_data_df, df], ignore_index=True)

    # 现在all_data_df包含了所有数据，将其保存为CSV文件
    all_data_df.to_csv(
        '/home/imss/syn/SearchNet/experiment_newsp/attention/attention_vaild1_sp_' + args.sp_b + '_jieduan'+str(jieduan) +'_'+ str(seed) + '.csv',index=False)

    all_data_df = pd.DataFrame()

    for tensor in best_test_attention1:
        # 将每个张量转换为NumPy数组，然后创建一个DataFrame
        df = pd.DataFrame(tensor.numpy())
        # 将当前DataFrame添加到总的DataFrame中
        all_data_df = pd.concat([all_data_df, df], ignore_index=True)

    # 现在all_data_df包含了所有数据，将其保存为CSV文件
    all_data_df.to_csv(
        '/home/imss/syn/SearchNet/experiment_newsp/attention/attention_test1_sp_' + args.sp_b + '_jieduan'+str(jieduan) +'_'+ str(seed) + '.csv', index=False)
# 保存注意力向量——————————————————————————————————————————————————————————————END


# best_val_acc = test_acc = 0
# best_val_loss = float('inf')
# patience = start_patience = 50
# for epoch in range(1, 1000):
#     train_loss = train(epoch)
#     _, train_acc = test(train_loader)
#     val_loss, val_acc = test(val_loader)
#     if val_loss < best_val_loss:
#         test_loss, test_acc = test(test_loader)
#         best_val_acc = val_acc
#         patience = start_patience
#         best_epoch =  epoch
#     else:
#         patience -= 1
#         if patience == 0:
#             break
#     print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, '
#           f'Train Acc: {train_acc:.3f}, Val Loss: {val_loss:.3f}, '
#           f'Val Acc: {val_acc:.3f}, Test Loss: {test_loss:.3f}, '
#           f'Test Acc: {test_acc:.3f}')
#
# print(f'The Best Epoch: {best_epoch:03d},Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')
