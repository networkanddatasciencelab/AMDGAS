import torch
import torch.nn.functional as F
from Comm.GSACN_singlegraph import GSACN_Net, GCN_Net, GAT_Net, GIN_Net, SAGE_Net, GraphConv_Net, MLP_Net
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from Comm.utils import *
import warnings
import argparse
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--seed', type=int, default=4212, help='Random seed')
parser.add_argument('--train_name', type=str, default='Pubmed', choices=[
    "Cora", "Citeseer", "Pubmed"], help='Name of the dataset used for training')
parser.add_argument('--model_name', type=str, default="GSACN",
                    choices=['GSACN', 'GCN', 'GAT', 'GIN', 'SAGE', 'GraphConv', 'MLP'], help='Model name')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--epochs', type=int, default=501, help='Number of training epochs')
parser.add_argument('--nhid', type=int, default=16, help='Number of hidden units')
parser.add_argument('--nout', type=int, default=10, help='Number of output units')
parser.add_argument('--NMF_feature_num', type=int, default=20, help='Number of NMF features')
parser.add_argument('--is_draw', type=bool, default=False, help='Flag to determine if drawing should be enabled')

args = parser.parse_args()
seed = args.seed
train_name = args.train_name
model_name = args.model_name
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epochs
nhid = args.nhid
nout = args.nout
NMF_feature_num = args.NMF_feature_num
is_draw = args.is_draw

print(f"Running model: {model_name} on train_dataset: {train_name} ")
print(f"with learning rate: {lr}  with weight decay: {weight_decay}  with hidden units: {nhid}"
      f"  with seed: {seed}"
      f"  with epochs: {epochs}"
      f"  with NMF_feature_num: {NMF_feature_num}"
      f"  with is_draw: {is_draw}")

setup_seed(seed)


def softmax(x):
    # e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 从每行中减去最大值以提高数值稳定性
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)  # 确保正确广播


def make_modularity_matrix(adj):
    adj = adj * (torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(axis=0).unsqueeze(1)
    # degrees = torch.unsqueeze(degrees,1)
    mod = adj - degrees @ degrees.t() / adj.sum()
    return mod


def loss_modularity(r, bin_adj, mod, device):
    tmp = (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0])).to(device)
    bin_adj_nodiag = bin_adj * tmp
    return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_list = ['Cora', 'Citeseer', 'Pubmed']
# dataset_list = ['Cora']
# dataset_list = ['Cora', 'Citeseer']
data_dict = {}
for dataset_name in dataset_list:
    dataset = Planetoid(root=f'../data/{dataset_name}', name=dataset_name)
    data = dataset[0]
    if dataset_name =='Cora':
        data_cora = data
    data = generate_features_from_nmf(data, NMF_feature_num, data.x.size(0))
    data.num_nodes = data.x.size(0)
    data.num_edges = data.edge_index.size(1)
    data.ac = calculate_ac_for_single(data, data.num_nodes)
    data.sparse_adj = get_sparse_adj(data)
    data.norm_adj = get_sparse_norm_adj(data)
    data.bin_adj_train = (data.sparse_adj.to_dense() > 0).float() + torch.eye(data.num_nodes)
    data.target = make_modularity_matrix(data.bin_adj_train)
    data = data.to(device)
    data_dict[dataset_name] = data
    print(f'{dataset_name} data: {data}')

model = None
if model_name == 'GSACN':
    model = GSACN_Net(
        nfeat=data_dict[train_name].num_features,
        nhid1=nhid,
        nhid2=nhid,
        nhid3=nhid,
        nout=nout,
        useac=True,
        ac_shape=data_dict[train_name].ac.size(1)).to(device)
elif model_name == 'GCN':
    model = GCN_Net(
        nfeat=data_dict[train_name].num_features,
        nhid=nhid,
        nout=nout,
        dropout=0.2).to(device)
elif model_name == 'GAT':
    model = GAT_Net(
        nfeat=data_dict[train_name].num_features,
        nhid=nhid,
        nout=nout,
        dropout=0.2).to(device)

elif model_name == 'GIN':
    model = GIN_Net(
        nfeat=data_dict[train_name].num_features,
        nhid=nhid,
        nout=nout,
        dropout=0.2).to(device)

elif model_name == 'SAGE':
    model = SAGE_Net(
        nfeat=data_dict[train_name].num_features,
        nhid=nhid,
        nout=nout,
        dropout=0.2).to(device)
elif model_name == 'GraphConv':
    model = GraphConv_Net(
        nfeat=data_dict[train_name].num_features,
        nhid=nhid,
        nout=nout,
        dropout=0.2).to(device)
elif model_name == 'MLP':
    model = MLP_Net(
        nfeat=data_dict[train_name].num_features,
        nhid=nhid,
        nout=nout,
        dropout=0.2).to(device)
else:
    raise ValueError("Model name not recognized")

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

loss_dict = {item: [] for item in dataset_list}

for epoch in range(epochs):
    aux_loss = 0
    if model_name == 'GSACN':
        out, attention_weights1, aux_loss = model(data_dict[train_name])
    else:
        out = model(data_dict[train_name])
    r = F.softmax(out, dim=1)
    loss = loss_modularity(r, data_dict[train_name].bin_adj_train, data_dict[train_name].target, device)
    loss = -loss #+ 0.1*aux_loss
    # print(f'auxloss: {aux_loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 250 == 0:
        for data_name, data in data_dict.items():
            if model_name == 'GSACN':
                out, attention_weights1, aux_loss = model(data_dict[data_name])
                attention_weights1_numpy = attention_weights1.squeeze().cpu().detach().numpy()
                # np.set_printoptions(precision=3, suppress=True)
                # attention_fusion = attention_weights1_numpy.sum(axis=1)
                for dim in range(attention_weights1_numpy.shape[1]):
                    attention_fusion = attention_weights1_numpy[:, dim, :]
                    # s_a_f = softmax(attention_fusion)
                    s_a_f = attention_fusion
                    # np.savetxt(f"Cora_train_{data_name}_attention_fusion_dim{dim}.csv", s_a_f,
                    #            delimiter=" ", fmt='%.3f',
                    #            header='gcnout, sageout, gatout, linout, ginout, graphout', comments='')
                # for i in range(2):
                #     # print(f'attention_weights{i}: \n{attention_weights1_numpy[i].sum(axis=1)}') # 验证attention_weights1的正确性？
                #     print(f'node {i} attention_weights: {s_a_f[i]}')

            else:
                out = model(data_dict[data_name])
            r = F.softmax(out, dim=1)
            r = torch.softmax(100 * r, dim=1)
            loss_test = loss_modularity(r, data_dict[data_name].bin_adj_train, data_dict[data_name].target, device)
            loss_dict[data_name].append(loss_test.item())
            # print(f'epoch{epoch + 1}   data_name: {data_name}   GSNET value:{loss_test.item()}')
            print('epoch: {:04d}, data_name: {:10}, value: {:.4f}'.format(epoch + 1, data_name, loss_test.item()))
            if epoch==500:
                hardc= soft2hard(r,device)
                plot_cora_hard_assignment(data,hardc,max_nodes=data.num_nodes,use_tsne=True)
                plot_pyg_graph_with_communities(data,hardc,name = train_name+'trained'+model_name+'test in '+data_name+'seed='+str(seed),use_tsne=True)

# for data_name, loss in loss_dict.items():
#     plt.plot(loss, label=data_name)
#     # 格式化文件名
# filename = (f"model_{model_name}_train_{train_name}_lr_{lr}_wd_{weight_decay}"
#             f"_nhid_{nhid}_nout{nout}_seed_{seed}_epochs_{epochs}_NMF_{NMF_feature_num}.png")
#
# plt.legend()
# if is_draw:
#     # 显示图表
#     plt.show()
#
# # 保存图表到文件
# plt.savefig('./log/' + filename)  # 确保文件名不包含特殊字符或者非法路径字符
