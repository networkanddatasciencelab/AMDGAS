import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 假设data是已经加载的DataFrame
file_name = 'attention_valid1_4213_label0.csv'
data = pd.read_csv(file_name)  # 你的CSV文件路径

# 初始化PCA，目标是降至2维
pca = PCA(n_components=2)

# 存储PCA转换结果的列表
pca_results = []

# 分组处理数据，每6列一组
for att in range(0, 6):
    # __________________不同操作子为一组，即0，6，12，18，24，30为一个向量
    # 为当前的attention构造列索引
    columns = [att + i * 6 for i in range(6)]
    # 使用loc或iloc通过列索引选取数据
    group_data = data.iloc[:, columns]
    # __________________不同att向量为一组，即0，1，2，3，4，5为一个向量
    # # 选择当前组的列
    # group_data = data.iloc[:, i:i+6]
    # 对当前组执行PCA
    pca_result = pca.fit_transform(group_data)
    pca_results.append(pca_result)

# 注意力系数类型名称，按顺序
# attention_types = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6']
attention_types = ['GCNConv', 'SAGEConv', 'GATConv', 'Linear', 'GINConv', 'GraphConv']
# 绘制PCA散点图，这里我们将绘制所有组，但在图例中区分不同的注意力系数
colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 为每组数据分配颜色
for idx, pca_result in enumerate(pca_results):
    # 为了图例正确显示，我们使用label参数来指定每组的注意力系数名称
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors[idx], label=f'{attention_types[idx]}')
plt.title('PCA Scatter Plot of Operator Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig('pca_png/' + file_name.split('.')[0] + "_op.png", dpi=300, bbox_inches='tight')
plt.show()
