import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # 导入3D轴创建工具

file_name = 'attention_valid1_4213_nojieduan_label2.csv'
data = pd.read_csv(file_name) # 你的CSV文件路径

# 初始化PCA，目标是降至3维
pca = PCA(n_components=3)

# 存储PCA转换结果的列表
pca_results = []

# 分组处理数据，每6列一组
for i in range(0, 36, 6):
    # 选择当前组的列
    group_data = data.iloc[:, i:i+6]
    # 对当前组执行PCA
    pca_result = pca.fit_transform(group_data)
    pca_results.append(pca_result)

# 创建一个3D绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 注意力系数类型名称，按顺序
attention_types = ['GCNConv', 'SageConv', 'GatConv', 'Linear', 'GinConv', 'GraphConv']

# 为每组数据分配颜色
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# 绘制PCA散点图，这里我们将绘制所有组，但在图例中区分不同的注意力系数
for idx, pca_result in enumerate(pca_results):
    # 在3D空间中绘制散点图
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=colors[idx], label=f'{attention_types[idx]}')

# 设置图表标题和坐标轴标签
ax.set_title('3D PCA Scatter Plot of Attention Vectors')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# 显示图例
ax.legend(title='Attention Types')

# 显示图表
plt.show()