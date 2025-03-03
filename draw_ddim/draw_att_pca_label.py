
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 假设data是已经加载的DataFrame
file_name = ['attention_valid1_4213_label0.csv', 'attention_valid1_4213_label1.csv', 'attention_valid1_4213_label2.csv']
data_label0 = pd.read_csv(file_name[0]) # 你的CSV文件路径
data_label1 = pd.read_csv(file_name[1])
data_label2 = pd.read_csv(file_name[2])
datas = [data_label0, data_label1, data_label2]
attentions = [0, 1, 2, 3, 4, 5]

# 初始化PCA，目标是降至2维
pca = PCA(n_components=2)

# 设置画布和子图
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

# 分组处理数据，每6列一组
for i, attention in enumerate(attentions):
    pca_results = []
    for data in datas:
        # 选择当前组的列
        group_data = data.iloc[:, attention*6:attention*6+6]
        # 对当前组执行PCA
        pca_result = pca.fit_transform(group_data)
        pca_results.append(pca_result)

    # 注意力系数类型名称，按顺序
    attention_types = ['Label0', 'Label1', 'Label2']

    ax = axes[i // 3, i % 3] # 定位到正确的子图

    # 绘制PCA散点图，这里我们将绘制所有组，但在图例中区分不同的注意力系数
    colors = ['g', 'y', 'm'] # 为每组数据分配颜色
    for idx, pca_result in enumerate(pca_results):
        # 在指定的子图上绘制散点图
        ax.scatter(pca_result[:, 0], pca_result[:, 1], c=colors[idx], label=f'{attention_types[idx]}')

    ax.set_title(f'PCA Scatter Plot of Attention {attention}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()

plt.tight_layout()
# 如果你需要保存图像，取消注释下面的代码，并确保你的路径是正确的
plt.savefig('pca_png/' + "combined_pca_label.png", dpi=300, bbox_inches='tight')
plt.show()