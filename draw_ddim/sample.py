import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['legend.fontsize'] = 14  # 调整图例字体大小
plt.rcParams['font.size'] = 14        # 调整全局字体大小

def sample_lines_by_label(file_path, target_label, max_count,output_file_path):
    """
    逐行读取CSV文件，根据指定标签筛选行，并在达到最大计数时停止。

    Parameters:
    - file_path: str, CSV文件的路径。
    - target_label: str, 目标标签的值。
    - max_count: int, 达到此数量后停止读取的最大行数。
    """
    count = 0
    with open(file_path, 'r') as file, open(output_file_path, 'w') as out_file:
        for line in file:
            if line.strip().split(',')[-1] == target_label:
                # print(line.strip())  # 打印符合条件的行
                out_file.write(line)  # 将符合条件的行写入到输出文件中
                count += 1
                if count >= max_count:
                    break

# 文件路径和参数
file_path = '/home/imss/syn/SearchNet/tongji-0.8-jieduan4-abs/attention_valid1_4213_nojieduan.csv'  # 请替换为你的文件路径
# file_path = '/home/imss/syn/SearchNet/experiment_newsp/attention/attention_vaild1_sp_0.9_jieduan4_4212.csv'
target_label = '0.0'  # 对于标签为0的情况
out_file_path1 = './processed/'+target_label+'.csv'
max_count = 5000  # 最大计数值
# 调用函数
sample_lines_by_label(file_path, target_label, max_count,out_file_path1)

target_label = '1.0'  # 对于标签为0的情况
out_file_path2 = './processed/'+target_label+'.csv'
# 调用函数
sample_lines_by_label(file_path, target_label, max_count,out_file_path2)

target_label = '2.0'  # 对于标签为0的情况
out_file_path3 = './processed/'+target_label+'.csv'
# 调用函数
sample_lines_by_label(file_path, target_label, max_count,out_file_path3)
#_______________________________绘图
file_name = [out_file_path1, out_file_path2, out_file_path3]
data_label0 = pd.read_csv(file_name[0]) # 你的CSV文件路径
data_label1 = pd.read_csv(file_name[1])
data_label2 = pd.read_csv(file_name[2])
datas = [data_label0, data_label1, data_label2]


# 初始化PCA，目标是降至2维
pca = PCA(n_components=2)

# 设置画布和子图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))
labels=["cycle","house","crane"]
# 分组处理数据，每6列一组
attention_types = ['S-Dim1','S-Dim2','S-Dim3','S-Dim4','S-Dim5','S-Dim6']
colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 为每组数据分配颜色
i=0
for data in datas:
    pca_results = []
    for att in range(0,36,6):
        # 选择当前组的列
        group_data = data.iloc[:, att:att + 6]
        # 对当前组执行PCA
        pca_result = pca.fit_transform(group_data)
        pca_results.append(pca_result)
    ax = axes[i]
    #将3个motif的图注意力系数打印在三个图上
    for idx, pca_result in enumerate(pca_results):
        ax.scatter(pca_result[:, 0], pca_result[:, 1], c=colors[idx], label=f'{attention_types[idx]}')
        # ax.set_title(f'{labels[i]}')
        # ax.set_xlabel('Principal Component 1')
        # ax.set_ylabel('Principal Component 2')
        ax.legend()
    i = i+1
plt.tight_layout()
plt.savefig('./picture/' + file_path.split('/')[-1].split('.')[0] + "_dim.png", dpi=300, bbox_inches='tight')
plt.show()




