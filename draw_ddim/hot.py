import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
# file_path = '/home/imss/syn/SearchNet/tongji-0.9-jieduan4-abs/attention_test1_4213.csv'  # 请替换为你的文件路径
file_path = '/home/imss/syn/SearchNet/experiment_newsp/attention/attention_vaild1_sp_0.9_jieduan4_4215.csv'
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
data_label0 = pd.read_csv(file_name[0],header=None) # 你的CSV文件路径
data_label1 = pd.read_csv(file_name[1],header=None)
data_label2 = pd.read_csv(file_name[2],header=None)
datas = [data_label0, data_label1, data_label2]
column_labels = ['GCNConv', 'SAGECOnv', 'GATConv', 'GINConv', 'Linear', 'GraphConv']
# 设置画布和子图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
labels=["cycle","house","crane"]
# 分组处理数据，每6列一组
attention_types = ['D-Dim1','D-Dim2','D-Dim3','D-Dim4','D-Dim5','D-Dim6']
for label,data in enumerate(datas):
    dims = []
    for att in range(0,36,6):
        # 选择当前组的列
        group_data = data.iloc[:, att:att + 6]
        dims.append(group_data)
    for df in dims:
        df.columns = column_labels
    average_vectors = pd.DataFrame([df.mean() for df in dims])
    means = pd.DataFrame({f'dim{i + 1}': dim.mean() for i, dim in enumerate(dims)})
    sns.heatmap(means, ax=axes[label], fmt='.2f', cmap='viridis')
    # axes[label].set_title('Average P Values Heatmap for Means1')
    axes[label].set_xlabel('Design Dimension')
    axes[label].set_ylabel('Graph-Architecture Mapping')

plt.tight_layout()
# plt.savefig('./picture/hot.png', dpi=300, bbox_inches='tight')
plt.show()
















