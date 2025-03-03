import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 修改后的函数，只提取前60个epochs的Test Acc
def extract_test_acc(file_path):
    test_accs = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Test Acc" in line:
                parts = line.split(',')
                for part in parts:
                    if "Test Acc" in part:
                        test_acc = float(part.split(':')[-1].strip())
                        test_accs.append(test_acc)
                if len(test_accs) == 30:  # 只要求前60个epochs
                    break
    return test_accs


folders = ['../experiment_newsp/sp_0.8_jieduan4', '../experiment_newsp/sp_0.8_jieduan6', '../experiment_newsp/sp_0.8_noac_jieduan4', '../experiment_newsp/sp_0.8_noac_jieduan6']

data = []

for folder in folders:
    folder_data = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        test_accs = extract_test_acc(file_path)
        folder_data.append(test_accs)
    df = pd.DataFrame(folder_data).T  # 转置以使每行是一个epoch
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    data.append(pd.DataFrame({'Epoch': range(1, 31), 'Test Acc': mean, 'Std': std, 'Model': folder}))

# 合并数据
all_data = pd.concat(data)

# 绘图
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Test Acc', hue='Model', data=all_data, ci=None)

# 添加阴影
for model in folders:
    model_data = all_data[all_data['Model'] == model]
    plt.fill_between(model_data['Epoch'], model_data['Test Acc'] - model_data['Std'], model_data['Test Acc'] + model_data['Std'], alpha=0.2)

plt.title('Test Accuracy over Epochs (1-60)')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
