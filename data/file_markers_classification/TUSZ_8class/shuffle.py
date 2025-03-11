import os
import random

# 定义输入和输出路径
input_folder = r"E:\hy\eeg-gnn-ssl-main\data\file_markers_classification\TUSZ_8class"
output_folder = r"E:\hy\eeg-gnn-ssl-main\data\file_markers_classification"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有txt文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # 读取文件中的所有行
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 打乱行的顺序
        random.shuffle(lines)

        # 将打乱后的内容写入新文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

print("所有文件已处理完毕！")
