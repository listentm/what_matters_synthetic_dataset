import json  # 导入json模块，用于处理JSON数据
import os  # 导入os模块，用于文件和操作系统相关操作

import yaml  # 导入yaml模块，用于处理YAML数据


def load_json(file_path):
    # 打开指定路径的文件，以只读模式读取，并使用utf-8编码
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)  # 使用json模块加载文件内容并返回


def save_json(data, file_path):
    # 打开指定路径的文件，以写模式写入，并使用utf-8编码
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  # 使用json模块将数据写入文件，确保非ASCII字符不被转义，并格式化输出


def load_yaml(file_path):
    # 打开指定路径的文件，以只读模式读取
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # 使用yaml模块安全地加载文件内容
    return config  # 返回加载的配置


def check_and_rename_file(path):
    """
    检查给定路径的文件是否存在。如果存在，通过在文件扩展名前添加一个数字(n)来重命名它。

    参数:
    - path (str): 原始文件路径。

    返回:
    - str: 新的文件路径，如果文件存在则附加一个数字。
    """
    if os.path.exists(path):  # 如果文件存在
        # 将路径分割为目录、文件名和扩展名
        directory, filename = os.path.split(path)
        base, extension = os.path.splitext(filename)

        # 初始化计数器
        counter = 1

        # 生成带有计数器的新文件名，直到它是唯一的
        new_filename = f"{base}({counter}){extension}"
        new_path = os.path.join(directory, new_filename)
        while os.path.exists(new_path):  # 如果新路径也存在，继续增加计数器
            counter += 1
            new_filename = f"{base}({counter}){extension}"
            new_path = os.path.join(directory, new_filename)

        return new_path  # 返回新的文件路径
    else:
        # 如果文件不存在，返回原始路径
        return path