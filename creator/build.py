import os
from rich import print
import numpy as np
from PerfectMaze import is_perfect_maze


def load_generator(filepath: str):
    """生成器：逐行读取文件内容"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip('\n')  # 去除换行符并生成每行内容


def build_generator(filepath: str):
    """生成器：将文件内容转换为数据列表"""
    for line in load_generator(filepath):
        line_data = [int(element) for element in line if element != " "]
        yield line_data


def to_np_style_generator(filepath: str):
    """生成器：将数据转换为numpy数组（逐行生成）"""
    for line_data in build_generator(filepath):
        yield np.array(line_data)


def file_generator(path: str):
    """生成器：遍历目录中的所有文件（带正确计数）"""
    files = os.listdir(path)
    total = len(files)  # 获取总文件数
    for index, filename in enumerate(files, 1):  # 从1开始计数
        filepath = os.path.join(path, filename)
        filepath = filepath.replace("\\", "/")
        yield filepath, index, total  # 返回文件路径、当前索引和总数量


def process_mazes_generator(path: str):
    """生成器：处理所有迷宫文件并返回结果（附带正确统计）"""
    true_count = 0
    total = 0
    for filepath, current_index, total in file_generator(path):
        # 处理单个迷宫文件
        maze_generator = to_np_style_generator(filepath)
        maze_data = np.array(list(maze_generator))  # 转换为完整数组
        result = is_perfect_maze(maze_data)
        
        if result:
            true_count += 1
        
        # 返回单个文件的结果和当前进度
        yield {
            "file": filepath,
            "is_perfect": result,
            "current": current_index,  # 当前处理的文件序号
            "total": total  # 总文件数量
        }
    
    # 所有文件处理完毕后，返回统计结果
    yield {
        "summary": True,
        "true_count": true_count,
        "total_count": total,
        "percentage": (true_count / total) * 100 if total > 0 else 0
    }


if __name__ == "__main__":
    # path = "samples"
    path = "generated_mazes"

    for item in process_mazes_generator(path):
        if "summary" in item:
            # 输出统计结果
            print("\n" + "="*50)
            print(f"总文件数: {item['total_count']}")
            print(f"完美迷宫数量: {item['true_count']}")
            print(f"占比: {item['percentage']:.2f}%")
            print("="*50)
        else:
            # 输出单个文件结果和进度
            print(f"文件: {item['file']} -> 完美迷宫: {item['is_perfect']} "
                  f"({item['current']}/{item['total']})")
    