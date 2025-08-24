# PerfectMaze

[English](https://github.com/starwindv/PerfectMaze/blob/main/readme.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/starwindv/PerfectMaze/blob/main/LICENSE)
![Python Version](https://img.shields.io/badge/python-3.10+-orange)

PerfectMaze 是一个用于生成完美迷宫的 Python 库，完美迷宫指没有环路且所有路径都连通的迷宫。当前版本实现了基于希尔伯特曲线（Hilbert Curve）的迷宫生成算法。

## 主要功能

- 🧩 基于希尔伯特曲线的迷宫生成
- 📊 迷宫可视化（使用 Matplotlib）
- 💾 迷宫的保存与加载功能
- 🧭 迷宫求解算法（DFS）
- ✅ 完美迷宫验证功能

## 安装

```bash
pip install PerfectMaze
```

## 快速开始

```python
from PerfectMaze.HilbertMaze import HilbertMaze

# 创建希尔伯特迷宫（阶数为3）
maze_gen = HilbertMaze(order=3)

# 生成迷宫
maze, start, end = maze_gen.generate()

# 显示迷宫
maze_gen.display()

# 求解并显示迷宫路径
maze_gen.solve_and_display()
```

## 示例输出

![Hilbert Maze](https://github.com/StarWindv/PerfectMaze/blob/main/assets/order6_1.png)
*阶数为6的希尔伯特迷宫示例*

## API参考

### `HilbertMaze` 类

#### 初始化参数：
```python
def __init__(
    order: int = 3,
    full_maze: bool = False,
    int_length: int = 16,
    info: bool = False,
    save_dir: str = ".",
    save_name: str = "Hilbert_maze"
): ...
```

#### 主要方法：
- `generate()`: 生成迷宫，返回(maze_grid, start_coord, end_coord)
- `display(overwrite=False)`: 可视化迷宫
- `solve_and_display(overwrite=False)`: 求解并可视化迷宫路径
- `save_maze(**kwargs)`: 保存迷宫到文件
- `load_maze(*args)`: 从文件加载迷宫

### 工具函数
```python
from PerfectMaze import is_perfect_maze
maze_grid = ...
# 验证迷宫是否为完美迷宫
is_perfect = is_perfect_maze(maze_grid)
```

## 依赖项

- Python 3.10+
- rich
- numpy
- matplotlib

## 贡献

欢迎贡献！请通过 GitHub Issues 提交问题或建议，通过 Pull Requests 提交代码改进。

## 许可证

本项目采用 [MIT](https://github.com/starwindv/PerfectMaze/blob/main/LICENSE) 许可证。